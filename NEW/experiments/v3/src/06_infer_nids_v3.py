import argparse
import datetime as dt
import importlib.util
import os
from pathlib import Path

import torch
from torch.utils.data import DataLoader

EXPERIMENT_SRC_DIR = Path(__file__).resolve().parent
EXPERIMENT_DIR = EXPERIMENT_SRC_DIR.parent
NEW_DIR = Path(__file__).resolve().parents[3]
SRC_DIR = NEW_DIR / "src"


def load_module_from_path(module_name, module_path):
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_source_module(module_name, filename):
    return load_module_from_path(module_name, SRC_DIR / filename)


st_data_loader = load_source_module("st_data_loader", "02_st_data_loader.py")
stt_architecture = load_module_from_path("stt_architecture_v3", EXPERIMENT_SRC_DIR / "03_stt_architecture_v3.py")
v3_train = load_module_from_path("train_multitask_nids_v3", EXPERIMENT_SRC_DIR / "05_train_multitask_nids_v3.py")

SpatioTemporalNIDSDataset = st_data_loader.SpatioTemporalNIDSDataset
SpatioTemporalTransformer = stt_architecture.SpatioTemporalTransformer
NIDSMultiTaskModel = stt_architecture.NIDSMultiTaskModel
DownstreamNIDSDataset = v3_train.VariantDownstreamNIDSDataset

CAT_VOCABS = [256, 256, 256, 65536, 65536, 256, 256, 256, 256]
DEFAULT_CHECKPOINT = str(EXPERIMENT_DIR / "checkpoints" / "nids_multitask_05_v3_full" / "nids_multitask_best.pt")
DEFAULT_STATS = str(NEW_DIR / "nids_normalization_stats.json")
DEFAULT_DATA_ROOT = str(NEW_DIR / "data" / "nids_src_grouped")
DEFAULT_SEQ_LEN = 32
DEFAULT_STRIDE = 16


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run v3 downstream NIDS inference and print human-readable alerts."
    )
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT, help="Path to the v3 downstream checkpoint.")
    parser.add_argument(
        "--demo-scenario",
        choices=["known_attack", "unknown_attack", "benign"],
        default=None,
        help=(
            "Apply thesis-demo defaults backed by real split flows. "
            "known_attack uses the standard test split with attack-only filtering, "
            "unknown_attack uses test_ood with attack-only filtering, and benign uses the standard test split with benign filtering. "
            "Add --status-filter manually if you want a stricter predicted-status demo."
        ),
    )
    parser.add_argument(
        "--split",
        default="test",
        help="Dataset split to score, for example validation, test, or test_ood (held-out-family benchmark split name kept for compatibility).",
    )
    parser.add_argument("--batch-size", type=int, default=512, help="Batch size for inference.")
    parser.add_argument(
        "--current-threshold",
        type=float,
        default=None,
        help="Optional override for the current-attack threshold stored in the checkpoint.",
    )
    parser.add_argument(
        "--known-threshold",
        type=float,
        default=None,
        help="Optional override for the known-family gate threshold stored in the checkpoint.",
    )
    parser.add_argument(
        "--future-threshold",
        type=float,
        default=None,
        help="Optional scalar override applied to every future-warning horizon threshold stored in the checkpoint.",
    )
    parser.add_argument(
        "--ood-threshold",
        type=float,
        default=None,
        help="Optional override for the novelty or unknown-risk threshold stored in the checkpoint.",
    )
    parser.add_argument(
        "--seq-len",
        type=int,
        default=None,
        help="Sequence length used for the dataset. Defaults to the checkpoint setting.",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=None,
        help="Stride used for the dataset. Defaults to the checkpoint setting.",
    )
    parser.add_argument("--clip-value", type=float, default=5.0, help="Continuous feature clamp value.")
    parser.add_argument(
        "--future-horizons-minutes",
        nargs="+",
        type=int,
        default=None,
        help=(
            "Deprecated compatibility flag. Inference uses the future horizons stored in the checkpoint; "
            "if provided, the values must exactly match that checkpoint."
        ),
    )
    parser.add_argument(
        "--future-horizon-minutes",
        type=int,
        default=None,
        help=(
            "Deprecated single-horizon compatibility flag. Inference uses the future horizons stored in the "
            "checkpoint; if provided, the value must exactly match that checkpoint."
        ),
    )
    parser.add_argument("--max-sequences", type=int, default=32, help="Maximum number of sequences to print.")
    parser.add_argument(
        "--dataset-max-sequences",
        type=int,
        default=None,
        help="Optional cap on how many sequences to load before filtering.",
    )
    parser.add_argument(
        "--only-attacks",
        action="store_true",
        help="Print only windows whose ground-truth current label is attack.",
    )
    parser.add_argument(
        "--status-filter",
        choices=["benign", "novelty_watch", "known_attack", "unknown_attack_warning"],
        default=None,
        help="Optional predicted-status filter applied before printing sequences.",
    )
    parser.add_argument(
        "--allow-split-fallback",
        action="store_true",
        help="Allow falling back to the test split when the requested split is missing.",
    )
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader worker count for inference.")
    return parser.parse_args()


def apply_demo_scenario_defaults(args):
    if args.demo_scenario is None:
        return args

    if args.demo_scenario == "known_attack":
        args.split = "test"
        args.only_attacks = True
        return args

    if args.demo_scenario == "unknown_attack":
        args.split = "test_ood"
        args.only_attacks = True
        return args

    args.split = "test"
    args.only_attacks = False
    if args.status_filter is None:
        args.status_filter = "benign"
    return args


def resolve_split_path(split_name, allow_fallback=False):
    requested_path = os.path.join(DEFAULT_DATA_ROOT, split_name)
    if os.path.exists(requested_path):
        return requested_path, split_name

    if not allow_fallback:
        raise FileNotFoundError(
            f"Requested split '{split_name}' does not exist under {DEFAULT_DATA_ROOT}. "
            "Pass --allow-split-fallback to fall back to the test split."
        )

    fallback_path = os.path.join(DEFAULT_DATA_ROOT, "test")
    print(f"Requested split '{split_name}' is not available. Falling back to 'test'.")
    return fallback_path, "test"


def format_timestamp(timestamp_ms):
    timestamp_ms = int(timestamp_ms)
    return dt.datetime.utcfromtimestamp(timestamp_ms / 1000.0).strftime("%Y-%m-%d %H:%M:%S UTC")


def checkpoint_uses_ood_head(checkpoint):
    if "use_ood_head" in checkpoint:
        return bool(checkpoint["use_ood_head"])
    model_state_dict = checkpoint.get("model_state_dict", {})
    return any(key.startswith("unknown_attack_head.") for key in model_state_dict)


def load_checkpoint(checkpoint_path, device):
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    thresholds = checkpoint.get(
        "thresholds",
        {"current": 0.50, "known": 0.55, "future": 0.50, "ood": 0.50},
    )
    known_attack_labels = checkpoint.get("known_attack_labels", [])
    pseudo_zero_day_families = checkpoint.get("pseudo_zero_day_families", [])
    future_horizons_minutes = v3_train.normalize_future_horizons_minutes(
        checkpoint.get("future_horizons_minutes"),
        checkpoint.get("future_horizon_minutes", 5),
    )
    thresholds = dict(thresholds)
    thresholds["future"] = v3_train.normalize_future_thresholds(
        thresholds.get("future"),
        future_horizons_minutes,
    )
    future_task_enabled = bool(checkpoint.get("future_task_enabled", True))
    checkpoint_seq_len = int(checkpoint.get("seq_len", DEFAULT_SEQ_LEN))
    checkpoint_stride = int(checkpoint.get("stride", DEFAULT_STRIDE))
    reconstruction_calibration = checkpoint.get("reconstruction_calibration")
    use_reconstruction_hybrid_ood = bool(
        checkpoint.get("use_reconstruction_hybrid_ood", False) and reconstruction_calibration is not None
    )
    unknown_risk_score_mode = str(
        checkpoint.get(
            "unknown_risk_score_mode",
            v3_train.resolve_unknown_risk_score_mode(
                None,
                use_reconstruction_hybrid_ood,
                bool(checkpoint.get("unknown_head_active", checkpoint_uses_ood_head(checkpoint))),
            ),
        )
    )
    reconstruction_validation_mae_mask_ratio = float(
        checkpoint.get("reconstruction_validation_mae_mask_ratio", 0.30)
    )
    reconstruction_validation_mfm_mask_ratio = float(
        checkpoint.get("reconstruction_validation_mfm_mask_ratio", 0.10)
    )
    reconstruction_train_mae_mask_ratio = float(checkpoint.get("reconstruction_train_mae_mask_ratio", 0.10))
    reconstruction_train_mfm_mask_ratio = float(checkpoint.get("reconstruction_train_mfm_mask_ratio", 0.00))
    return (
        checkpoint,
        thresholds,
        known_attack_labels,
        pseudo_zero_day_families,
        future_horizons_minutes,
        future_task_enabled,
        checkpoint_seq_len,
        checkpoint_stride,
        reconstruction_calibration,
        use_reconstruction_hybrid_ood,
        reconstruction_validation_mae_mask_ratio,
        reconstruction_validation_mfm_mask_ratio,
        reconstruction_train_mae_mask_ratio,
        reconstruction_train_mfm_mask_ratio,
        unknown_risk_score_mode,
        str(checkpoint.get("run_mode", v3_train.RUN_MODE_CLOSED_SET)),
        str(checkpoint.get("thesis_claim", v3_train.DEFAULT_CLOSED_SET_THESIS_CLAIM)),
        str(checkpoint.get("novelty_score_mode", v3_train.DEFAULT_NOVELTY_SCORE_MODE)),
        str(checkpoint.get("decision_policy", v3_train.DEFAULT_DECISION_POLICY)),
        dict(checkpoint.get("task_activation", {})),
        bool(checkpoint.get("unknown_head_active", checkpoint_uses_ood_head(checkpoint))),
    )


def resolve_future_horizons_for_inference(
    requested_horizons,
    requested_single_horizon,
    checkpoint_horizons,
):
    if not requested_horizons and requested_single_horizon is None:
        return list(checkpoint_horizons)

    normalized = v3_train.normalize_future_horizons_minutes(
        requested_horizons,
        requested_single_horizon,
    )
    if list(normalized) != list(checkpoint_horizons):
        raise ValueError(
            "Future horizon overrides are not supported at inference time. "
            f"This checkpoint was trained with future_horizons_minutes={list(checkpoint_horizons)} "
            "and the future head shape depends on those horizons. Use the stored checkpoint horizons "
            "or retrain for a different horizon set."
        )
    return list(normalized)


def apply_threshold_overrides(
    thresholds,
    future_horizons_minutes,
    current_threshold=None,
    known_threshold=None,
    future_threshold=None,
):
    applied_thresholds = dict(thresholds)
    if current_threshold is not None:
        applied_thresholds["current"] = float(current_threshold)
    if known_threshold is not None:
        applied_thresholds["known"] = float(known_threshold)
    applied_thresholds["future"] = v3_train.normalize_future_thresholds(
        applied_thresholds.get("future"),
        future_horizons_minutes,
    )
    if future_threshold is not None:
        applied_thresholds["future"] = {
            label: float(future_threshold)
            for label in v3_train.build_future_horizon_labels(future_horizons_minutes)
        }
    if applied_thresholds.get("ood") is None:
        applied_thresholds["ood"] = 0.50
    return applied_thresholds


def apply_ood_threshold_override(thresholds, ood_threshold=None):
    applied_thresholds = dict(thresholds)
    if ood_threshold is not None:
        applied_thresholds["ood"] = float(ood_threshold)
    return applied_thresholds


def build_dataset(split_path, seq_len, stride, clip_value, future_horizons_minutes, known_attack_labels, max_sequences):
    base_dataset = SpatioTemporalNIDSDataset(
        arrow_dir_path=split_path,
        stats_path=DEFAULT_STATS,
        seq_len=seq_len,
        stride=stride,
        clip_value=clip_value,
    )
    known_attack_to_idx = {attack_name: idx for idx, attack_name in enumerate(known_attack_labels)}
    downstream_dataset = DownstreamNIDSDataset(
        base_dataset=base_dataset,
        future_horizons_minutes=future_horizons_minutes,
        known_attack_to_idx=known_attack_to_idx,
        max_sequences=max_sequences,
    )
    return downstream_dataset


def load_model(checkpoint, dataset, device, seq_len, future_task_enabled, future_horizons_minutes):
    num_cont = len(dataset.base_dataset.cont_cols)
    model = NIDSMultiTaskModel(
        backbone=SpatioTemporalTransformer(
            num_cont_features=num_cont,
            cat_vocab_sizes=CAT_VOCABS,
            seq_len=seq_len,
            init_mae=float(checkpoint.get("reconstruction_train_mae_mask_ratio", 0.10)),
            init_mfm=float(checkpoint.get("reconstruction_train_mfm_mask_ratio", 0.00)),
        ),
        num_known_attack_classes=len(checkpoint.get("known_attack_labels", [])),
        use_future_head=future_task_enabled,
        use_ood_head=checkpoint_uses_ood_head(checkpoint),
        future_horizons_minutes=future_horizons_minutes,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def normalize_vector_values(values):
    if isinstance(values, torch.Tensor):
        return [float(value) for value in values.detach().cpu().view(-1).tolist()]
    if isinstance(values, (list, tuple)):
        return [float(value) for value in values]
    return [float(values)]


def describe_prediction(sequence_idx, item, prediction, future_horizons_minutes):
    start_time = format_timestamp(item["start_time"])
    end_time = format_timestamp(item["end_time"])
    ground_truth_attack = item["attack"]
    current_ground_truth = "attack" if int(item["label"]) == 1 else "benign"
    future_horizon_labels = v3_train.build_future_horizon_labels(future_horizons_minutes)
    future_truths = normalize_vector_values(item["future_attack"])
    future_leads = normalize_vector_values(item["future_lead_minutes"])

    lines = [
        f"Sequence {sequence_idx + 1}",
        f"Window: {start_time} -> {end_time}",
        f"Current status: {prediction['status']}",
        f"Predicted attack type: {prediction['attack_type']}",
        f"Current attack probability: {prediction['current_attack_probability']:.4f}",
    ]

    if prediction.get("hybrid_unknown_risk_enabled", False):
        lines.append(
            "Novelty or unknown-risk probability: "
            f"{prediction['hybrid_unknown_risk_probability']:.4f} "
            f"(raw_unknown_head={prediction['raw_unknown_head_probability']:.4f}, "
            f"reconstruction_novelty={prediction['reconstruction_novelty_probability']:.4f}, "
            f"score={prediction['reconstruction_novelty_score']:.6f})"
        )
    elif prediction.get("ood_task_enabled", False):
        lines.append(
            f"Unknown-risk probability: {prediction['unknown_attack_probability']:.4f}"
        )

    lines.append(f"Decision policy: {prediction.get('decision_policy', 'unknown')}")
    if prediction.get("novelty_watch", False):
        lines.append("Interpretation: anomalous sequence without enough present-attack evidence.")
    elif prediction.get("unknown_attack_warning", False):
        lines.append("Interpretation: present-attack evidence plus novelty or family rejection.")

    if "known_attack_confidence" in prediction:
        lines.append(f"Known-family confidence: {prediction['known_attack_confidence']:.4f}")

    if prediction.get("future_task_enabled", True):
        lines.append(
            f"Early warning across horizons: {'yes' if prediction['future_warning'] else 'no'} "
            f"(max_probability={prediction['future_attack_probability']:.4f})"
        )
    else:
        lines.append("Early warning: disabled for this checkpoint")

    lines.extend(
        [
            f"Ground truth current label: {current_ground_truth}",
            f"Ground truth attack type: {ground_truth_attack}",
        ]
    )

    if prediction.get("future_task_enabled", True):
        future_probabilities = prediction.get("future_attack_probabilities", {})
        future_warnings = prediction.get("future_warnings_by_horizon", {})
        for horizon_idx, horizon_label in enumerate(future_horizon_labels):
            future_truth = "yes" if future_truths[horizon_idx] >= 0.5 else "no"
            warning = future_warnings.get(horizon_label, False)
            probability = future_probabilities.get(horizon_label, 0.0)
            line = (
                f"Future <= {future_horizons_minutes[horizon_idx]} min: "
                f"prediction={'yes' if warning else 'no'} "
                f"(probability={probability:.4f}) | truth={future_truth}"
            )
            if future_leads[horizon_idx] >= 0:
                line += f" | lead={future_leads[horizon_idx]:.2f} min"
            lines.append(line)

    return "\n".join(lines)


def summarize_predictions(predictions, title):
    status_counts = {}
    future_warning_counts = {}
    future_task_enabled = any(prediction.get("future_task_enabled", True) for prediction in predictions)

    for prediction in predictions:
        status_counts[prediction["status"]] = status_counts.get(prediction["status"], 0) + 1
        for horizon_label, warning in prediction.get("future_warnings_by_horizon", {}).items():
            future_warning_counts[horizon_label] = future_warning_counts.get(horizon_label, 0) + int(warning)

    summary_lines = [title, f"sequence count: {len(predictions)}"]
    for status_name in sorted(status_counts):
        summary_lines.append(f"{status_name}: {status_counts[status_name]}")
    if future_task_enabled:
        if future_warning_counts:
            for horizon_label in sorted(future_warning_counts, key=lambda value: int(value.rstrip('m'))):
                summary_lines.append(f"future warnings <= {horizon_label}: {future_warning_counts[horizon_label]}")
        else:
            summary_lines.append("future warnings: 0")
    else:
        summary_lines.append("future warnings: disabled")
    return "\n".join(summary_lines)


def prioritize_demo_candidates(candidates, demo_scenario):
    if demo_scenario is None or not candidates:
        return candidates

    if demo_scenario == "known_attack":
        return sorted(
            candidates,
            key=lambda candidate: (
                float(candidate["prediction"].get("status") == "known_attack"),
                float(candidate["prediction"].get("current_attack_probability", 0.0)),
                float(candidate["prediction"].get("known_attack_confidence", 0.0)),
            ),
            reverse=True,
        )

    if demo_scenario == "unknown_attack":
        return sorted(
            candidates,
            key=lambda candidate: (
                float(candidate["prediction"].get("status") == "unknown_attack_warning"),
                float(candidate["prediction"].get("novelty_watch", False)),
                float(
                    candidate["prediction"].get(
                        "hybrid_unknown_risk_probability",
                        candidate["prediction"].get("unknown_attack_probability", 0.0),
                    )
                ),
                float(candidate["prediction"].get("current_attack_probability", 0.0)),
            ),
            reverse=True,
        )

    return sorted(
        candidates,
        key=lambda candidate: (
            float(candidate["prediction"].get("current_attack_probability", 0.0)),
            float(
                candidate["prediction"].get(
                    "hybrid_unknown_risk_probability",
                    candidate["prediction"].get("unknown_attack_probability", 0.0),
                )
            ),
        ),
    )


def run_inference():
    args = parse_args()
    args = apply_demo_scenario_defaults(args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    (
        checkpoint,
        thresholds,
        known_attack_labels,
        pseudo_zero_day_families,
        checkpoint_horizons,
        future_task_enabled,
        checkpoint_seq_len,
        checkpoint_stride,
        reconstruction_calibration,
        use_reconstruction_hybrid_ood,
        reconstruction_validation_mae_mask_ratio,
        reconstruction_validation_mfm_mask_ratio,
        reconstruction_train_mae_mask_ratio,
        reconstruction_train_mfm_mask_ratio,
        unknown_risk_score_mode,
        run_mode,
        thesis_claim,
        novelty_score_mode,
        decision_policy,
        task_activation,
        unknown_head_active,
    ) = load_checkpoint(args.checkpoint, device)
    future_horizons_minutes = resolve_future_horizons_for_inference(
        args.future_horizons_minutes,
        args.future_horizon_minutes,
        checkpoint_horizons,
    )
    thresholds = apply_threshold_overrides(
        thresholds,
        future_horizons_minutes,
        current_threshold=args.current_threshold,
        known_threshold=args.known_threshold,
        future_threshold=args.future_threshold,
    )
    thresholds = apply_ood_threshold_override(thresholds, ood_threshold=args.ood_threshold)
    if args.seq_len is not None and args.seq_len != checkpoint_seq_len:
        raise ValueError(
            "Sequence length overrides are not supported at inference time. "
            f"This checkpoint expects seq_len={checkpoint_seq_len}."
        )
    seq_len = args.seq_len if args.seq_len is not None else checkpoint_seq_len
    stride = args.stride if args.stride is not None else checkpoint_stride
    split_path, resolved_split = resolve_split_path(
        args.split,
        allow_fallback=args.allow_split_fallback,
    )

    dataset = build_dataset(
        split_path=split_path,
        seq_len=seq_len,
        stride=stride,
        clip_value=args.clip_value,
        future_horizons_minutes=future_horizons_minutes,
        known_attack_labels=known_attack_labels,
        max_sequences=args.dataset_max_sequences,
    )
    data_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    model = load_model(checkpoint, dataset, device, seq_len, future_task_enabled, future_horizons_minutes)

    print(f"Loaded checkpoint: {args.checkpoint}")
    print(f"Demo scenario: {args.demo_scenario or 'manual'}")
    print(f"Running inference on split: {resolved_split}")
    print(f"Window config: seq_len={seq_len}, stride={stride}")
    print(f"Sequences available in replay dataset: {len(dataset)}")
    print(f"Sequences to print: {args.max_sequences}")
    print(f"Known attack labels: {known_attack_labels}")
    print(f"Pseudo-zero-day families: {pseudo_zero_day_families}")
    print(f"Run mode: {run_mode}")
    print(f"Thesis claim policy: {thesis_claim}")
    print(f"Decision policy: {decision_policy}")
    print(f"Novelty score mode: {novelty_score_mode}")
    print(f"Task activation: {task_activation}")
    print(f"Active thresholds: {thresholds}")
    print(f"Explicit unknown head available: {checkpoint_uses_ood_head(checkpoint)}")
    print(f"Unknown-head supervision active: {unknown_head_active}")
    print(f"Unknown-risk score mode: {unknown_risk_score_mode}")
    print(
        "Hybrid reconstruction-backed unknown risk: "
        f"{use_reconstruction_hybrid_ood}"
    )
    if reconstruction_calibration is not None:
        print(
            "Reconstruction mask ratios: "
            f"train(MAE/MFM)={reconstruction_train_mae_mask_ratio:.3f}/{reconstruction_train_mfm_mask_ratio:.3f} | "
            f"validation(MAE/MFM)={reconstruction_validation_mae_mask_ratio:.3f}/{reconstruction_validation_mfm_mask_ratio:.3f}",
        )
    print(f"Future task enabled: {future_task_enabled}")
    print(f"Future horizons minutes: {future_horizons_minutes}")
    print(f"Only attacks: {args.only_attacks}")
    print(f"Status filter: {args.status_filter}")
    print(f"Allow split fallback: {args.allow_split_fallback}")

    dataset_predictions = []
    displayed_predictions = []
    display_candidates = []
    manual_display_mode = args.demo_scenario is None

    with torch.no_grad():
        for batch_index, batch in enumerate(data_loader):
            cont = batch["continuous"].to(device, non_blocking=True)
            cat = batch["categorical"].to(device, non_blocking=True)
            reconstruction_mask = None
            reconstruction_mfm_mask = None
            if use_reconstruction_hybrid_ood:
                reconstruction_mask = stt_architecture.build_fixed_spatial_mask(
                    cont.shape[0],
                    cont.shape[1],
                    reconstruction_validation_mae_mask_ratio,
                    device,
                    v3_train.DEFAULT_RECONSTRUCTION_VALIDATION_MASK_SEED + batch_index,
                )
                if novelty_score_mode == v3_train.DEFAULT_NOVELTY_SCORE_MODE:
                    reconstruction_mfm_mask = stt_architecture.build_fixed_spatial_mask(
                        cont.shape[0],
                        cont.shape[1],
                        reconstruction_validation_mfm_mask_ratio,
                        device,
                        v3_train.DEFAULT_RECONSTRUCTION_VALIDATION_MASK_SEED + 100_000 + batch_index,
                    )

            if use_reconstruction_hybrid_ood and novelty_score_mode == v3_train.DEFAULT_NOVELTY_SCORE_MODE:
                outputs = stt_architecture.compute_combined_reconstruction_outputs(
                    model,
                    cont,
                    cat,
                    reconstruction_mask,
                    reconstruction_mfm_mask,
                )
            else:
                outputs = model(
                    cont,
                    cat,
                    apply_mfm=False,
                    compute_reconstruction=use_reconstruction_hybrid_ood,
                    reconstruction_mask=reconstruction_mask,
                    reconstruction_apply_mfm=False,
                )
            decoded_predictions = NIDSMultiTaskModel.decode_predictions(
                outputs,
                known_attack_labels,
                current_threshold=thresholds["current"],
                known_attack_threshold=thresholds["known"],
                future_threshold=thresholds["future"],
                ood_threshold=thresholds["ood"],
                future_horizons_minutes=future_horizons_minutes,
                reconstruction_calibration=reconstruction_calibration if use_reconstruction_hybrid_ood else None,
                hybrid_ood_threshold=thresholds["ood"] if use_reconstruction_hybrid_ood else None,
                unknown_risk_score_mode=unknown_risk_score_mode,
            )

            batch_size = len(decoded_predictions)
            for batch_idx in range(batch_size):
                prediction = decoded_predictions[batch_idx]
                dataset_predictions.append(prediction)

                if args.status_filter is not None and prediction["status"] != args.status_filter:
                    continue

                if args.only_attacks and int(batch["label"][batch_idx].item()) == 0:
                    continue

                item = {
                    "start_time": batch["start_time"][batch_idx].item(),
                    "end_time": batch["end_time"][batch_idx].item(),
                    "label": batch["label"][batch_idx].item(),
                    "attack": batch["attack"][batch_idx],
                    "future_attack": batch["future_attack"][batch_idx].tolist(),
                    "future_lead_minutes": batch["future_lead_minutes"][batch_idx].tolist(),
                }
                display_candidates.append({"prediction": prediction, "item": item})
                if manual_display_mode and len(display_candidates) >= args.max_sequences:
                    break

            if manual_display_mode and len(display_candidates) >= args.max_sequences:
                break

    ranked_candidates = prioritize_demo_candidates(display_candidates, args.demo_scenario)
    for printed_index, candidate in enumerate(ranked_candidates[: args.max_sequences]):
        displayed_predictions.append(candidate["prediction"])
        print(
            describe_prediction(
                printed_index,
                candidate["item"],
                candidate["prediction"],
                future_horizons_minutes,
            )
        )
        print("-" * 72)

    print(summarize_predictions(dataset_predictions, "Dataset-wide inference summary"))
    if displayed_predictions != dataset_predictions:
        print(summarize_predictions(displayed_predictions, "Displayed subset summary"))


if __name__ == "__main__":
    run_inference()