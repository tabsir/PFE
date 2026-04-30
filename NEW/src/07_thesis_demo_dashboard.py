import importlib.util
import json
from collections import Counter
from pathlib import Path

import pandas as pd
import streamlit as st
import torch
from torch.utils.data import DataLoader


APP_DIR = Path(__file__).resolve().parent
NEW_DIR = APP_DIR.parent
CHECKPOINT_DIR = NEW_DIR / "checkpoints" / "nids_multitask_05_with_future"
DEFAULT_CHECKPOINT = CHECKPOINT_DIR / "nids_multitask_best.pt"
FINAL_TEST_JSON = CHECKPOINT_DIR / "final_eval_test.json"
FINAL_TEST_OOD_JSON = CHECKPOINT_DIR / "final_eval_test_ood.json"
FINAL_TEST_MD = CHECKPOINT_DIR / "final_eval_test.md"
FINAL_TEST_OOD_MD = CHECKPOINT_DIR / "final_eval_test_ood.md"
DEMO_TEST_REPLAY = CHECKPOINT_DIR / "demo_test_detected_attacks.txt"
DEMO_TEST_REPLAY_FALLBACK = CHECKPOINT_DIR / "demo_test_replay.txt"
DEMO_OOD_REPLAY = CHECKPOINT_DIR / "demo_test_ood_replay.txt"
VALIDATION_CSV = CHECKPOINT_DIR / "05_validation_metrics_by_epoch.csv"
PLOT_FILES = {
    "Validation Overview": CHECKPOINT_DIR / "validation_metrics_overview.png",
    "Present Detection": CHECKPOINT_DIR / "validation_present_metrics.png",
    "Future And Auxiliary": CHECKPOINT_DIR / "validation_future_aux_metrics.png",
    "Threshold And Loss": CHECKPOINT_DIR / "validation_threshold_loss.png",
}


def load_local_module(module_name, file_path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@st.cache_resource
def load_infer_module():
    return load_local_module("thesis_infer_module", APP_DIR / "06_infer_nids.py")


@st.cache_data
def load_json(path_str):
    return json.loads(Path(path_str).read_text())


@st.cache_data
def load_markdown(path_str):
    return Path(path_str).read_text()


@st.cache_data
def load_validation_history(path_str):
    return pd.read_csv(path_str)


def format_metric(value, digits=4):
    if value is None:
        return "n/a"
    return f"{float(value):.{digits}f}"


def inject_styles():
    st.markdown(
        """
        <style>
        .stApp {
            background:
                radial-gradient(circle at top left, rgba(11, 102, 106, 0.08), transparent 32%),
                linear-gradient(180deg, #f7f8f3 0%, #eef2ea 100%);
        }
        .demo-card {
            background: rgba(255, 255, 255, 0.88);
            border: 1px solid rgba(20, 57, 69, 0.12);
            border-radius: 18px;
            padding: 1rem 1.15rem;
            box-shadow: 0 10px 30px rgba(16, 34, 45, 0.08);
            margin-bottom: 0.9rem;
        }
        .demo-kicker {
            font-size: 0.8rem;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            color: #56737a;
            margin-bottom: 0.2rem;
        }
        .demo-title {
            font-size: 2rem;
            font-weight: 700;
            color: #10222d;
            margin-bottom: 0.25rem;
        }
        .status-pill {
            display: inline-block;
            padding: 0.2rem 0.7rem;
            border-radius: 999px;
            font-size: 0.78rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }
        .status-benign {
            background: #dfece2;
            color: #275537;
        }
        .status-known_attack {
            background: #d6ecff;
            color: #0e4976;
        }
        .status-unknown_attack_warning {
            background: #fff1d8;
            color: #8a4d00;
        }
        .small-muted {
            color: #61777b;
            font-size: 0.92rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_header():
    st.markdown('<div class="demo-kicker">Local Defense Dashboard</div>', unsafe_allow_html=True)
    st.markdown('<div class="demo-title">Spatio-Temporal NIDS Thesis Demo</div>', unsafe_allow_html=True)
    st.markdown(
        "This dashboard presents the official 05 test metrics and a replay of model decisions without using a raw terminal during the defense.",
    )


def render_eval_cards(title, data, emphasize_ood=False):
    current = data["current_at_validation_threshold"]
    future = data["future_at_validation_threshold"]
    top = st.columns(4)
    top[0].metric(f"{title} PR-AUC", format_metric(current.get("pr_auc"), 3))
    top[1].metric(f"{title} AUC", format_metric(current.get("auc"), 3))
    top[2].metric(f"{title} F1", format_metric(current.get("f1"), 3))
    top[3].metric(f"{title} Recall", format_metric(current.get("recall"), 3))

    bottom = st.columns(4)
    bottom[0].metric("Benign FPR", format_metric(current.get("false_positive_rate"), 3))
    bottom[1].metric("Future AUC", format_metric(future.get("auc"), 3))
    bottom[2].metric("Known Family Acc", format_metric(data.get("known_family_accuracy"), 3))
    bottom[3].metric("Unknown Recall", format_metric(data.get("unknown_warning_recall"), 3))

    if emphasize_ood:
        st.warning(
            "OOD ranking signal exists, but operational unknown detection at the transferred threshold is currently weak. Use this split to discuss limitations honestly.",
        )
    else:
        st.success(
            "This is the main thesis result: strong present-attack detection on the standard held-out test split.",
        )


def parse_saved_replay(path_str):
    text = Path(path_str).read_text(errors="ignore")
    raw_lines = [line.strip() for line in text.splitlines()]
    start_idx = next((idx for idx, line in enumerate(raw_lines) if line.startswith("Loaded checkpoint:")), 0)
    lines = raw_lines[start_idx:]

    header = {}
    sequences = []
    current_sequence = None
    summary = {}
    in_summary = False

    for line in lines:
        if not line or line.startswith(">>>") or line.startswith("<<<"):
            continue
        if line == "-" * 72:
            if current_sequence is not None:
                sequences.append(current_sequence)
                current_sequence = None
            continue
        if line == "Inference summary":
            if current_sequence is not None:
                sequences.append(current_sequence)
                current_sequence = None
            in_summary = True
            continue
        if in_summary:
            if ":" in line:
                key, value = line.split(":", 1)
                summary[key.strip()] = value.strip()
            continue
        if line.startswith("Sequence "):
            if current_sequence is not None:
                sequences.append(current_sequence)
            current_sequence = {"Sequence": line}
            continue
        if current_sequence is None:
            if ":" in line:
                key, value = line.split(":", 1)
                header[key.strip()] = value.strip()
            continue
        if ":" in line:
            key, value = line.split(":", 1)
            current_sequence[key.strip()] = value.strip()

    if current_sequence is not None:
        sequences.append(current_sequence)

    return {
        "header": header,
        "sequences": sequences,
        "summary": summary,
    }


def render_status_pill(status):
    safe_status = status or "benign"
    st.markdown(
        f'<span class="status-pill status-{safe_status}">{safe_status.replace("_", " ")}</span>',
        unsafe_allow_html=True,
    )


def render_sequence_cards(sequence_payload):
    if not sequence_payload["sequences"]:
        st.info("No saved replay sequences were found for this selection.")
        return

    header = sequence_payload["header"]
    summary = sequence_payload["summary"]
    info_cols = st.columns(3)
    info_cols[0].caption(f"Split: {header.get('Running inference on split', 'n/a')}")
    info_cols[1].caption(f"Printed sequences: {header.get('Sequences to print', 'n/a')}")
    info_cols[2].caption(f"Status filter: {header.get('Status filter', 'n/a')}")

    if summary:
        st.caption(
            "Summary: " + ", ".join(f"{key}={value}" for key, value in summary.items()),
        )

    for sequence in sequence_payload["sequences"]:
        with st.container(border=True):
            upper = st.columns([1.2, 1, 1])
            upper[0].markdown(f"**{sequence.get('Sequence', 'Sequence')}**")
            upper[0].caption(sequence.get("Window", ""))
            with upper[1]:
                render_status_pill(sequence.get("Current status", "benign"))
                st.write(f"Predicted type: {sequence.get('Predicted attack type', 'n/a')}")
            with upper[2]:
                st.write(f"Current probability: {sequence.get('Current attack probability', 'n/a')}")
                if "Known-family confidence" in sequence:
                    st.write(f"Known confidence: {sequence.get('Known-family confidence')}" )

            lower = st.columns(2)
            lower[0].write(f"Ground truth label: {sequence.get('Ground truth current label', 'n/a')}")
            lower[0].write(f"Ground truth type: {sequence.get('Ground truth attack type', 'n/a')}")
            lower[1].write(f"Early warning: {sequence.get('Early warning', 'n/a')}")
            if "Ground truth future attack in horizon" in sequence:
                lower[1].write(
                    f"Future attack in horizon: {sequence.get('Ground truth future attack in horizon', 'n/a')}"
                )
            if "Ground truth future lead time" in sequence:
                lower[1].write(f"Future lead: {sequence.get('Ground truth future lead time')}" )


@st.cache_data(show_spinner=False)
def collect_live_replay(
    checkpoint_path_str,
    split,
    max_sequences,
    only_attacks,
    status_filter,
    dataset_max_sequences,
    batch_size,
    num_workers,
):
    infer_module = load_infer_module()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    (
        checkpoint,
        thresholds,
        known_attack_labels,
        checkpoint_horizon,
        future_task_enabled,
        checkpoint_seq_len,
        checkpoint_stride,
    ) = infer_module.load_checkpoint(checkpoint_path_str, device)

    split_path, resolved_split = infer_module.resolve_split_path(split)
    dataset = infer_module.build_dataset(
        split_path=split_path,
        seq_len=checkpoint_seq_len,
        stride=checkpoint_stride,
        clip_value=5.0,
        future_horizon_minutes=checkpoint_horizon,
        known_attack_labels=known_attack_labels,
        max_sequences=dataset_max_sequences,
    )
    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )
    model = infer_module.load_model(checkpoint, dataset, device, checkpoint_seq_len, future_task_enabled)

    records = []
    status_counts = Counter()
    future_warning_count = 0

    with torch.no_grad():
        for batch in data_loader:
            cont = batch["continuous"].to(device, non_blocking=True)
            cat = batch["categorical"].to(device, non_blocking=True)
            outputs = model(cont, cat, apply_mfm=False)
            decoded_predictions = infer_module.NIDSMultiTaskModel.decode_predictions(
                outputs,
                known_attack_labels,
                current_threshold=thresholds["current"],
                known_attack_threshold=thresholds["known"],
                future_threshold=thresholds["future"],
            )
            for batch_idx, prediction in enumerate(decoded_predictions):
                if status_filter is not None and prediction["status"] != status_filter:
                    continue
                label = int(batch["label"][batch_idx].item())
                if only_attacks and label == 0:
                    continue

                record = {
                    "Sequence": f"Sequence {len(records) + 1}",
                    "Window": (
                        f"{infer_module.format_timestamp(batch['start_time'][batch_idx].item())} -> "
                        f"{infer_module.format_timestamp(batch['end_time'][batch_idx].item())}"
                    ),
                    "Current status": prediction["status"],
                    "Predicted attack type": prediction["attack_type"],
                    "Current attack probability": f"{prediction['current_attack_probability']:.4f}",
                    "Ground truth current label": "attack" if label == 1 else "benign",
                    "Ground truth attack type": batch["attack"][batch_idx],
                    "Ground truth future attack in horizon": "yes" if float(batch["future_attack"][batch_idx].item()) >= 0.5 else "no",
                }
                if "known_attack_confidence" in prediction:
                    record["Known-family confidence"] = f"{prediction['known_attack_confidence']:.4f}"
                if prediction.get("future_task_enabled", True):
                    record["Early warning"] = (
                        f"attack likely within the next {checkpoint_horizon} minutes "
                        f"(probability={prediction['future_attack_probability']:.4f})"
                        if prediction["future_warning"]
                        else f"no imminent attack predicted within the next {checkpoint_horizon} minutes "
                        f"(probability={prediction['future_attack_probability']:.4f})"
                    )
                future_lead = float(batch["future_lead_minutes"][batch_idx].item())
                if future_lead >= 0:
                    record["Ground truth future lead time"] = f"{future_lead:.2f} minutes"

                records.append(record)
                status_counts[prediction["status"]] += 1
                future_warning_count += int(prediction["future_warning"])
                if len(records) >= max_sequences:
                    break
            if len(records) >= max_sequences:
                break

    return {
        "header": {
            "Loaded checkpoint": checkpoint_path_str,
            "Running inference on split": resolved_split,
            "Sequences available in replay dataset": str(len(dataset)),
            "Sequences to print": str(max_sequences),
            "Status filter": status_filter or "none",
        },
        "summary": {
            **{key: str(value) for key, value in status_counts.items()},
            "future warnings": str(future_warning_count),
        },
        "sequences": records,
    }


def render_live_replay_panel():
    st.markdown("### Live Replay")
    st.caption(
        "This section can rerun inference on demand. For the defense day, the Saved Replay tab is safer because it opens instantly and avoids waiting on a full split scan.",
    )
    with st.form("live_replay_form"):
        cols = st.columns(4)
        split = cols[0].selectbox("Split", ["test", "test_ood"], index=0)
        max_sequences = cols[1].number_input("Sequences to display", min_value=1, max_value=12, value=3)
        batch_size = cols[2].number_input("Batch size", min_value=16, max_value=512, value=128, step=16)
        dataset_limit = cols[3].number_input(
            "Dataset scan limit (0 = full split)",
            min_value=0,
            max_value=500000,
            value=0,
            step=1000,
        )
        flags = st.columns(3)
        only_attacks = flags[0].checkbox("Only ground-truth attacks", value=True)
        status_filter = flags[1].selectbox(
            "Predicted status filter",
            ["none", "known_attack", "unknown_attack_warning", "benign"],
            index=1,
        )
        num_workers = flags[2].number_input("Workers", min_value=0, max_value=8, value=0)
        submitted = st.form_submit_button("Run Live Replay", width="stretch")

    if submitted:
        effective_dataset_limit = None if dataset_limit == 0 else int(dataset_limit)
        effective_status_filter = None if status_filter == "none" else status_filter
        with st.spinner("Running live replay..."):
            payload = collect_live_replay(
                str(DEFAULT_CHECKPOINT),
                split,
                int(max_sequences),
                bool(only_attacks),
                effective_status_filter,
                effective_dataset_limit,
                int(batch_size),
                int(num_workers),
            )
        render_sequence_cards(payload)


def render_saved_replay_tab():
    replay_options = {
        "Detected Test Attacks": DEMO_TEST_REPLAY,
        "Early Test Windows": DEMO_TEST_REPLAY_FALLBACK,
        "OOD Replay": DEMO_OOD_REPLAY,
    }
    selection = st.radio("Saved replay", list(replay_options.keys()), horizontal=True)
    payload = parse_saved_replay(str(replay_options[selection]))
    render_sequence_cards(payload)


def render_plots_tab():
    history = load_validation_history(str(VALIDATION_CSV))
    st.markdown("### Validation Curves")
    st.caption("These are the saved plots exported from the 05 validation history.")
    for title, path in PLOT_FILES.items():
        if path.exists():
            st.image(str(path), caption=title, width="stretch")
    st.markdown("### Validation History Snapshot")
    st.dataframe(
        history[[
            "epoch_number",
            "validation_score",
            "best_current_auc",
            "best_current_pr_auc",
            "best_current_f1",
            "future_auc",
            "known_family_accuracy",
        ]],
        width="stretch",
        hide_index=True,
    )


def main():
    st.set_page_config(page_title="NIDS Thesis Demo", layout="wide")
    inject_styles()
    render_header()

    with st.sidebar:
        st.markdown("### Demo Assets")
        st.write(f"Checkpoint: {DEFAULT_CHECKPOINT.name}")
        st.write(f"Test report: {FINAL_TEST_MD.name}")
        st.write(f"OOD report: {FINAL_TEST_OOD_MD.name}")
        st.write(f"Validation history: {VALIDATION_CSV.name}")
        st.info(
            "Use Official Results for the thesis numbers. Use Saved Replay for the polished live demo. Use Live Replay only if you want to rerun inference on the spot.",
        )

    test_metrics = load_json(str(FINAL_TEST_JSON))
    ood_metrics = load_json(str(FINAL_TEST_OOD_JSON))

    overview, official, replay, live, plots = st.tabs(
        ["Overview", "Official Results", "Saved Replay", "Live Replay", "Plots"]
    )

    with overview:
        cols = st.columns(4)
        cols[0].metric("Best checkpoint", DEFAULT_CHECKPOINT.name)
        cols[1].metric("Test PR-AUC", format_metric(test_metrics["current_at_validation_threshold"]["pr_auc"], 3))
        cols[2].metric("Test AUC", format_metric(test_metrics["current_at_validation_threshold"]["auc"], 3))
        cols[3].metric("Test Benign FPR", format_metric(test_metrics["current_at_validation_threshold"]["false_positive_rate"], 3))
        st.markdown(
            "This dashboard is designed for the thesis defense. It shows the official aggregate test metrics and then replays individual windows from the same best checkpoint in a more professional format than a raw terminal.",
        )
        st.markdown(
            "The recommended flow is: start on Official Results, move to Saved Replay for the qualitative demonstration, and keep the OOD panel ready in case the jury asks about zero-day behavior.",
        )

    with official:
        left, right = st.columns(2)
        with left:
            st.markdown("### Standard Test Split")
            render_eval_cards("Test", test_metrics, emphasize_ood=False)
            with st.expander("Open raw report text"):
                st.markdown(load_markdown(str(FINAL_TEST_MD)))
        with right:
            st.markdown("### OOD Split")
            render_eval_cards("OOD", ood_metrics, emphasize_ood=True)
            with st.expander("Open raw OOD report text"):
                st.markdown(load_markdown(str(FINAL_TEST_OOD_MD)))

    with replay:
        render_saved_replay_tab()

    with live:
        render_live_replay_panel()

    with plots:
        render_plots_tab()


if __name__ == "__main__":
    main()