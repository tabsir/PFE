import argparse
import math
import os
from pathlib import Path

import torch


DEFAULT_BASELINE = "/home/aka/PFE-code/checkpoints/nids_multitask_baseline"
DEFAULT_TEST = "/home/aka/PFE-code/checkpoints/nids_multitask_test"


def parse_args():
    parser = argparse.ArgumentParser(description="Compare two downstream NIDS best checkpoints.")
    parser.add_argument("--baseline", default=DEFAULT_BASELINE, help="Baseline output directory or best checkpoint path.")
    parser.add_argument("--test", default=DEFAULT_TEST, help="Test output directory or best checkpoint path.")
    return parser.parse_args()


def resolve_checkpoint_path(path_value):
    path = Path(path_value)
    if path.is_dir():
        path = path / "nids_multitask_best.pt"
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    return path


def nan_to_zero(value):
    if value is None:
        return 0.0
    value = float(value)
    if math.isnan(value):
        return 0.0
    return value


def get_validation_score(metrics):
    return (
        nan_to_zero(metrics["current"].get("auc"))
        + nan_to_zero(metrics["future"].get("auc"))
        + nan_to_zero(metrics.get("known_family_accuracy"))
        + nan_to_zero(metrics.get("unknown_warning_recall"))
    )


def load_summary(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    metrics = checkpoint.get("validation_metrics", {})
    if not metrics:
        raise ValueError(f"validation_metrics missing from checkpoint: {checkpoint_path}")

    return {
        "path": str(checkpoint_path),
        "epoch": int(checkpoint.get("epoch", -1)) + 1,
        "foundation_checkpoint": checkpoint.get("foundation_checkpoint", "unknown"),
        "output_dir": checkpoint.get("output_dir", str(checkpoint_path.parent)),
        "validation_score": float(checkpoint.get("validation_score", get_validation_score(metrics))),
        "current_auc": float(metrics["current"].get("auc", float("nan"))),
        "current_f1": float(metrics["current"].get("f1", float("nan"))),
        "future_auc": float(metrics["future"].get("auc", float("nan"))),
        "known_family_accuracy": float(metrics.get("known_family_accuracy", float("nan"))),
        "unknown_warning_recall": float(metrics.get("unknown_warning_recall", float("nan"))),
        "mean_future_lead_minutes": float(metrics.get("mean_future_lead_minutes", float("nan"))),
    }


def metric_winner(left_name, left_value, right_name, right_value):
    left_score = nan_to_zero(left_value)
    right_score = nan_to_zero(right_value)
    if math.isclose(left_score, right_score, rel_tol=0.0, abs_tol=1e-9):
        return "tie"
    return left_name if left_score > right_score else right_name


def print_run_summary(label, summary):
    print(f"{label} run")
    print(f"  best checkpoint: {summary['path']}")
    print(f"  epoch: {summary['epoch']}")
    print(f"  foundation checkpoint: {summary['foundation_checkpoint']}")
    print(f"  output dir: {summary['output_dir']}")
    print(f"  validation score: {summary['validation_score']:.6f}")
    print(f"  current auc: {summary['current_auc']:.4f}")
    print(f"  current f1: {summary['current_f1']:.4f}")
    print(f"  future auc: {summary['future_auc']:.4f}")
    print(f"  known family accuracy: {summary['known_family_accuracy']:.4f}")
    print(f"  unknown warning recall: {summary['unknown_warning_recall']:.4f}")
    print(f"  mean future lead minutes: {summary['mean_future_lead_minutes']:.2f}")


def main():
    args = parse_args()
    baseline_path = resolve_checkpoint_path(args.baseline)
    test_path = resolve_checkpoint_path(args.test)

    baseline = load_summary(baseline_path)
    test = load_summary(test_path)

    print_run_summary("baseline", baseline)
    print()
    print_run_summary("test", test)
    print()

    winners = {
        "validation_score": metric_winner("baseline", baseline["validation_score"], "test", test["validation_score"]),
        "current_auc": metric_winner("baseline", baseline["current_auc"], "test", test["current_auc"]),
        "current_f1": metric_winner("baseline", baseline["current_f1"], "test", test["current_f1"]),
        "future_auc": metric_winner("baseline", baseline["future_auc"], "test", test["future_auc"]),
        "known_family_accuracy": metric_winner("baseline", baseline["known_family_accuracy"], "test", test["known_family_accuracy"]),
        "unknown_warning_recall": metric_winner("baseline", baseline["unknown_warning_recall"], "test", test["unknown_warning_recall"]),
        "mean_future_lead_minutes": metric_winner("baseline", baseline["mean_future_lead_minutes"], "test", test["mean_future_lead_minutes"]),
    }

    print("Metric winners")
    for metric_name, winner in winners.items():
        print(f"  {metric_name}: {winner}")

    final_winner = winners["validation_score"]
    print()
    print(f"Overall downstream winner: {final_winner}")


if __name__ == "__main__":
    main()