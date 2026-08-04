#!/usr/bin/env python3
"""H044: compare normalized-objective and meter-objective training.

Corrected metric-source policy
------------------------------
Training histories are used for optimization-loss and epoch curves.

Formal validation MAE/RMSE bars are NOT taken blindly from the training log.
The legacy strict normalized-objective training log records meter RMSE but may
not record meter MAE. Therefore this script first searches the selected formal
run for an existing validation evaluation output:

    **/val/per_tile_metrics.csv

It reconstructs exact pixel-weighted validation metrics from:
    mae_m_core_loss_pixel
    sse_m_core_loss_pixel
    n_core_loss_pixels

Only when no compatible evaluation output exists does it fall back to the
training history. Every row records metric_source and evaluation_metrics_path,
so a missing or wrong source cannot silently produce an empty bar.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PRESETS = ("CA", "CO", "Santiam")
RIVER_LABELS = {
    "CA": "CA Klamath",
    "CO": "CO Upper Colorado",
    "Santiam": "OR Santiam",
}
OBJECTIVES = ("Normalized objective", "Meter objective")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Compare normalized and meter objectives under strict and relaxed masks.",
    )
    parser.add_argument(
        "--strict_normalized_root",
        type=Path,
        default=Path(
            "/tank/data/SFS/xinyis/data/bathymetry/MAE-Topography/"
            "Downstream_Task_Bathy/cross_validation_v2"
        ),
    )
    parser.add_argument(
        "--strict_meter_root",
        type=Path,
        default=Path(
            "/tank/data/SFS/xinyis/data/bathymetry/MAE-Topography/"
            "Downstream_Task_Bathy/cross_validation_v4_meterMAE_BaselineEval"
        ),
    )
    parser.add_argument(
        "--relax_root",
        type=Path,
        default=Path(
            "/tank/data/SFS/xinyis/data/bathymetry/MAE-Topography/"
            "Downstream_Task_Bathy_relax_HiddenMask"
        ),
    )
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--dpi", type=int, default=220)
    return parser.parse_args()


def safe_float(value: Any) -> float:
    try:
        result = float(value)
    except Exception:
        return float("nan")
    return result if math.isfinite(result) else float("nan")


def flatten_json_log(record: Mapping[str, Any]) -> Dict[str, Any]:
    row: Dict[str, Any] = {"epoch": record.get("epoch")}
    for phase in ("train", "val"):
        payload = record.get(phase)
        if isinstance(payload, Mapping):
            for key, value in payload.items():
                row[f"{phase}_{key}"] = value
    for key in ("best_epoch", "best_metric", "best_metric_value", "phase"):
        if key in record:
            row[key] = record[key]
    return row


def load_history(run_dir: Path) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    log = run_dir / "log.txt"
    if log.is_file():
        for line in log.read_text(errors="ignore").splitlines():
            try:
                obj = json.loads(line)
            except Exception:
                continue
            if isinstance(obj, dict) and "epoch" in obj:
                rows.append(flatten_json_log(obj))
    if rows:
        frame = pd.DataFrame(rows)
    else:
        candidates = [run_dir / "history.csv"] + sorted(run_dir.glob("*history*.csv"))
        history = next((path for path in candidates if path.is_file()), None)
        if history is None:
            raise FileNotFoundError(f"No log.txt or history CSV in {run_dir}")
        frame = pd.read_csv(history)
    if "epoch" not in frame.columns:
        raise RuntimeError(f"History missing epoch column: {run_dir}")
    frame["epoch"] = pd.to_numeric(frame["epoch"], errors="coerce")
    frame = frame[np.isfinite(frame["epoch"])].copy()
    return frame.sort_values("epoch").drop_duplicates("epoch", keep="last").reset_index(drop=True)


def latest_formal_run(parent: Path, preferred_token: str) -> Path:
    if not parent.is_dir():
        raise FileNotFoundError(parent)
    candidates = []
    for checkpoint in parent.glob("*/checkpoint-best.pth"):
        run = checkpoint.parent
        if "smoke" in run.name.lower():
            continue
        candidates.append(run)
    if not candidates:
        raise FileNotFoundError(f"No checkpoint-best.pth below {parent}")
    preferred = [run for run in candidates if preferred_token.lower() in run.name.lower()]
    pool = preferred or candidates
    return max(pool, key=lambda run: (run / "checkpoint-best.pth").stat().st_mtime)


def best_epoch(frame: pd.DataFrame, objective: str) -> int:
    if "best_epoch" in frame.columns:
        values = pd.to_numeric(frame["best_epoch"], errors="coerce").dropna()
        if len(values):
            return int(round(float(values.iloc[-1])))
    selection_column = "val_loss" if objective == "Normalized objective" else "val_mae_m_mask"
    if selection_column not in frame.columns:
        raise RuntimeError(f"Missing {selection_column} in history")
    values = pd.to_numeric(frame[selection_column], errors="coerce")
    return int(round(float(frame.loc[values.idxmin(), "epoch"])))


def row_for_epoch(frame: pd.DataFrame, epoch: int) -> pd.Series:
    exact = frame[np.isclose(frame["epoch"].to_numpy(float), float(epoch))]
    if len(exact):
        return exact.iloc[-1]
    index = int(np.argmin(np.abs(frame["epoch"].to_numpy(float) - epoch)))
    return frame.iloc[index]


def metric(row: pd.Series, *names: str) -> float:
    for name in names:
        if name in row.index:
            value = safe_float(row[name])
            if math.isfinite(value):
                return value
    return float("nan")



def finite_series(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(dtype=float)
    values = pd.to_numeric(frame[column], errors="coerce")
    return values[np.isfinite(values)]


def evaluation_metrics_from_per_tile_csv(path: Path) -> Dict[str, Any]:
    """Reconstruct pixel-weighted validation metrics from evaluator output."""
    frame = pd.read_csv(path)

    required = {
        "mae_m_core_loss_pixel",
        "sse_m_core_loss_pixel",
        "n_core_loss_pixels",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise RuntimeError(
            f"Evaluation CSV is missing required columns {missing}: {path}"
        )

    count = pd.to_numeric(frame["n_core_loss_pixels"], errors="coerce").to_numpy(float)
    mae = pd.to_numeric(frame["mae_m_core_loss_pixel"], errors="coerce").to_numpy(float)
    sse = pd.to_numeric(frame["sse_m_core_loss_pixel"], errors="coerce").to_numpy(float)

    valid = (
        np.isfinite(count)
        & np.isfinite(mae)
        & np.isfinite(sse)
        & (count > 0)
    )
    if not valid.any():
        raise RuntimeError(f"No valid core-loss metric rows in {path}")

    total_pixels = float(count[valid].sum())
    global_mae = float(np.sum(mae[valid] * count[valid]) / total_pixels)
    global_rmse = float(np.sqrt(np.sum(sse[valid]) / total_pixels))

    normalized_mse = float("nan")
    if "sse_norm_core_loss_pixel" in frame.columns:
        sse_norm = pd.to_numeric(
            frame["sse_norm_core_loss_pixel"], errors="coerce"
        ).to_numpy(float)
        valid_norm = valid & np.isfinite(sse_norm)
        if valid_norm.any():
            normalized_mse = float(
                np.sum(sse_norm[valid_norm]) / np.sum(count[valid_norm])
            )

    return {
        "val_mae_m": global_mae,
        "val_rmse_m": global_rmse,
        "val_normalized_mse": normalized_mse,
        "evaluation_pixels": int(round(total_pixels)),
        "evaluation_metrics_path": str(path),
        "metric_source": "pixel_weighted_validation_evaluation",
    }


def evaluation_candidate_score(path: Path) -> Tuple[int, int, float]:
    """Prefer val outputs, core-pixel evaluators, then the newest file."""
    text = str(path).lower()
    val_score = int("/val/" in text.replace("\\", "/"))
    core_score = int("core" in text and "pixel" in text)
    return val_score, core_score, path.stat().st_mtime


def find_validation_evaluation_metrics(run_dir: Path) -> Optional[Dict[str, Any]]:
    candidates: List[Path] = []
    for path in run_dir.rglob("per_tile_metrics.csv"):
        text = str(path).lower().replace("\\", "/")
        if "/val/" not in text:
            continue
        try:
            frame = pd.read_csv(path, nrows=2)
        except Exception:
            continue
        required = {
            "mae_m_core_loss_pixel",
            "sse_m_core_loss_pixel",
            "n_core_loss_pixels",
        }
        if required.issubset(frame.columns):
            candidates.append(path)

    if not candidates:
        return None

    candidates.sort(key=evaluation_candidate_score, reverse=True)
    errors: List[str] = []
    for path in candidates:
        try:
            result = evaluation_metrics_from_per_tile_csv(path)
            result["evaluation_candidate_count"] = len(candidates)
            return result
        except Exception as exc:
            errors.append(f"{path}: {exc}")

    raise RuntimeError(
        f"Compatible evaluation CSVs were found below {run_dir}, "
        "but none could be read:\n" + "\n".join(errors)
    )


def history_metrics(frame: pd.DataFrame, row: pd.Series) -> Dict[str, Any]:
    return {
        "val_mae_m": metric(
            row,
            "val_mae_m_mask",
            "val_mae_m_core_loss_pixel",
            "val_mae_m",
        ),
        "val_rmse_m": metric(
            row,
            "val_rmse_m_mask",
            "val_rmse_m_core_loss_pixel",
            "val_rmse_m",
        ),
        "val_normalized_mse": metric(
            row,
            "val_normalized_mse_mask",
            "val_normalized_mse_core_loss_pixel",
        ),
        "evaluation_pixels": None,
        "evaluation_metrics_path": "",
        "metric_source": "training_history_fallback",
    }

def run_locations(args: argparse.Namespace) -> Dict[Tuple[str, str, str], Tuple[Path, str]]:
    locations: Dict[Tuple[str, str, str], Tuple[Path, str]] = {}
    for preset in PRESETS:
        locations[("Strict", preset, "Normalized objective")] = (
            args.strict_normalized_root / "runs" / f"holdout_{preset}_D001NoDataSafe",
            "train_holdout",
        )
        locations[("Strict", preset, "Meter objective")] = (
            args.strict_meter_root
            / "runs"
            / f"holdout_{preset}_D003MeterMAE_BaselineEval_D001NoDataSafe",
            "train_holdout",
        )
        locations[("Relaxed", preset, "Normalized objective")] = (
            args.relax_root
            / "results"
            / "NormOnly"
            / "runs"
            / f"holdout_{preset}_D001cAnyVisiblePatch_D001NoDataSafe",
            "train_holdout",
        )
        locations[("Relaxed", preset, "Meter objective")] = (
            args.relax_root
            / "results"
            / "MeterOnly"
            / "runs"
            / f"holdout_{preset}_D044MeterOnly_D001cAnyVisiblePatch_D001NoDataSafe",
            "train_holdout",
        )
    return locations


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    fields: List[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def summarize(args: argparse.Namespace):
    summaries: List[Dict[str, Any]] = []
    histories: Dict[Tuple[str, str, str], pd.DataFrame] = {}
    audit_rows: List[Dict[str, Any]] = []

    for key, (parent, token) in run_locations(args).items():
        regime, preset, objective = key
        run = latest_formal_run(parent, token)
        frame = load_history(run)
        epoch = best_epoch(frame, objective)
        row = row_for_epoch(frame, epoch)

        evaluation = find_validation_evaluation_metrics(run)
        if evaluation is None:
            selected_metrics = history_metrics(frame, row)
        else:
            selected_metrics = evaluation

        summary = {
            "mask_regime": regime,
            "preset": preset,
            "river": RIVER_LABELS[preset],
            "objective": objective,
            "run_parent": str(parent),
            "run_dir": str(run),
            "best_epoch": epoch,
            "selection_metric": (
                "val_loss"
                if objective == "Normalized objective"
                else "val_mae_m_mask"
            ),
            "train_loss": metric(row, "train_loss"),
            "val_loss": metric(row, "val_loss"),
            **selected_metrics,
        }

        # Do not silently emit an empty formal bar.
        if not math.isfinite(safe_float(summary["val_mae_m"])):
            raise RuntimeError(
                "Formal validation meter MAE is unavailable for:\n"
                f"  regime={regime}\n"
                f"  preset={preset}\n"
                f"  objective={objective}\n"
                f"  run={run}\n"
                "The training history did not contain meter MAE and no compatible "
                "val/per_tile_metrics.csv was found below the formal run."
            )
        if not math.isfinite(safe_float(summary["val_rmse_m"])):
            raise RuntimeError(
                "Formal validation meter RMSE is unavailable for:\n"
                f"  regime={regime}\n"
                f"  preset={preset}\n"
                f"  objective={objective}\n"
                f"  run={run}"
            )

        summaries.append(summary)
        histories[key] = frame
        audit_rows.append(
            {
                "mask_regime": regime,
                "preset": preset,
                "objective": objective,
                "run_parent": str(parent),
                "selected_run_dir": str(run),
                "best_epoch": epoch,
                "metric_source": summary["metric_source"],
                "evaluation_metrics_path": summary["evaluation_metrics_path"],
                "evaluation_pixels": summary["evaluation_pixels"],
                "val_mae_m": summary["val_mae_m"],
                "val_rmse_m": summary["val_rmse_m"],
            }
        )
    return summaries, histories, audit_rows

def summary_value(frame: pd.DataFrame, regime: str, preset: str, objective: str, metric_name: str) -> float:
    row = frame[
        (frame["mask_regime"] == regime)
        & (frame["preset"] == preset)
        & (frame["objective"] == objective)
    ]
    return float(row.iloc[0][metric_name]) if not row.empty else float("nan")


def grouped_metric_plot(frame: pd.DataFrame, metric_name: str, ylabel: str, output: Path, dpi: int) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.4), sharey=True)
    width = 0.36
    for ax, regime in zip(axes, ("Strict", "Relaxed")):
        x = np.arange(len(PRESETS), dtype=float)
        for index, objective in enumerate(OBJECTIVES):
            offset = (-0.5 if index == 0 else 0.5) * width
            values = [summary_value(frame, regime, preset, objective, metric_name) for preset in PRESETS]
            bars = ax.bar(x + offset, values, width, label=objective)
            for bar, value in zip(bars, values):
                if math.isfinite(value):
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        bar.get_height(),
                        f"{value:.3f}",
                        ha="center",
                        va="bottom",
                        fontsize=8,
                        rotation=90,
                    )
        ax.set_xticks(x, [RIVER_LABELS[p] for p in PRESETS])
        ax.set_title(f"{regime} Hidden Mask")
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", alpha=0.25)
        ax.legend()
    fig.suptitle(f"{ylabel}: normalized objective versus meter objective")
    fig.tight_layout()
    fig.savefig(output, dpi=dpi)
    plt.close(fig)


def delta_plot(frame: pd.DataFrame, output: Path, dpi: int) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), sharey=True)
    for ax, regime in zip(axes, ("Strict", "Relaxed")):
        deltas = []
        for preset in PRESETS:
            normalized = summary_value(frame, regime, preset, "Normalized objective", "val_mae_m")
            meter = summary_value(frame, regime, preset, "Meter objective", "val_mae_m")
            deltas.append(meter - normalized)
        x = np.arange(len(PRESETS), dtype=float)
        bars = ax.bar(x, deltas)
        ax.axhline(0, linewidth=1)
        for bar, value in zip(bars, deltas):
            if math.isfinite(value):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    value,
                    f"{value:+.3f}",
                    ha="center",
                    va="bottom" if value >= 0 else "top",
                    fontsize=9,
                )
        ax.set_xticks(x, [RIVER_LABELS[p] for p in PRESETS])
        ax.set_title(f"{regime} Hidden Mask")
        ax.set_ylabel("Meter-objective MAE − normalized-objective MAE (m)\nnegative = meter objective better")
        ax.grid(axis="y", alpha=0.25)
    fig.suptitle("Direct evidence for the meter-domain objective")
    fig.tight_layout()
    fig.savefig(output, dpi=dpi)
    plt.close(fig)


def training_curve_plots(
    summaries: Sequence[Mapping[str, Any]],
    histories: Mapping[Tuple[str, str, str], pd.DataFrame],
    output_dir: Path,
    dpi: int,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_frame = pd.DataFrame(summaries)
    for regime in ("Strict", "Relaxed"):
        for preset in PRESETS:
            fig, axes = plt.subplots(2, 2, figsize=(13, 9))
            for column_index, objective in enumerate(OBJECTIVES):
                frame = histories[(regime, preset, objective)]
                summary = summary_frame[
                    (summary_frame["mask_regime"] == regime)
                    & (summary_frame["preset"] == preset)
                    & (summary_frame["objective"] == objective)
                ].iloc[0]

                ax = axes[0, column_index]
                if "train_loss" in frame.columns:
                    ax.plot(frame["epoch"], pd.to_numeric(frame["train_loss"], errors="coerce"), label="Train optimization loss")
                if "val_loss" in frame.columns:
                    ax.plot(frame["epoch"], pd.to_numeric(frame["val_loss"], errors="coerce"), label="Validation optimization loss")
                ax.axvline(int(summary["best_epoch"]), linestyle="--", linewidth=1)
                ax.set_title(f"{objective}: optimization loss")
                ax.set_xlabel("Epoch")
                ax.set_ylabel("Objective-space loss")
                ax.grid(alpha=0.25)
                handles, labels = ax.get_legend_handles_labels()
                if handles:
                    ax.legend()

                ax = axes[1, column_index]
                plotted_mae_curve = False
                for column, label in (
                    ("train_mae_m_mask", "Train meter MAE"),
                    ("train_mae_m_core_loss_pixel", "Train meter MAE"),
                    ("val_mae_m_mask", "Validation meter MAE"),
                    ("val_mae_m_core_loss_pixel", "Validation meter MAE"),
                ):
                    if column in frame.columns:
                        values = pd.to_numeric(frame[column], errors="coerce")
                        if np.isfinite(values.to_numpy(float)).any():
                            ax.plot(frame["epoch"], values, label=label)
                            plotted_mae_curve = True
                if not plotted_mae_curve:
                    ax.text(
                        0.5,
                        0.5,
                        "Legacy training history did not record\nmeter-domain MAE by epoch.\n"
                        "Formal validation MAE is read from\nthe saved evaluation output.",
                        transform=ax.transAxes,
                        ha="center",
                        va="center",
                        fontsize=10,
                    )
                ax.axvline(int(summary["best_epoch"]), linestyle="--", linewidth=1)
                ax.set_title(
                    f"{objective}: meter-domain MAE\n"
                    f"best val MAE={float(summary['val_mae_m']):.4f} m"
                )
                ax.set_xlabel("Epoch")
                ax.set_ylabel("MAE (m)")
                ax.grid(alpha=0.25)
                handles, labels = ax.get_legend_handles_labels()
                if handles:
                    ax.legend()
            fig.suptitle(f"{regime} mask — {RIVER_LABELS[preset]}")
            fig.tight_layout()
            fig.savefig(output_dir / f"H044_{regime.lower()}_{preset}_training_curves.png", dpi=dpi)
            plt.close(fig)


def main() -> None:
    args = parse_args()
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)

    summaries, histories, audit_rows = summarize(args)
    frame = pd.DataFrame(summaries)
    write_csv(output / "H044_training_objective_per_river.csv", summaries)
    write_csv(output / "H044_metric_source_audit.csv", audit_rows)

    macro_rows = []
    for regime in ("Strict", "Relaxed"):
        for objective in OBJECTIVES:
            subset = frame[(frame["mask_regime"] == regime) & (frame["objective"] == objective)]
            macro_rows.append(
                {
                    "mask_regime": regime,
                    "objective": objective,
                    "macro_mean_val_mae_m": float(subset["val_mae_m"].mean()),
                    "macro_mean_val_rmse_m": float(subset["val_rmse_m"].mean()),
                    "n_rivers": int(len(subset)),
                }
            )
    write_csv(output / "H044_training_objective_macro.csv", macro_rows)

    grouped_metric_plot(frame, "val_mae_m", "Formal validation MAE (m)", output / "H044_validation_mae.png", args.dpi)
    grouped_metric_plot(frame, "val_rmse_m", "Formal validation RMSE (m)", output / "H044_validation_rmse.png", args.dpi)
    delta_plot(frame, output / "H044_meter_minus_normalized_mae.png", args.dpi)
    training_curve_plots(summaries, histories, output / "training_curves", args.dpi)

    interpretation = {
        "comparison": "Normalized objective versus meter objective",
        "mask_regimes": ["Strict", "Relaxed"],
        "primary_report_metric": "pixel-weighted meter-domain validation MAE",
        "metric_source_policy": (
            "Prefer existing val/per_tile_metrics.csv under each formal run; "
            "fall back to the training history only when the evaluator output "
            "is unavailable and the history contains the required metric."
        ),
        "important_interpretation": (
            "A negative bar in H044_meter_minus_normalized_mae.png means the "
            "meter objective has lower meter-domain validation MAE. CA is the "
            "key high-std holdout for demonstrating robustness to scale variation."
        ),
        "outputs": {
            "per_river_csv": "H044_training_objective_per_river.csv",
            "metric_source_audit_csv": "H044_metric_source_audit.csv",
            "macro_csv": "H044_training_objective_macro.csv",
            "mae_figure": "H044_validation_mae.png",
            "rmse_figure": "H044_validation_rmse.png",
            "delta_figure": "H044_meter_minus_normalized_mae.png",
        },
    }
    (output / "H044_summary.json").write_text(json.dumps(interpretation, indent=2))
    print(json.dumps(interpretation, indent=2))
    print("[DONE]", output)


if __name__ == "__main__":
    main()
