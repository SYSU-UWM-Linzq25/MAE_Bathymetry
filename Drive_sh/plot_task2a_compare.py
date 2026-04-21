import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# =========================
# 修改这里
# =========================
ROOT = Path(r"/tank/data/SFS/xinyis/data/bathymetry/MAE-Topography/Upstream_Model_ReTrain/eval_stage2_compare")
OUTDIR = ROOT / "plots_task2a_compare"
OUTDIR.mkdir(parents=True, exist_ok=True)

# 目录名 -> 图上显示名
RUNS = {
    "upstream_mr_0.75": "UP-0.75",
    "upstream_mr_0.50": "UP-0.50",
    "upstream_mr_0.35": "UP-0.35",
    "upstream_mr_0.20": "UP-0.20",
    "stage2_decoder_mr_0.20": "DEC-0.20",
}

# 重点 baseline：同样是 0.20，下游 decoder 和 upstream 的直接对比
BASELINE_KEY = "upstream_mr_0.20"
DECODER_KEY = "stage2_decoder_mr_0.20"


def load_all_results(root: Path, runs: dict[str, str]) -> tuple[dict, dict]:
    summaries = {}
    metrics = {}

    for folder, label in runs.items():
        d = root / folder
        summary_file = d / "summary.json"
        metrics_file = d / "metrics.csv"

        if not summary_file.exists():
            raise FileNotFoundError(f"Missing: {summary_file}")
        if not metrics_file.exists():
            raise FileNotFoundError(f"Missing: {metrics_file}")

        with open(summary_file, "r", encoding="utf-8") as f:
            summaries[folder] = json.load(f)

        df = pd.read_csv(metrics_file)
        if "path" not in df.columns or "rmse_m_mask" not in df.columns:
            raise ValueError(f"{metrics_file} 缺少必要列 path / rmse_m_mask")

        df["run_key"] = folder
        df["run_label"] = label
        metrics[folder] = df

    return summaries, metrics


def make_summary_table(summaries: dict, runs: dict[str, str]) -> pd.DataFrame:
    rows = []
    for folder, label in runs.items():
        s = summaries[folder]["rmse_m_mask"]
        rows.append({
            "run_key": folder,
            "run_label": label,
            "count": s["count"],
            "mean": s["mean"],
            "median": s["median"],
            "p10": s["p10"],
            "p90": s["p90"],
            "p95": s["p95"],
            "min": s["min"],
            "max": s["max"],
        })
    return pd.DataFrame(rows)


def plot_upstream_trend(summary_df: pd.DataFrame, out_png: Path):
    # 只画 upstream 的四组趋势
    upstream_order = ["upstream_mr_0.75", "upstream_mr_0.50", "upstream_mr_0.35", "upstream_mr_0.20"]
    sub = summary_df.set_index("run_key").loc[upstream_order].reset_index()

    x = np.array([0.75, 0.50, 0.35, 0.20])
    mean_y = sub["mean"].values
    p95_y = sub["p95"].values
    p10_y = sub["p10"].values
    p90_y = sub["p90"].values

    plt.figure(figsize=(7, 5))
    plt.plot(x, mean_y, marker="o", label="mean rmse_m_mask")
    plt.plot(x, p95_y, marker="s", label="p95 rmse_m_mask")
    plt.fill_between(x, p10_y, p90_y, alpha=0.15, label="p10-p90")

    plt.gca().invert_xaxis()
    plt.xlabel("mask_ratio")
    plt.ylabel("rmse_m_mask (m)")
    plt.title("Stage2 val: upstream rmse_m_mask vs mask_ratio")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


def plot_all_boxplot(metrics: dict, runs: dict[str, str], out_png: Path):
    order = ["upstream_mr_0.75", "upstream_mr_0.50", "upstream_mr_0.35", "upstream_mr_0.20", "stage2_decoder_mr_0.20"]
    data = [metrics[k]["rmse_m_mask"].dropna().values for k in order]
    labels = [runs[k] for k in order]

    plt.figure(figsize=(8, 5))
    plt.boxplot(data, labels=labels, showfliers=True)
    plt.xlabel("model / mask setting")
    plt.ylabel("rmse_m_mask (m)")
    plt.title("Stage2 val: rmse_m_mask distribution")
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


def build_decoder_vs_up20_delta(metrics: dict) -> pd.DataFrame:
    base = metrics[BASELINE_KEY][["path", "rmse_m_mask"]].copy()
    base = base.rename(columns={"rmse_m_mask": "rmse_up20"})

    dec = metrics[DECODER_KEY][["path", "rmse_m_mask"]].copy()
    dec = dec.rename(columns={"rmse_m_mask": "rmse_dec20"})

    merged = pd.merge(base, dec, on="path", how="inner")
    merged["delta_dec_minus_up20"] = merged["rmse_dec20"] - merged["rmse_up20"]
    return merged


def plot_decoder_delta_hist(delta_df: pd.DataFrame, out_png: Path):
    vals = delta_df["delta_dec_minus_up20"].dropna().values

    plt.figure(figsize=(7, 5))
    plt.hist(vals, bins=40, alpha=0.8)
    plt.axvline(0.0, linestyle="--", linewidth=1)
    plt.xlabel("rmse_m_mask(DEC-0.20) - rmse_m_mask(UP-0.20) (m)")
    plt.ylabel("count")
    plt.title("Stage2 val: decoder-20 vs upstream-20 paired delta")
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


def plot_decoder_delta_box(delta_df: pd.DataFrame, out_png: Path):
    vals = delta_df["delta_dec_minus_up20"].dropna().values

    plt.figure(figsize=(5, 5))
    plt.boxplot([vals], labels=["DEC-0.20 - UP-0.20"], showfliers=True)
    plt.axhline(0.0, linestyle="--", linewidth=1)
    plt.ylabel("rmse_m_mask difference (m)")
    plt.title("Stage2 val: paired delta")
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


def main():
    summaries, metrics = load_all_results(ROOT, RUNS)

    # 1) summary table
    summary_df = make_summary_table(summaries, RUNS)
    summary_csv = OUTDIR / "task2a_rmse_m_mask_summary_table.csv"
    summary_df.to_csv(summary_csv, index=False)

    # 2) upstream trend only
    plot_upstream_trend(summary_df, OUTDIR / "task2a_upstream_trend.png")

    # 3) all boxplot
    plot_all_boxplot(metrics, RUNS, OUTDIR / "task2a_all_boxplot.png")

    # 4) decoder vs upstream20 paired delta
    delta_df = build_decoder_vs_up20_delta(metrics)
    delta_csv = OUTDIR / "task2a_decoder_vs_up20_delta.csv"
    delta_df.to_csv(delta_csv, index=False)

    plot_decoder_delta_hist(delta_df, OUTDIR / "task2a_decoder_vs_up20_delta_hist.png")
    plot_decoder_delta_box(delta_df, OUTDIR / "task2a_decoder_vs_up20_delta_box.png")

    print("[DONE]", summary_csv)
    print("[DONE]", OUTDIR / "task2a_upstream_trend.png")
    print("[DONE]", OUTDIR / "task2a_all_boxplot.png")
    print("[DONE]", delta_csv)
    print("[DONE]", OUTDIR / "task2a_decoder_vs_up20_delta_hist.png")
    print("[DONE]", OUTDIR / "task2a_decoder_vs_up20_delta_box.png")


if __name__ == "__main__":
    main()
