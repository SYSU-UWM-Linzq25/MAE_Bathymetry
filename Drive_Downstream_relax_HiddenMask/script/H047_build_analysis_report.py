#!/usr/bin/env python3
"""H047: assemble the analysis-only outputs into one HTML/Markdown report."""
from __future__ import annotations

import argparse
import html
import json
import os
from pathlib import Path

import pandas as pd


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--training_dir", type=Path, required=True)
    p.add_argument("--fullriver_dir", type=Path, required=True)
    p.add_argument("--reach_dir", type=Path, required=True)
    p.add_argument("--output_dir", type=Path, required=True)
    return p.parse_args()


def rel(path: Path, out: Path) -> str:
    return os.path.relpath(path.resolve(), out.resolve()).replace(os.sep, "/")


def table_html(frame: pd.DataFrame) -> str:
    copy = frame.copy()
    for column in copy.select_dtypes(include="number").columns:
        copy[column] = copy[column].round(4)
    return copy.to_html(index=False, border=0, classes="data-table")


def table_md(frame: pd.DataFrame) -> str:
    copy = frame.copy()
    for column in copy.select_dtypes(include="number").columns:
        copy[column] = copy[column].round(4)
    headers = [str(c) for c in copy.columns]
    rows = ["| " + " | ".join(headers) + " |"]
    rows.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for _, row in copy.iterrows():
        rows.append("| " + " | ".join("" if pd.isna(v) else str(v) for v in row) + " |")
    return "\n".join(rows)


def main():
    args = parse_args()
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)

    training = pd.read_csv(args.training_dir / "H044_training_objective_per_river.csv")
    training_macro = pd.read_csv(args.training_dir / "H044_training_objective_macro.csv")
    fullriver = pd.read_csv(args.fullriver_dir / "H045_fourway_common_loss_metrics.csv")
    pairwise = pd.read_csv(args.fullriver_dir / "H045_pairwise_common_loss_metrics.csv")
    reaches = pd.read_csv(args.reach_dir / "H046_all_reach_metrics.csv")
    selected = pd.read_csv(args.reach_dir / "H046_selected_reaches.csv")

    meter_delta = training[
        ["mask_regime", "preset", "river", "objective", "val_mae_m", "val_rmse_m"]
    ].copy()

    fullriver_main = fullriver[
        [
            "preset",
            "river_label",
            "configuration_label",
            "common_loss_pixels",
            "mae_m",
            "rmse_m",
            "bias_m",
            "median_abs_error_m",
            "p90_abs_error_m",
            "p95_abs_error_m",
        ]
    ].copy()

    meter_pairs = pairwise[pairwise["pair"].isin(
        ["Strict: normalized vs meter", "Relaxed: normalized vs meter", "Meter: strict vs relaxed"]
    )].copy()

    reach_summary = (
        reaches.groupby("preset", as_index=False)
        .agg(
            n_reaches=("segment_id", "count"),
            mean_common_pixels=("n_fourway_common_loss_pixels", "mean"),
            strict_normalized_mae=("strict_normalized_mae_m", "mean"),
            strict_meter_mae=("strict_meter_mae_m", "mean"),
            relaxed_normalized_mae=("relaxed_normalized_mae_m", "mean"),
            relaxed_meter_mae=("relaxed_meter_mae_m", "mean"),
        )
    )

    all_reaches_gallery = args.reach_dir / "H046_all_reaches_gallery.html"

    figs = [
        args.training_dir / "H044_validation_mae.png",
        args.training_dir / "H044_validation_rmse.png",
        args.training_dir / "H044_meter_minus_normalized_mae.png",
        args.fullriver_dir / "H045_fourway_common_mae_rmse.png",
        args.fullriver_dir / "H045_fourway_common_error_density_cdf.png",
        args.fullriver_dir / "H045_pairwise_common_mae_deltas.png",
        args.fullriver_dir / "H045_fourway_common_threshold_fraction.png",
        args.reach_dir / "H046_reach_mae_profiles.png",
    ]

    style = """
    body { font-family: Arial, sans-serif; max-width: 1500px; margin: 2rem auto; line-height: 1.5; }
    h1,h2,h3 { margin-top: 1.5em; }
    .callout { background:#f3f3f3; border-left:6px solid #444; padding:1rem; }
    .figure-grid { display:grid; grid-template-columns:repeat(auto-fit,minmax(430px,1fr)); gap:1rem; }
    .figure-grid img { width:100%; border:1px solid #bbb; }
    .data-table { border-collapse:collapse; width:100%; font-size:0.9rem; }
    .data-table th,.data-table td { border:1px solid #ccc; padding:0.4rem; }
    .data-table th { background:#eee; }
    """

    parts = [
        "<!doctype html><html><head><meta charset='utf-8'>",
        "<title>Bathymetry objective and mask analysis</title>",
        f"<style>{style}</style></head><body>",
        "<h1>Bathymetry objective and Hidden-Mask analysis</h1>",
        "<div class='callout'><strong>Main analysis rule:</strong> "
        "Two-stage models are not reconsidered here. The analysis compares the normalized-domain "
        "objective and meter-domain objective under strict and relaxed masks. Full-river figures "
        "use exact four-way common loss pixels, and each overlap-averaged geospatial pixel is counted once.</div>",
        "<h2>1. Training and validation</h2>",
        "<p>The purpose is to test whether the meter-domain objective gives more reliable physical-domain "
        "errors, particularly for high-variation rivers such as CA.</p>",
        table_html(training),
        "<h3>Macro results</h3>",
        table_html(training_macro),
        "<div class='figure-grid'>",
    ]
    for figure in figs[:3]:
        if figure.is_file():
            parts.append(f"<figure><img src='{html.escape(rel(figure,out))}'><figcaption>{html.escape(figure.stem)}</figcaption></figure>")
    parts += [
        "</div>",
        "<h2>2. Full-river reconstruction</h2>",
        "<p><strong>Comparison footprint:</strong> Core_Loss_Mask_Pixel AND valid GT AND valid predictions "
        "from strict-normalized, strict-meter, relaxed-normalized, and relaxed-meter. "
        "This is the four-way common loss-pixel footprint.</p>",
        table_html(fullriver_main),
        "<h3>Pairwise common-footprint comparisons</h3>",
        table_html(meter_pairs),
        "<div class='figure-grid'>",
    ]
    for figure in figs[3:7]:
        if figure.is_file():
            parts.append(f"<figure><img src='{html.escape(rel(figure,out))}'><figcaption>{html.escape(figure.stem)}</figcaption></figure>")
    parts += [
        "</div>",
        "<h2>3. Continuous-reach local reconstruction</h2>",
        "<p>Every successfully assembled continuous reach has four separate six-panel figures: "
        "strict-normalized, strict-meter, relaxed-normalized, and relaxed-meter. "
        "The best, median, worst, and diagnostic subsets are selected from this complete archive.</p>",
        (
            f"<p><a href='{html.escape(rel(all_reaches_gallery, out))}'>"
            "Open the searchable all-reaches gallery</a></p>"
            if all_reaches_gallery.is_file()
            else ""
        ),
        table_html(reach_summary),
    ]
    if figs[7].is_file():
        parts.append(f"<img style='width:100%' src='{html.escape(rel(figs[7],out))}'>")

    # Include a small subset of ready-to-use PPT figures.
    if not selected.empty:
        parts.append("<h3>Representative reach figures</h3>")
        shown = selected.sort_values(["preset", "selection_category", "selection_rank"]).head(24)
        for _, row in shown.iterrows():
            parts.append(
                f"<h4>{html.escape(str(row['preset']))} — "
                f"{html.escape(str(row['selection_category']))} rank {int(row['selection_rank'])}</h4>"
            )
            parts.append("<div class='figure-grid'>")
            for column in (
                "strict_normalized_6panel_png",
                "strict_meter_6panel_png",
                "relaxed_normalized_6panel_png",
                "relaxed_meter_6panel_png",
            ):
                path = Path(str(row[column]))
                if path.is_file():
                    parts.append(f"<img src='{html.escape(rel(path,out))}'>")
            parts.append("</div>")

    parts += [
        "<h2>Interpretation guardrails</h2><ul>",
        "<li>Training comparison uses each objective's formal selected checkpoint.</li>",
        "<li>Full-river comparisons use common loss pixels; unequal prediction coverage is not allowed to create an artificial advantage.</li>",
        "<li>Relaxed predictions must contain prediction_patch_filter_applied=true; newly opened patches are not scored as predictions.</li>",
        "<li>Local figures use identical extents and comparable value ranges within each reach.</li>",
        "</ul></body></html>",
    ]
    (out / "H047_analysis_report.html").write_text("\n".join(parts), encoding="utf-8")

    md = [
        "# Bathymetry objective and Hidden-Mask analysis",
        "",
        "## Main rule",
        "",
        "Full-river results use exact four-way common loss pixels. Each overlap-averaged geospatial pixel is counted once.",
        "",
        "## Training/validation",
        "",
        table_md(training),
        "",
        "## Full-river common loss pixels",
        "",
        table_md(fullriver_main),
        "",
        "## Pairwise common footprints",
        "",
        table_md(meter_pairs),
        "",
        "## Reach summary",
        "",
        table_md(reach_summary),
    ]
    (out / "H047_analysis_report.md").write_text("\n".join(md), encoding="utf-8")

    summary = {
        "training_rows": int(len(training)),
        "fullriver_rows": int(len(fullriver)),
        "pairwise_rows": int(len(pairwise)),
        "reach_rows": int(len(reaches)),
        "selected_reach_rows": int(len(selected)),
        "main_fullriver_footprint": (
            "Core_Loss_Mask_Pixel AND valid GT AND all four valid predictions; "
            "unique overlap-averaged geospatial pixels"
        ),
    }
    (out / "H047_report_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
