#!/usr/bin/env python3
"""Resolve the existing relaxed-normalized full-river prediction root.

The earlier shell resolver searched only directory names such as
``FullRiver_Predictions*``. That can miss a valid branch when:

- the output root has a different name;
- the result lives below ``Results`` rather than ``results``;
- the experiment was written with a generic F010/F044 directory name;
- the objective is identifiable only from ``F044_summary.json`` checkpoint
  metadata.

This resolver searches prediction manifests recursively and classifies a branch
from its checkpoint/argument metadata. It accepts only a complete CA/CO/Santiam
branch and, by default, requires the corrected prediction-patch-filter marker.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

RIVERS = {
    "CA": "CA_KlamathRiver_TopoBathy_2018_D18",
    "CO": "CO_UpperColorado_Topobathy_1_2020",
    "Santiam": "OR_SantiamRiverTB_Topobathy_1_D23",
}

POSITIVE_TOKENS = (
    "normonly",
    "normalizedobjective",
    "normalized_objective",
    "/normalized/",
    "/norm/",
    "d040",
    "d041",
)
NEGATIVE_TOKENS = (
    "meteronly",
    "meter_then",
    "meterthen",
    "normthenmeter",
    "norm_then_meter",
    "d044meteronly",
    "d048normthenmeter",
    "d054meterthennorm",
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    p.add_argument("--relax-root", type=Path, required=True)
    p.add_argument("--explicit-root", type=Path, default=None)
    p.add_argument("--output-json", type=Path, default=None)
    p.add_argument("--print-root", action="store_true")
    p.add_argument(
        "--allow-unmarked",
        action="store_true",
        help=(
            "Allow a relaxed prediction branch without "
            "prediction_patch_filter_applied=true. Not recommended."
        ),
    )
    return p.parse_args()


def read_json(path: Path) -> Dict[str, Any]:
    try:
        value = json.loads(path.read_text())
        return value if isinstance(value, dict) else {}
    except Exception:
        return {}


def normalized_text(value: Any) -> str:
    return str(value).replace("\\", "/").lower()


def summary_candidates(river_dir: Path) -> List[Path]:
    files = sorted(river_dir.glob("*summary.json"))
    files.sort(
        key=lambda p: (
            int(p.name in ("F044_summary.json", "F010_summary.json")),
            p.stat().st_mtime,
        ),
        reverse=True,
    )
    return files


def metadata_text(river_dir: Path, summary: Dict[str, Any]) -> str:
    pieces = [str(river_dir), json.dumps(summary, sort_keys=True)]
    roots = [river_dir, river_dir.parent, river_dir.parent.parent]
    for root in roots:
        for pattern in ("*args*.json", "*summary*.json"):
            for path in sorted(root.glob(pattern)):
                if path.parent == river_dir and path.name.endswith("summary.json"):
                    continue
                try:
                    pieces.append(path.read_text(errors="ignore"))
                except Exception:
                    pass
    return normalized_text("\n".join(pieces))


def objective_score(text: str) -> int:
    if any(token in text for token in NEGATIVE_TOKENS):
        return -100
    score = 0
    for token in POSITIVE_TOKENS:
        if token in text:
            score += 10
    if "normalized" in text:
        score += 5
    if re.search(r"(^|[/_.-])norm($|[/_.-])", text):
        score += 3
    return score


def corrected_marker(summary: Dict[str, Any], river_dir: Path) -> bool:
    if summary.get("prediction_patch_filter_applied") is True:
        return True

    # Some corrected runs preserve the marker only in the root/all-river
    # summary or args file. Search nearby JSON metadata defensively.
    for root in (river_dir.parent, river_dir.parent.parent):
        for path in root.glob("*.json"):
            data = read_json(path)
            if data.get("prediction_patch_filter_applied") is True:
                return True
            if isinstance(data, list):
                for item in data:
                    if (
                        isinstance(item, dict)
                        and item.get("prediction_patch_filter_applied") is True
                    ):
                        return True
    return False


def infer_root(river_dir: Path, search_root: Path) -> Path:
    # Normal structure:
    # root / holdout_experiment / river / manifest
    parent = river_dir.parent
    if parent.name.startswith("holdout_") or "holdout" in parent.name.lower():
        return parent.parent

    # A root may contain river folders directly.
    if river_dir.parent == search_root:
        return search_root

    # Prefer the closest ancestor containing at least two holdout directories.
    for ancestor in river_dir.parents:
        if ancestor == search_root.parent:
            break
        try:
            holdouts = [
                child for child in ancestor.iterdir()
                if child.is_dir() and "holdout" in child.name.lower()
            ]
        except Exception:
            holdouts = []
        if len(holdouts) >= 2:
            return ancestor

    return river_dir.parent.parent


def manifests_below(root: Path) -> List[Path]:
    if not root.is_dir():
        return []
    patterns = (
        "*tileavg_prediction_manifest.csv",
        "F044_tileavg_prediction_manifest.csv",
        "F010_tileavg_prediction_manifest.csv",
    )
    found = set()
    for pattern in patterns:
        found.update(root.rglob(pattern))
    return sorted(found)


def evaluate_candidate_root(
    root: Path,
    all_records: Sequence[Dict[str, Any]],
    allow_unmarked: bool,
) -> Dict[str, Any]:
    records = [record for record in all_records if record["root"] == root]
    by_preset: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_preset[record["preset"]].append(record)

    selected: Dict[str, Dict[str, Any]] = {}
    for preset in RIVERS:
        options = sorted(
            by_preset.get(preset, []),
            key=lambda item: (
                item["objective_score"],
                int(item["corrected_marker"]),
                item["manifest_mtime"],
            ),
            reverse=True,
        )
        if options:
            selected[preset] = options[0]

    complete = all(preset in selected for preset in RIVERS)
    normalized = complete and all(
        selected[preset]["objective_score"] > 0 for preset in RIVERS
    )
    corrected = complete and all(
        selected[preset]["corrected_marker"] for preset in RIVERS
    )
    accepted = complete and normalized and (corrected or allow_unmarked)

    return {
        "root": str(root),
        "complete_three_rivers": complete,
        "classified_normalized": normalized,
        "corrected_prediction_patch_filter": corrected,
        "accepted": accepted,
        "selected": {
            preset: {
                key: value
                for key, value in selected[preset].items()
                if key not in ("root",)
            }
            for preset in selected
        },
    }


def discover(
    relax_root: Path,
    explicit_root: Optional[Path],
    allow_unmarked: bool,
) -> Tuple[Optional[Path], Dict[str, Any]]:
    search_roots: List[Path] = []
    if explicit_root is not None:
        search_roots.append(explicit_root)

    common = (
        relax_root / "results",
        relax_root / "Results",
        relax_root,
        relax_root.parent / "Processed_Results",
    )
    for path in common:
        if path.is_dir() and path not in search_roots:
            search_roots.append(path)

    exact_names = (
        "FullRiver_Predictions_F049_NormalizedObjective_D001cAnyVisiblePatch",
        "FullRiver_Predictions_H062_NormalizedObjective_D001cAnyVisiblePatch",
        "FullRiver_Predictions_H045_NormalizedObjective_D001cAnyVisiblePatch",
        "FullRiver_Predictions_F044_NormalizedObjective_D001cAnyVisiblePatch",
        "FullRiver_Predictions_F044_NormOnly_D001cAnyVisiblePatch",
        "FullRiver_Predictions_F010_TileAvgVRT_D001cAnyVisiblePatch",
        "FullRiver_Predictions_F010_TileAvgVRT_D001NoDataSafe",
    )
    exact_paths: List[Path] = []
    for base in search_roots:
        for name in exact_names:
            path = base / name
            if path.is_dir():
                exact_paths.append(path)

    records: List[Dict[str, Any]] = []
    seen_manifests = set()
    scan_roots = [*exact_paths, *search_roots]
    for search_root in scan_roots:
        for manifest in manifests_below(search_root):
            resolved = manifest.resolve()
            if resolved in seen_manifests:
                continue
            seen_manifests.add(resolved)
            river_dir = manifest.parent

            preset = next(
                (
                    preset
                    for preset, river in RIVERS.items()
                    if river.lower() in normalized_text(river_dir)
                ),
                None,
            )
            if preset is None:
                continue

            summaries = summary_candidates(river_dir)
            summary_path = summaries[0] if summaries else None
            summary = read_json(summary_path) if summary_path else {}
            text = metadata_text(river_dir, summary)
            root = infer_root(river_dir, search_root)

            records.append(
                {
                    "preset": preset,
                    "river_dir": str(river_dir),
                    "manifest": str(manifest),
                    "summary": str(summary_path) if summary_path else "",
                    "checkpoint": str(summary.get("checkpoint", "")),
                    "objective_score": objective_score(text),
                    "corrected_marker": corrected_marker(summary, river_dir),
                    "manifest_mtime": manifest.stat().st_mtime,
                    "root": root,
                }
            )

    candidate_roots = sorted({record["root"] for record in records})
    audits = [
        evaluate_candidate_root(root, records, allow_unmarked)
        for root in candidate_roots
    ]
    accepted = [audit for audit in audits if audit["accepted"]]

    # Exact explicit root wins when it is valid.
    if explicit_root is not None:
        explicit_resolved = explicit_root.resolve()
        exact = [
            audit
            for audit in accepted
            if Path(audit["root"]).resolve() == explicit_resolved
        ]
        if len(exact) == 1:
            return explicit_root, {
                "selected_root": str(explicit_root),
                "selection_reason": "explicit_root_validated",
                "search_roots": [str(path) for path in search_roots],
                "candidate_audit": audits,
            }

    if len(accepted) == 1:
        selected = Path(accepted[0]["root"])
        return selected, {
            "selected_root": str(selected),
            "selection_reason": "unique_complete_corrected_normalized_branch",
            "search_roots": [str(path) for path in search_roots],
            "candidate_audit": audits,
        }

    payload = {
        "selected_root": None,
        "selection_reason": (
            "no_complete_corrected_relaxed_normalized_branch"
            if not accepted
            else "multiple_complete_corrected_relaxed_normalized_branches"
        ),
        "search_roots": [str(path) for path in search_roots],
        "candidate_audit": audits,
        "manifest_records": [
            {
                key: (str(value) if isinstance(value, Path) else value)
                for key, value in record.items()
                if key != "root"
            }
            | {"root": str(record["root"])}
            for record in records
        ],
    }
    return None, payload


def main() -> None:
    args = parse_args()
    selected, payload = discover(
        args.relax_root,
        args.explicit_root,
        args.allow_unmarked,
    )

    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(payload, indent=2))

    if selected is None:
        print(
            "[ERROR] No complete corrected relaxed-normalized full-river "
            "prediction branch could be resolved.",
            file=sys.stderr,
        )
        print(
            "This is not a simple directory-name failure. The four-way H045/H046 "
            "analysis needs an existing relaxed-normalized prediction branch for "
            "CA, CO, and Santiam.",
            file=sys.stderr,
        )
        print(
            "Searched roots:\n  "
            + "\n  ".join(payload["search_roots"]),
            file=sys.stderr,
        )
        if payload["candidate_audit"]:
            print(
                "Candidate audit:\n"
                + json.dumps(payload["candidate_audit"], indent=2),
                file=sys.stderr,
            )
        else:
            print(
                "No compatible prediction manifests were found.",
                file=sys.stderr,
            )
        print(
            "Set RELAX_NORMALIZED_PRED_ROOT only when that branch already exists.",
            file=sys.stderr,
        )
        raise SystemExit(2)

    if args.print_root:
        print(selected)
    else:
        print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
