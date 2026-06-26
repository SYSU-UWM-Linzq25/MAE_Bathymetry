#!/usr/bin/env python3
"""Prepare leave-one-river-out (LOORO) splits and per-fold audit CSVs.

This reuses the already completed allRiverCanonicalND A004 audit. Patch quality
is a property of each tile/mask pair under fixed thresholds, so it does not
need to be recomputed for every holdout fold.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

RIVERS: Tuple[str, ...] = (
    "BadgerFinNull",
    "Estabrook_Combined",
    "KewaFix2Null",
    "Kletzch_Combined_UpMax3Null",
    "CA_KlamathRiver_TopoBathy_2018_D18",
    "CO_UpperColorado_Topobathy_1_2020",
    "MD_PotomacRiver_Bathy_2019",
    "NE_Niobrara_Topobathy_2018",
    "OR_MKRC_Topobathy_2021",
    "OR_SantiamRiverTB_Topobathy_1_D23",
    "WA_ChehalisRiverTB_Topobathy_1_D23",
    "WA_Nisqually_Bathymetric_2020",
)

TILE_RE = re.compile(r"^Select_tile_Basin_1m_(.+)_ID\d+\.tif$")


def read_lines(path: Path) -> List[str]:
    with path.open() as f:
        return [line.strip() for line in f if line.strip()]


def write_lines(path: Path, values: Iterable[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for value in values:
            f.write(f"{value}\n")


def river_from_dem(path_text: str) -> str:
    name = Path(path_text).name
    match = TILE_RE.match(name)
    if not match:
        raise ValueError(f"Cannot parse river from DEM filename: {name}")
    return match.group(1)


def load_audit_rows(paths: Sequence[Path]) -> Tuple[List[str], Dict[str, dict]]:
    fieldnames: List[str] | None = None
    by_path: Dict[str, dict] = {}
    by_name: Dict[str, dict] = {}

    for path in paths:
        with path.open(newline="") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames is None:
                raise ValueError(f"Audit has no header: {path}")
            if fieldnames is None:
                fieldnames = list(reader.fieldnames)
            elif list(reader.fieldnames) != fieldnames:
                raise ValueError(f"Audit headers differ: {path}")

            for row in reader:
                dem_path = (row.get("dem_path") or "").strip()
                if not dem_path:
                    raise ValueError(f"Missing dem_path in {path}")
                if dem_path in by_path:
                    raise ValueError(f"Duplicate audit DEM path: {dem_path}")
                by_path[dem_path] = row

                name = Path(dem_path).name
                if name in by_name:
                    raise ValueError(f"Duplicate audit DEM basename: {name}")
                by_name[name] = row

    assert fieldnames is not None
    # Store basename fallback using a reserved prefix.
    for name, row in by_name.items():
        by_path[f"__BASENAME__/{name}"] = row
    return fieldnames, by_path


def find_audit_row(index: Dict[str, dict], dem_path: str) -> dict:
    row = index.get(dem_path)
    if row is None:
        row = index.get(f"__BASENAME__/{Path(dem_path).name}")
    if row is None:
        raise KeyError(f"No A004 audit row for: {dem_path}")
    return row


def write_audit(path: Path, fieldnames: Sequence[str], rows: Sequence[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def status_counts(rows: Sequence[dict]) -> dict:
    counts = Counter((row.get("status") or "UNKNOWN").strip() for row in rows)
    return dict(sorted(counts.items()))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--work",
        default=(
            "/tank/data/SFS/xinyis/data/bathymetry/MAE-Topography/"
            "Downstream_Task_Bathy"
        ),
    )
    parser.add_argument(
        "--source_holdout",
        default="OR_SantiamRiverTB_Topobathy_1_D23",
        help="Existing complete split whose train+val union contains all 12 rivers.",
    )
    parser.add_argument(
        "--river",
        action="append",
        dest="rivers",
        help="Prepare only this holdout river; repeat as needed. Default: all 12.",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    work = Path(args.work)
    source_split = work / "splits" / f"bathy_lcc_1m_holdout_{args.source_holdout}"
    source_audit = work / "splits" / "nodata_core_loss_audit_allRiverCanonicalND"

    dem_train = read_lines(source_split / "train.txt")
    dem_val = read_lines(source_split / "val.txt")
    mask_train = read_lines(source_split / "train_masks.txt")
    mask_val = read_lines(source_split / "val_masks.txt")

    if len(dem_train) != len(mask_train) or len(dem_val) != len(mask_val):
        raise ValueError("Source DEM/mask list lengths do not match.")

    pairs = list(zip(dem_train + dem_val, mask_train + mask_val))
    if len({dem for dem, _ in pairs}) != len(pairs):
        raise ValueError("Duplicate DEM paths in source train+val union.")

    records = []
    counts = Counter()
    for dem, mask in pairs:
        river = river_from_dem(dem)
        if river not in RIVERS:
            raise ValueError(
                f"Unexpected river in retained dataset: {river}. "
                "Update RIVERS only after verifying the dataset."
            )
        counts[river] += 1
        records.append((river, dem, mask))

    missing_rivers = [river for river in RIVERS if counts[river] == 0]
    if missing_rivers:
        raise ValueError(f"No tiles found for retained rivers: {missing_rivers}")

    train_audit = source_audit / "train_core_loss_audit_allRiverCanonicalND.csv"
    val_audit = source_audit / "val_core_loss_audit_allRiverCanonicalND.csv"
    fieldnames, audit_index = load_audit_rows((train_audit, val_audit))

    if len(audit_index) // 2 != len(records):
        raise ValueError(
            "Combined A004 audit does not cover exactly the source train+val union: "
            f"audit={len(audit_index)//2}, pairs={len(records)}"
        )

    selected = tuple(args.rivers) if args.rivers else RIVERS
    unknown = [river for river in selected if river not in RIVERS]
    if unknown:
        raise ValueError(f"Unknown holdout river(s): {unknown}")

    cv_root = work / "cross_validation"
    manifest_rows = []

    for holdout in selected:
        split_dir = cv_root / "splits" / f"holdout_{holdout}"
        audit_dir = cv_root / "audits" / f"holdout_{holdout}"

        if not args.overwrite:
            existing = [
                split_dir / "train.txt",
                split_dir / "val.txt",
                audit_dir / "train_core_loss_audit.csv",
                audit_dir / "val_core_loss_audit.csv",
            ]
            if any(path.exists() for path in existing):
                raise FileExistsError(
                    f"Fold output exists for {holdout}; use --overwrite."
                )

        train_records = [record for record in records if record[0] != holdout]
        val_records = [record for record in records if record[0] == holdout]

        write_lines(split_dir / "train.txt", (r[1] for r in train_records))
        write_lines(split_dir / "train_masks.txt", (r[2] for r in train_records))
        write_lines(split_dir / "val.txt", (r[1] for r in val_records))
        write_lines(split_dir / "val_masks.txt", (r[2] for r in val_records))

        train_rows = [find_audit_row(audit_index, r[1]) for r in train_records]
        val_rows = [find_audit_row(audit_index, r[1]) for r in val_records]
        write_audit(
            audit_dir / "train_core_loss_audit.csv", fieldnames, train_rows
        )
        write_audit(
            audit_dir / "val_core_loss_audit.csv", fieldnames, val_rows
        )

        fold_summary = {
            "holdout_river": holdout,
            "train_total": len(train_records),
            "val_total": len(val_records),
            "train_audit_status": status_counts(train_rows),
            "val_audit_status": status_counts(val_rows),
            "train_river_counts": dict(
                sorted(Counter(r[0] for r in train_records).items())
            ),
            "val_river_counts": dict(
                sorted(Counter(r[0] for r in val_records).items())
            ),
        }
        with (split_dir / "fold_summary.json").open("w") as f:
            json.dump(fold_summary, f, indent=2)

        train_pass = sum(
            (row.get("status") or "").strip() == "PASS" for row in train_rows
        )
        val_pass = sum(
            (row.get("status") or "").strip() == "PASS" for row in val_rows
        )
        manifest_rows.append(
            {
                "holdout_river": holdout,
                "train_total": len(train_records),
                "train_pass": train_pass,
                "train_drop": len(train_records) - train_pass,
                "val_total": len(val_records),
                "val_pass": val_pass,
                "val_drop": len(val_records) - val_pass,
                "split_dir": str(split_dir),
                "audit_dir": str(audit_dir),
            }
        )
        print(
            f"[{holdout}] train={len(train_records)} "
            f"(pass={train_pass}) val={len(val_records)} (pass={val_pass})"
        )

    manifest = cv_root / "LOORO_fold_manifest.csv"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    with manifest.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(manifest_rows[0]))
        writer.writeheader()
        writer.writerows(manifest_rows)

    print(f"Prepared folds: {len(manifest_rows)}")
    print(f"Manifest: {manifest}")


if __name__ == "__main__":
    main()
