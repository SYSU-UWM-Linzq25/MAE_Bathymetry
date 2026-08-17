#!/usr/bin/env python3
"""Verify state counts, concentric offsets, VRT/GeoTIFF geometry, NoData, and split consistency."""

from __future__ import annotations

import argparse
import logging
from collections import Counter, defaultdict
from pathlib import Path

import rasterio

from conus_common import SCALES, parse_states, read_csv, write_csv


LOG = logging.getLogger("sampling-qa")


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sampling-root", required=True)
    p.add_argument("--split-dir", default=None)
    p.add_argument("--states", nargs="*", default=None)
    p.add_argument("--target-per-state", type=int, default=1000)
    p.add_argument("--open-files-per-state", type=int, default=5)
    p.add_argument("--report", required=True)
    p.add_argument("--log-level", default="INFO")
    return p


def add_error(errors: list[dict], state: str, center_id: str, category: str, detail: str):
    errors.append({"state": state, "center_id": center_id, "category": category, "detail": detail})


def main() -> int:
    args = parser().parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper()), format="%(asctime)s %(levelname)s %(message)s")
    states = parse_states(args.states)
    root = Path(args.sampling_root).resolve()
    errors: list[dict] = []
    summary: list[dict] = []

    all_center_ids: set[str] = set()
    all_nested: dict[str, list[dict]] = defaultdict(list)
    for state in states:
        center_path = root / "manifests" / f"centers_{state}.csv"
        nested_path = root / "manifests" / f"nested_{state}.csv"
        if not center_path.exists() or not nested_path.exists():
            add_error(errors, state, "", "missing_manifest", f"{center_path} or {nested_path}")
            continue
        centers = read_csv(center_path)
        nested = read_csv(nested_path)
        if len(centers) != args.target_per_state:
            add_error(errors, state, "", "state_count", f"expected={args.target_per_state} found={len(centers)}")
        if len(nested) != len(centers) * len(SCALES):
            add_error(errors, state, "", "nested_count", f"expected={len(centers) * len(SCALES)} found={len(nested)}")
        center_ids_here = {row["center_id"] for row in centers}
        orphan_ids = {row["center_id"] for row in nested} - center_ids_here
        if orphan_ids:
            add_error(errors, state, "", "orphan_nested_center", f"count={len(orphan_ids)}")
        coords = [(row["center_x_5070"], row["center_y_5070"]) for row in centers]
        if len(coords) != len(set(coords)):
            add_error(errors, state, "", "duplicate_center", "duplicate center coordinates")
        for row in centers:
            if row["center_id"] in all_center_ids:
                add_error(errors, state, row["center_id"], "duplicate_center_id", "ID already used")
            all_center_ids.add(row["center_id"])
        for row in nested:
            all_nested[row["center_id"]].append(row)

        opened = 0
        for center in centers:
            cid = center["center_id"]
            rows = [r for r in nested if r["center_id"] == cid]
            by_scale = {r["scale"]: r for r in rows}
            if set(by_scale) != set(SCALES):
                add_error(errors, state, cid, "scale_set", f"found={sorted(by_scale)}")
                continue
            max_col = int(by_scale["S10"]["col_off"])
            max_row = int(by_scale["S10"]["row_off"])
            for scale, size in SCALES.items():
                row = by_scale[scale]
                inset = (SCALES["S10"] - size) // 2
                if int(row["size_pixels"]) != size:
                    add_error(errors, state, cid, "size_manifest", f"{scale}={row['size_pixels']}")
                if int(row["col_off"]) != max_col + inset or int(row["row_off"]) != max_row + inset:
                    add_error(errors, state, cid, "not_concentric", scale)

            if opened < args.open_files_per_state:
                for scale, size in SCALES.items():
                    path_text = by_scale[scale].get("sample_path", "")
                    if not path_text:
                        continue
                    path = Path(path_text)
                    if not path.exists():
                        add_error(errors, state, cid, "missing_sample", str(path))
                        continue
                    try:
                        with rasterio.open(path) as src:
                            if (src.width, src.height) != (size, size):
                                add_error(errors, state, cid, "file_size", f"{scale}={src.width}x{src.height}")
                            if abs(src.transform.a - 1.0) > 1e-6 or abs(src.transform.e + 1.0) > 1e-6:
                                add_error(errors, state, cid, "file_resolution", f"{scale}={src.transform}")
                    except Exception as exc:
                        add_error(errors, state, cid, "open_sample", f"{path}: {exc}")
                opened += 1
        summary.append(
            {
                "state": state,
                "centers": len(centers),
                "nested_rows": len(nested),
                "expected_nested_rows": args.target_per_state * len(SCALES),
            }
        )

    if args.split_dir:
        split_path = Path(args.split_dir).resolve() / "center_splits.csv"
        if not split_path.exists():
            add_error(errors, "", "", "missing_split_manifest", str(split_path))
        else:
            split_rows = read_csv(split_path)
            split_ids = [r["center_id"] for r in split_rows]
            if len(split_ids) != len(set(split_ids)):
                add_error(errors, "", "", "duplicate_split_id", "center appears more than once")
            missing = all_center_ids - set(split_ids)
            if missing:
                add_error(errors, "", "", "missing_split_center", f"count={len(missing)}")
            counts = Counter(r["split"] for r in split_rows)
            for name in ("train", "val"):
                if counts[name] == 0:
                    add_error(errors, "", "", "empty_split", name)

    report = Path(args.report).resolve()
    write_csv(report, errors, ("state", "center_id", "category", "detail"))
    write_csv(report.with_name(report.stem + "_summary.csv"), summary, ("state", "centers", "nested_rows", "expected_nested_rows"))
    if errors:
        LOG.error("QA FAIL: %d errors; report=%s", len(errors), report)
        for error in errors[:20]:
            LOG.error("%s", error)
        return 6
    LOG.info("QA PASS: states=%d centers=%d", len(states), len(all_center_ids))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
