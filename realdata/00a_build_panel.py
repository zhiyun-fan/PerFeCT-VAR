#!/usr/bin/env python3
"""Construct the fixed N25p17 revenue panel directly from movement CSV files.

No logarithm, imputation, deterministic adjustment or normalization is applied.
Panel membership, order, time range and aggregation rules are fixed below.
Each run recalculates and replaces its output without reading previous outputs.
"""
from __future__ import annotations

import argparse
import tempfile
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
CLIENT_IDS = (
    136, 72, 67, 33, 84, 32, 126, 112, 129, 2, 94, 53, 105,
    121, 119, 47, 109, 9, 80, 59, 137, 68, 93, 102, 14,
)
FAMILY_IDS = (
    "ana", "bjc", "che", "coo", "cra", "did", "fec", "frd", "fre",
    "frj", "fsf", "gro", "lnd", "oat", "sna", "soa", "tpa",
)
COLUMNS = ("STORE", "WEEK", "MOVE", "QTY", "PRICE", "OK")
RAW_DIR = HERE / "data" / "dominick_raw"
WEEK_MIN, WEEK_MAX = 278, 399
WEEK_ONE_START_DATE = "1989-09-14"
CSV_CHUNKSIZE = 750_000
REVENUE_FORMULA = "MOVE * PRICE / QTY"
VALID_ROW_RULE = (
    "Integer STORE/WEEK in the fixed panel; OK=1; finite MOVE>=0; finite QTY>0; "
    "for positive MOVE, finite PRICE>=0. Zero MOVE remains valid even if PRICE is missing."
)






def save_panel(path, arrays):
    """Save the newly calculated panel, replacing any previous output atomically."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("wb", dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
        np.savez_compressed(handle, **arrays)
    temporary.replace(path)
    return "recomputed_and_saved"


def aggregate_category(source, client_ids, week_min, week_max, chunksize):
    """Match the historical float64, per-chunk bincount aggregation order."""
    count_weeks = week_max - week_min + 1
    bins = len(client_ids) * count_weeks
    sums = np.zeros(bins, dtype=np.float64)
    total = np.zeros(bins, dtype=np.int64)
    valid_count = np.zeros(bins, dtype=np.int64)
    positive_count = np.zeros(bins, dtype=np.int64)
    store_map = np.full(max(client_ids) + 1, -1, dtype=np.int64)
    store_map[client_ids] = np.arange(len(client_ids))
    diagnostics = dict(input_rows=0, selected_panel_rows=0, valid_rows=0,
                       positive_move_rows=0, invalid_ok_rows=0, invalid_qty_rows=0,
                       invalid_move_rows=0, invalid_price_for_positive_move_rows=0)
    started = perf_counter()
    for frame in pd.read_csv(source, usecols=list(COLUMNS), dtype={name: "float64" for name in COLUMNS},
                             chunksize=chunksize, engine="c"):
        diagnostics["input_rows"] += len(frame)
        store = frame["STORE"].to_numpy(copy=False)
        week = frame["WEEK"].to_numpy(copy=False)
        location = (np.isfinite(store) & np.isfinite(week)
                    & (store >= 0) & (store <= max(client_ids))
                    & (week >= week_min) & (week <= week_max))
        positions = np.flatnonzero(location)
        location[positions] &= ((store[positions] == np.floor(store[positions]))
                                & (week[positions] == np.floor(week[positions])))
        index = np.full(len(frame), -1, dtype=np.int64)
        positions = np.flatnonzero(location)
        index[positions] = store_map[store[positions].astype(np.int64)]
        location &= index >= 0
        positions = np.flatnonzero(location)
        diagnostics["selected_panel_rows"] += len(positions)
        if not len(positions):
            continue
        keys = index[positions] * count_weeks + week[positions].astype(np.int64) - week_min
        total += np.bincount(keys, minlength=bins)
        move = frame["MOVE"].to_numpy(copy=False)[positions]
        qty = frame["QTY"].to_numpy(copy=False)[positions]
        price = frame["PRICE"].to_numpy(copy=False)[positions]
        ok = frame["OK"].to_numpy(copy=False)[positions]
        good_move = np.isfinite(move) & (move >= 0)
        good_qty = np.isfinite(qty) & (qty > 0)
        positive = np.isfinite(move) & (move > 0)
        good_price = (~positive) | (np.isfinite(price) & (price >= 0))
        valid = (ok == 1) & good_move & good_qty & good_price
        diagnostics["invalid_ok_rows"] += int(np.count_nonzero(ok != 1))
        diagnostics["invalid_qty_rows"] += int(np.count_nonzero(~good_qty))
        diagnostics["invalid_move_rows"] += int(np.count_nonzero(~good_move))
        diagnostics["invalid_price_for_positive_move_rows"] += int(np.count_nonzero(positive & ~good_price))
        diagnostics["valid_rows"] += int(np.count_nonzero(valid))
        diagnostics["positive_move_rows"] += int(np.count_nonzero(valid & positive))
        selected_keys = keys[valid]
        selected_move, selected_qty, selected_price = move[valid], qty[valid], price[valid]
        positive_valid = selected_move > 0
        amounts = np.zeros(len(selected_move), dtype=np.float64)
        amounts[positive_valid] = selected_move[positive_valid] * selected_price[positive_valid] / selected_qty[positive_valid]
        if not np.all(np.isfinite(amounts)):
            raise ValueError(f"Non-finite raw revenue encountered: {source}")
        # Keep one accumulation per source CSV chunk; regrouping can change rounding.
        valid_count += np.bincount(selected_keys, minlength=bins)
        positive_count += np.bincount(selected_keys, weights=positive_valid.astype(np.float64), minlength=bins).astype(np.int64)
        sums += np.bincount(selected_keys, weights=amounts, minlength=bins)
    shape = (len(client_ids), count_weeks)
    revenue = sums.reshape(shape)
    counts = valid_count.reshape(shape)
    revenue[counts == 0] = np.nan
    result = dict(revenue=revenue, total_rows=total.reshape(shape), valid_rows=counts,
                  nonzero_rows=positive_count.reshape(shape))
    diagnostics.update(observed_cells=int(np.count_nonzero(counts)),
                       missing_cells=int(np.count_nonzero(counts == 0)),
                       observed_zero_cells=int(np.count_nonzero((counts > 0) & (revenue == 0))),
                       seconds=round(perf_counter() - started, 3))
    return result, diagnostics




def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, help="Directory containing the seventeen wXXX.csv files.")
    parser.add_argument("--output", type=Path, default=HERE / "data" / "raw_panel.npz")
    args = parser.parse_args()
    ids = np.asarray(CLIENT_IDS, dtype=np.int64)
    families = list(FAMILY_IDS)
    first, last = WEEK_MIN, WEEK_MAX
    chunksize = CSV_CHUNKSIZE
    raw_dir = (args.raw_dir or RAW_DIR).resolve()
    sources = [raw_dir / f"w{code}.csv" for code in families]
    missing_sources = [str(p) for p in sources if not p.is_file()]
    if missing_sources:
        raise FileNotFoundError("Missing movement CSVs: " + ", ".join(missing_sources))
    protected = {p.resolve() for p in sources}
    output = args.output.resolve()
    if output in protected:
        raise ValueError("Output files must differ from source files.")
    results = []
    for number, (code, source) in enumerate(zip(families, sources), start=1):
        print(f"[{number}/{len(families)}] Aggregating {source.name}", flush=True)
        stat_before = source.stat()
        result, diagnostics = aggregate_category(source, ids.tolist(), first, last, chunksize)
        stat_after = source.stat()
        if (stat_before.st_size, stat_before.st_mtime_ns) != (stat_after.st_size, stat_after.st_mtime_ns):
            raise RuntimeError(f"Raw file changed while being read: {source}")
        results.append(result)
        print(f"  {diagnostics['valid_rows']:,} valid rows; {diagnostics['missing_cells']} missing cells", flush=True)
    weeks = np.arange(first, last + 1, dtype=np.int64)
    arrays = {key: np.stack([result[key] for result in results], axis=2)
              for key in ("revenue", "total_rows", "valid_rows", "nonzero_rows")}
    arrays.update(client_ids=ids, family_ids=np.asarray(families), weeks=weeks,
                  week_start_dates=np.datetime64(WEEK_ONE_START_DATE, "D") + (weeks - 1) * np.timedelta64(7, "D"))
    arrays["observed_mask"] = arrays["valid_rows"] > 0
    if not np.array_equal(arrays["observed_mask"], np.isfinite(arrays["revenue"])):
        raise ValueError("Observed-cell definition and finite revenue disagree.")
    if np.any(arrays["revenue"][arrays["observed_mask"]] < 0):
        raise ValueError("Negative observed revenue.")
    if np.any(arrays["nonzero_rows"] > arrays["valid_rows"]) or np.any(arrays["valid_rows"] > arrays["total_rows"]):
        raise ValueError("Raw row count consistency failed.")
    status = save_panel(args.output, arrays)
    observed = int(arrays["observed_mask"].sum())
    missing = int((~arrays["observed_mask"]).sum())
    print(f"{status}: {args.output.resolve()} ({observed} observed, {missing} missing)", flush=True)


if __name__ == "__main__":
    main()
