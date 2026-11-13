#!/usr/bin/env python3
"""Build validation, test and full-sample views from the frozen N25p17 panel.

Each view independently fits deterministic effects to its originally observed
estimation rows, linearly interpolates the resulting residuals, centers each
store/category, and scales each category across all selected stores and time.
Preprocessing rules and splits are fixed below; panel coordinates come from raw_panel.npz.
No fitted VAR coefficients or historical experiment directories are required.
Each run recalculates and replaces its output without reading previous outputs.
"""
from __future__ import annotations

import argparse
import calendar
from datetime import date, timedelta
import os
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
N_CLIENTS, N_VARIABLES = 25, 17
SPLITS = {
    "validation": {"window_weeks": 82, "origins": 20, "scored_weeks": 17},
    "test": {"window_weeks": 102, "origins": 20, "scored_weeks": 20},
    "full": {"window_weeks": 122},
}
# Fixed paper preprocessing: log1p; observed-row OLS; residual interpolation;
# per-store/category centering; pooled category standard deviation (ddof=0).
SEASONAL_PERIOD_WEEKS = 52.0
FOURIER_ORDER = 3
INTERPOLATION_MAX_GAP = 2
VALIDATION_EXCLUDED_PANEL_WEEKS = (93, 94, 95)
RAW_WEEK_MIN, RAW_WEEK_MAX = 278, 399
WEEK_ONE_START_DATE = "1989-09-14"
HOLIDAYS = (
    "new_year_week", "easter_week", "memorial_day_week",
    "independence_day_week", "labor_day_week", "thanksgiving_week",
    "christmas_week",
)
METHOD = "bidirectional_linear_after_deterministic_adjustment"




def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as saved:
        return {key: saved[key].copy() for key in saved.files}




def easter_sunday(year: int) -> date:
    a = year % 19
    b, c = divmod(year, 100)
    d, e = divmod(b, 4)
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i, k = divmod(c, 4)
    ell = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * ell) // 451
    month = (h + ell - 7 * m + 114) // 31
    day = (h + ell - 7 * m + 114) % 31 + 1
    return date(year, month, day)


def nth_weekday(year: int, month: int, weekday: int, occurrence: int) -> date:
    first = date(year, month, 1)
    return first + timedelta(days=(weekday - first.weekday()) % 7 + 7 * (occurrence - 1))


def last_weekday(year: int, month: int, weekday: int) -> date:
    last = date(year, month, calendar.monthrange(year, month)[1])
    return last - timedelta(days=(last.weekday() - weekday) % 7)


def fixed_effect_design(week_starts: np.ndarray) -> tuple[np.ndarray, list[str]]:
    """Same unscaled time index and holiday-week convention as the paper data."""
    time = np.arange(len(week_starts), dtype=float)
    period, order = SEASONAL_PERIOD_WEEKS, FOURIER_ORDER
    fourier = np.column_stack([
        function(2 * np.pi * harmonic * time / period)
        for harmonic in range(1, order + 1)
        for function in (np.sin, np.cos)
    ])
    names = ["intercept", "trend"] + [
        name for harmonic in range(1, order + 1)
        for name in (f"sin_{harmonic}", f"cos_{harmonic}")
    ]
    starts = week_starts.astype("datetime64[D]").astype(object).tolist()
    years = range(min(value.year for value in starts), max(value.year for value in starts) + 1)
    events = {
        "new_year_week": [date(year, 1, 1) for year in years],
        "easter_week": [easter_sunday(year) for year in years],
        "memorial_day_week": [last_weekday(year, 5, 0) for year in years],
        "independence_day_week": [date(year, 7, 4) for year in years],
        "labor_day_week": [nth_weekday(year, 9, 0, 1) for year in years],
        "thanksgiving_week": [nth_weekday(year, 11, 3, 4) for year in years],
        "christmas_week": [date(year, 12, 25) for year in years],
    }
    columns = [np.ones(len(time)), time, fourier]
    for name in HOLIDAYS:
        columns.append(np.asarray([
            any(start <= event <= start + timedelta(days=6) for event in events[name])
            for start in starts
        ], dtype=float))
        names.append(name)
    return np.column_stack(columns), names


def missing_runs(observed: np.ndarray, max_gap: int) -> list[dict]:
    """Describe interpolation dependencies once; indices are zero-based."""
    if max_gap < 1:
        raise ValueError("interpolation_max_gap must be positive.")
    runs = []
    for client in range(observed.shape[0]):
        for variable in range(observed.shape[2]):
            positions = np.flatnonzero(~observed[client, :, variable])
            if not positions.size:
                continue
            breaks = np.flatnonzero(np.diff(positions) > 1) + 1
            for segment in np.split(positions, breaks):
                first, last = int(segment[0]), int(segment[-1])
                left, right = first - 1, last + 1
                if len(segment) > max_gap:
                    raise ValueError(f"Missing run exceeds {max_gap} at client {client}, variable {variable}.")
                if left < 0 or right >= observed.shape[1]:
                    raise ValueError("Linear interpolation requires observed endpoints on both sides.")
                runs.append(dict(client_index=client, variable_index=variable,
                                 first_index=first, last_index=last,
                                 left_index=left, right_index=right))
    return runs


def preprocess_view(log_values: np.ndarray, observed: np.ndarray, design: np.ndarray,
                    fit_end: int, runs: list[dict]) -> tuple:
    n, levels, p = log_values.shape
    if not 2 <= fit_end <= levels:
        raise ValueError(f"Invalid preprocessing fit_end={fit_end}.")
    # A gap in the estimation segment must not use an endpoint outside it.
    if any(run["first_index"] < fit_end <= run["right_index"] for run in runs):
        raise ValueError("An estimation-window gap requires an unavailable future endpoint.")
    series = log_values.transpose(1, 0, 2).reshape(levels, n * p)
    mask = observed.transpose(1, 0, 2).reshape(levels, n * p)
    coefficients = np.empty((design.shape[1], n * p), dtype=float)
    for column in range(n * p):
        fit_mask = mask[:fit_end, column]
        if fit_mask.sum() < design.shape[1]:
            raise ValueError(f"Too few observed estimation rows in series {column}.")
        coefficients[:, column] = np.linalg.lstsq(
            design[:fit_end][fit_mask], series[:fit_end, column][fit_mask], rcond=None,
        )[0]
    adjusted = series - design @ coefficients
    adjusted[~mask] = np.nan
    filled = adjusted.reshape(levels, n, p).transpose(1, 0, 2).copy()
    for run in runs:
        client, variable = run["client_index"], run["variable_index"]
        left, right = run["left_index"], run["right_index"]
        a, b = float(filled[client, left, variable]), float(filled[client, right, variable])
        for position in range(left + 1, right):
            weight = (position - left) / (right - left)
            filled[client, position, variable] = (1.0 - weight) * a + weight * b
    if not np.isfinite(filled).all():
        raise ValueError("Nonfinite residuals remain after interpolation.")
    means = filled[:, :fit_end].mean(axis=1, keepdims=True)
    centered = filled - means
    scales = centered[:, :fit_end].std(axis=(0, 1), ddof=0)
    if np.any(~np.isfinite(scales)) or np.any(scales <= 0):
        raise ValueError("Each category requires a finite positive scale.")
    standardized = centered / scales[None, None, :]
    np.testing.assert_allclose(standardized[:, :fit_end].mean(axis=1), 0, rtol=0, atol=1e-10)
    np.testing.assert_allclose(standardized[:, :fit_end].std(axis=(0, 1)), 1, rtol=0, atol=1e-10)
    return standardized, coefficients, means, scales


def validate_panel(panel: dict) -> None:
    required = {"revenue", "observed_mask", "client_ids", "family_ids", "weeks", "week_start_dates"}
    if not required.issubset(panel):
        raise ValueError(f"Raw panel is missing arrays: {sorted(required - panel.keys())}")
    for key, length in (("client_ids", N_CLIENTS), ("family_ids", N_VARIABLES)):
        if panel[key].shape != (length,):
            raise ValueError(f"Unexpected {key} dimensions for the fixed N25p17 panel.")
        if len(set(panel[key].tolist())) != len(panel[key]):
            raise ValueError(f"Duplicate values in {key}.")
    weeks = np.arange(RAW_WEEK_MIN, RAW_WEEK_MAX + 1)
    np.testing.assert_array_equal(panel["weeks"], weeks)
    expected_dates = np.datetime64(WEEK_ONE_START_DATE, "D") + (weeks - 1) * np.timedelta64(7, "D")
    np.testing.assert_array_equal(panel["week_start_dates"].astype("datetime64[D]"), expected_dates)
    expected_shape = (N_CLIENTS, len(weeks), N_VARIABLES)
    values, observed = panel["revenue"], panel["observed_mask"]
    if values.shape != expected_shape or observed.shape != expected_shape or observed.dtype != bool:
        raise ValueError("Raw revenue or observed-mask dimensions/dtype are invalid.")
    if np.any(np.isinf(values)) or not np.array_equal(observed, np.isfinite(values)):
        raise ValueError("The original mask must identify exactly the finite revenue cells.")
    if np.any(values[observed] < 0):
        raise ValueError("Negative revenue is incompatible with the frozen raw-panel rules.")


def build_views(panel: dict) -> dict:
    validate_panel(panel)
    observed = panel["observed_mask"]
    log_values = np.log1p(panel["revenue"].astype(np.float64, copy=False))
    dates = panel["week_start_dates"].astype("datetime64[D]")
    design, names = fixed_effect_design(dates)
    runs = missing_runs(observed, INTERPOLATION_MAX_GAP)
    arrays = dict(client_ids=panel["client_ids"].astype(np.int64),
                  family_ids=panel["family_ids"].astype(str), week_start_dates=dates,
                  weeks=panel["weeks"].astype(np.int64), log1p_tensor=log_values,
                  original_observed_mask=observed, imputation_mask=~observed,
                  preprocessing_method=np.asarray(METHOD))
    for view in ("validation", "test", "full"):
        split = SPLITS[view]
        fit_end = int(split["window_weeks"])
        standardized, coefficients, means, scales = preprocess_view(
            log_values, observed, design, fit_end, runs,
        )
        for suffix, value in (("fit_end", np.asarray(fit_end)), ("coefficients", coefficients),
                              ("means", means), ("scales", scales), ("standardized", standardized)):
            arrays[f"{view}_{suffix}"] = value
        if view == "full":
            if fit_end != len(dates):
                raise ValueError("The descriptive full-sample fit must use every panel week.")
            continue
        origins = int(split["origins"])
        if origins < 1 or fit_end + origins > len(dates):
            raise ValueError(f"Invalid {view} forecast range.")
        original = observed[:, fit_end:fit_end + origins].copy()
        mask = original.copy()
        target_weeks = np.arange(fit_end + 1, fit_end + origins + 1)
        if view == "validation":
            excluded = np.asarray(VALIDATION_EXCLUDED_PANEL_WEEKS, dtype=int)
            if not np.isin(excluded, target_weeks).all():
                raise ValueError("A fixed validation exclusion is outside the validation interval.")
            mask[:, np.isin(target_weeks, excluded), :] = False
        for offset in range(origins):
            target = fit_end + offset
            # The model uses rows offset ... target-1. A missing row in this
            # range is usable only once its observed right endpoint is known.
            if any(run["first_index"] < target and run["last_index"] >= offset
                   and run["right_index"] >= target for run in runs):
                if mask[:, offset, :].any():
                    raise ValueError(f"{view} panel week {target + 1} is scored despite an unavailable interpolation endpoint.")
        scored_weeks = int(mask.any(axis=(0, 2)).sum())
        if scored_weeks != int(split["scored_weeks"]):
            raise ValueError(f"{view} scored-week count does not match the fixed stage definition.")
        if view == "test" and not original.all():
            raise ValueError("The frozen paper test interval is expected to be fully observed.")
        arrays[f"{view}_actual"] = standardized[:, fit_end:fit_end + origins].copy()
        arrays[f"{view}_original_mask"] = original
        arrays[f"{view}_score_mask"] = mask
    arrays["fixed_effect_coefficients"] = arrays["full_coefficients"]
    arrays["client_means"] = arrays["full_means"]
    arrays["variable_stds"] = arrays["full_scales"]
    arrays["fixed_effect_names"] = np.asarray(names, dtype=str)
    for key in ("fixed_effect_coefficients", "client_means", "variable_stds", "fixed_effect_names"):
        arrays[f"full_{key}"] = arrays[key]
    for view in ("validation", "test"):
        fit_end = int(arrays[f"{view}_fit_end"])
        origins = int(SPLITS[view]["origins"])
        arrays[f"{view}_target_panel_weeks"] = np.arange(fit_end + 1, fit_end + origins + 1)
        arrays[f"{view}_target_week_start_dates"] = dates[fit_end:fit_end + origins]
    return arrays


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=HERE / "data/raw_panel.npz")
    parser.add_argument("--output", type=Path, default=HERE / "data/processed.npz")
    args = parser.parse_args()
    arrays = build_views(load_npz(args.input))
    output = args.output.resolve()
    if output == args.input.resolve():
        raise ValueError("The processed output must differ from the raw input.")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + f".{os.getpid()}.tmp.npz")
    try:
        np.savez_compressed(temporary, **arrays)
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)
    print(f"Saved {output}: 25 clients, 17 variables, 122 weeks; "
          "validation/test/full views with 17/20 scored forecast weeks.")


if __name__ == "__main__":
    main()
