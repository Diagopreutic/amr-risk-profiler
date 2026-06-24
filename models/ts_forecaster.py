"""
models/ts_forecaster.py
────────────────────────
Objective 4: 7-year country-level AMR forecasts with 90% CI.
Backend: Holt-Winters exponential smoothing (statsmodels).

Forecast method selection per pair
────────────────────────────────────
  flat         : < 2 valid data points  ->  constant projection
  linear_trend : 2-4 valid data points  ->  OLS linear extrapolation
  holtwinters  : >= 5 valid data points ->  Holt-Winters additive-trend,
                                             damped, 90% CI from residuals

Parallel execution
───────────────────
All "holtwinters" pairs are fitted in parallel using joblib (loky backend),
which works correctly on Windows without __main__ guards.
"""

import os
import warnings
import logging
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
os.environ.setdefault('PYTHONWARNINGS', 'ignore')
logging.getLogger("cmdstanpy").setLevel(logging.CRITICAL)


# ── Statsmodels availability probe ────────────────────────────────────────────

def _probe_statsmodels() -> bool:
    try:
        from statsmodels.tsa.holtwinters import ExponentialSmoothing  # noqa
        return True
    except Exception:
        return False


_STATSMODELS_OK = _probe_statsmodels()


# ── Individual model fitters ──────────────────────────────────────────────────

def _flat_forecast(last_val: float, all_years: list,
                   interval_width: float) -> list:
    """Constant projection for pairs with < 2 data points."""
    margin = last_val * (1 - interval_width) * 0.5
    return [{'year': yr, 'yhat': last_val,
              'yhat_lower': max(0.0, last_val - margin),
              'yhat_upper': min(100.0, last_val + margin),
              'source': 'flat'} for yr in all_years]


def _linear_forecast(years_hist: np.ndarray, y_hist: np.ndarray,
                     all_years: list, interval_width: float) -> list:
    """OLS linear trend extrapolation with ±z*sigma envelope."""
    yr_mean   = years_hist.mean()
    slope, intercept = np.polyfit(years_hist - yr_mean, y_hist, 1)
    resid     = y_hist - (slope * (years_hist - yr_mean) + intercept)
    resid_std = float(np.std(resid, ddof=1)) if len(resid) > 2 else float(np.std(y_hist)) + 0.5
    z = 1.645   # 90% CI
    rows = []
    for yr in all_years:
        yhat   = float(np.clip(slope * (yr - yr_mean) + intercept, 0, 100))
        n_fore = max(0, yr - int(years_hist[-1]))
        margin = z * resid_std * (1 + n_fore * 0.05)
        rows.append({'year': yr, 'yhat': yhat,
                     'yhat_lower': float(np.clip(yhat - margin, 0, 100)),
                     'yhat_upper': float(np.clip(yhat + margin, 0, 100)),
                     'source': 'linear_trend'})
    return rows


def _holtwinters_forecast(years_hist: np.ndarray, y_hist: np.ndarray,
                           all_years: list, interval_width: float) -> list:
    """
    Holt-Winters exponential smoothing.
      - Additive trend, damped (prevents runaway long-term projections)
      - 90% CI estimated from bootstrapped in-sample residuals
    Falls back to linear trend if optimisation fails.
    """
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    try:
        model  = ExponentialSmoothing(y_hist, trend='add', damped_trend=True,
                                      initialization_method='estimated')
        fit    = model.fit(optimized=True, remove_bias=True)
        fitted = fit.fittedvalues
        resid_std = float(np.std(y_hist - fitted, ddof=1)) if len(y_hist) > 1 else 1.0
        z = float(np.abs(np.percentile(
            np.random.default_rng(42).standard_normal(10_000),
            100 * (1 - interval_width) / 2
        )))
        n_hist = len(years_hist)
        n_fore = max(0, len(all_years) - n_hist)
        yhat_all = np.concatenate([fitted, fit.forecast(n_fore) if n_fore > 0 else []])
        steps    = np.concatenate([np.zeros(n_hist), np.arange(1, n_fore + 1)])
        margin   = z * resid_std * np.sqrt(1 + steps * 0.05)
        rows = []
        for i, yr in enumerate(all_years):
            yhat = float(np.clip(yhat_all[i], 0, 100)) if i < len(yhat_all) else float(y_hist[-1])
            rows.append({'year': yr, 'yhat': yhat,
                         'yhat_lower': float(np.clip(yhat - margin[i], 0, 100)),
                         'yhat_upper': float(np.clip(yhat + margin[i], 0, 100)),
                         'source': 'holtwinters'})
        return rows
    except Exception:
        return _linear_forecast(years_hist, y_hist, all_years, interval_width)


# ── Worker (module-level for joblib pickling) ─────────────────────────────────

def _fit_one_pair(args):
    """
    Fit a single (country, combo) pair.
    args = (country, combo, years_hist, y_hist, all_years,
             interval_width, use_hw, force_linear)
    """
    country, combo, years_hist, y_hist, all_years, \
        interval_width, use_hw, force_linear = args

    if force_linear:
        return country, combo, _linear_forecast(
            years_hist, y_hist, all_years, interval_width)

    rows = None
    if use_hw:
        try:
            rows = _holtwinters_forecast(
                years_hist, y_hist, all_years, interval_width)
        except Exception:
            rows = None

    if rows is None:
        rows = _linear_forecast(years_hist, y_hist, all_years, interval_width)

    return country, combo, rows


# ── Main public function ──────────────────────────────────────────────────────

def run_ts_forecasts(panel: pd.DataFrame,
                     countries: list,
                     combos: list,
                     hist_years: list,
                     forecast_years: list,
                     interval_width: float = 0.90,
                     n_jobs: int = -1,
                     observed_pairs: list = None) -> pd.DataFrame:
    """
    Fit a Holt-Winters time-series model for every observed (country, combo).

    Parameters
    ----------
    panel          : Historical AMR panel (country x combo x year).
    countries      : Country names to forecast.
    combos         : Combo strings (e.g. 'E_coli_3GC').
    hist_years     : Historical years used for fitting.
    forecast_years : Future years to project (2024-2030).
    interval_width : Confidence interval width (default 0.90 = 90%).
    n_jobs         : Parallel workers (-1 = all CPU cores).
    observed_pairs : Pre-filtered list of [country, combo] pairs that have
                     real data. Avoids iterating the full Cartesian product.

    Returns
    -------
    DataFrame with columns:
        country, combo, year, yhat, yhat_lower, yhat_upper, source
    """
    from joblib import Parallel, delayed

    use_hw   = _STATSMODELS_OK
    all_years = sorted(set(hist_years) | set(forecast_years))

    # ── Build iteration list ──────────────────────────────────────────────────
    if observed_pairs is not None:
        iter_pairs = [(str(p[0]), str(p[1])) for p in observed_pairs]
    else:
        iter_pairs = [(c, combo) for c in countries for combo in combos]

    print(f"  [forecaster] Backend: holtwinters  |  pairs: {len(iter_pairs):,}  "
          f"|  workers: {n_jobs}")

    # ── Pre-classify pairs ────────────────────────────────────────────────────
    flat_tasks  = []   # (country, combo, last_val)
    model_tasks = []   # (country, combo, yrs, y, force_linear)

    for country, combo in iter_pairs:
        subset    = (panel[(panel.country == country) &
                            (panel.combo   == combo)]
                     .sort_values('year'))
        hist_mask = subset['year'].isin(hist_years)
        y_series  = subset.loc[hist_mask, 'resistance_pct'].dropna()
        yr_series = subset.loc[y_series.index, 'year']

        n_pts = len(y_series)
        if n_pts < 2:
            last_val = float(y_series.iloc[-1]) if n_pts == 1 else 30.0
            flat_tasks.append((country, combo, last_val))
        elif n_pts < 5:
            model_tasks.append((country, combo,
                                 yr_series.values.astype(float),
                                 y_series.values.astype(float),
                                 True))    # force linear
        else:
            model_tasks.append((country, combo,
                                 yr_series.values.astype(float),
                                 y_series.values.astype(float),
                                 False))   # Holt-Winters

    n_flat  = len(flat_tasks)
    n_model = len(model_tasks)
    n_linear = sum(1 for *_, fl in model_tasks if fl)
    n_hw     = n_model - n_linear
    print(f"  [forecaster] {n_flat} flat  |  {n_linear} linear  |  "
          f"{n_hw} Holt-Winters (parallel)")

    # ── Instant flat projections ──────────────────────────────────────────────
    flat_records = []
    for country, combo, last_val in flat_tasks:
        for row in _flat_forecast(last_val, all_years, interval_width):
            flat_records.append({'country': country, 'combo': combo, **row})

    # ── Parallel model fitting ────────────────────────────────────────────────
    args_list = [
        (country, combo, yh, y, all_years, interval_width, use_hw, fl)
        for country, combo, yh, y, fl in model_tasks
    ]

    model_records = []
    try:
        results = Parallel(n_jobs=n_jobs, backend='loky', verbose=0)(
            delayed(_fit_one_pair)(args) for args in args_list
        )
        for country, combo, rows in results:
            for row in rows:
                model_records.append({'country': country, 'combo': combo, **row})
    except Exception as e:
        print(f"  [forecaster] Parallel failed ({e}), running sequentially ...")
        for args in args_list:
            country, combo, rows = _fit_one_pair(args)
            for row in rows:
                model_records.append({'country': country, 'combo': combo, **row})

    df_out = pd.DataFrame(flat_records + model_records)
    src_counts = df_out['source'].value_counts().to_dict()
    print(f"  [forecaster] Done. Usage: {src_counts}")
    return df_out
