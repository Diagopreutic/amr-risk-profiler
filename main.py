#!/usr/bin/env python3
"""
main.py — AMR Temporal Forecasting Challenge
═════════════════════════════════════════════
Vivli AMR Surveillance Data Challenge 2026

Usage
─────
  # Synthetic mode (no files needed — demo / testing)
  python main.py

  # Single real AMR file
  python main.py --amr glass_data.csv

  # Multiple real AMR files (any mix of CSV / Excel / schemas)
  python main.py --amr atlas.csv glass_2022.xlsx earsnet.csv

  # Full options
  python main.py \\
      --amr atlas.csv glass.xlsx \\
      --aware who_aware_2023.csv \\
      --conflict mean \\
      --combos E_coli_3GC K_pneumoniae_CARB S_aureus_MRSA \\
      --countries India Nigeria Germany

Pipeline steps
──────────────
  Step 1  Data Assembly & Harmonisation
  Step 2  Feature Engineering
  Step 3  Model Training  (RF + Ridge, TimeSeriesSplit CV)
  Step 4  Multi-Step RF Forecasting  (2024–2030)
  Step 5  XGBoost Quantile Forecasts  (independent validation)
  Step 6  Country Risk Scoring & Tier Classification
  Step 7  Analytical Questions & Feature Importance
  Step 8  All Visualisations (8 figures)
  Step 9  Export Results + Summary Report
"""

import argparse
import os
import re
import sys
import time
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(__file__))

from config import (
    COUNTRIES, COUNTRY_CODES, INCOME_GROUPS, REGIONS,
    PATHOGEN_DRUG_COMBOS, COMBO_LABELS,
    FORECAST_YEARS,
    WB_INDICATORS, RISK_TIERS, RISK_TIER_COLORS,
    SES_WEIGHTS, SES_NORM, RF_PARAMS, RIDGE_ALPHA,
    TSCV_N_SPLITS, FORECAST_CI_WIDTH, GBM_N_ESTIMATORS,
    OUTPUT_DIR, FIGURES_DIR, RESULTS_DIR,
)
from data.data_loader import assemble_panel, assemble_panel_from_files
from data.multi_file_loader import data_quality_report
from features.feature_engineering import (
    build_features, get_train_ready,
    ALL_FEATURES, TEMPORAL_FEATURES, CONSUMPTION_FEATURES,
    SOCIOECONOMIC_FEATURES, TARGET,
)
from models.supervised_models  import cross_validate_rf, get_ridge_coefficients
from models.gbm_forecaster import (run_gbm_forecasts, GBM_BACKEND_NAME,
                                    get_gbm_feature_importances, GBM_FEATURES)
from analysis.forecasting      import (
    iterative_rf_forecast, compute_country_risk_scores,
    build_risk_tier_table, compute_antibiotic_pressure_index,
)
from analysis.visualisations   import (
    plot_resistance_trends, plot_ses_vs_resistance, plot_gdp_gradient,
    plot_ridge_coefficients, plot_feature_importance,
    plot_gbm_feature_importance, plot_gbm_forecasts,
    plot_risk_tier_heatmap, plot_api_bubble,
)


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        description="AMR Temporal Forecasting Pipeline (Vivli 2026)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        '--amr', nargs='+', metavar='FILE',
        help='One or more AMR surveillance files (.csv / .xlsx). '
             'If omitted, synthetic data is used.',
    )
    p.add_argument(
        '--aware', metavar='FILE', default=None,
        help='Optional WHO AWaRe consumption CSV '
             '(columns: country, year, total_DDD, watch_proportion). '
             'Falls back to synthetic if omitted.',
    )
    p.add_argument(
        '--conflict', default='mean',
        choices=['mean', 'median', 'max', 'first', 'last'],
        help='Strategy for resolving overlapping (country, combo, year) entries '
             'across multiple files. Default: mean.',
    )
    p.add_argument(
        '--combos', nargs='+', metavar='COMBO', default=None,
        help='Restrict to specific pathogen-drug combos, '
             'e.g. E_coli_3GC K_pneumoniae_CARB. '
             'Default: all combos found in data.',
    )
    p.add_argument(
        '--countries', nargs='+', metavar='COUNTRY', default=None,
        help='Restrict to specific countries (use exact names, quoted if spaces). '
             'Default: all countries found in data.',
    )
    p.add_argument(
        '--schemas', nargs='+', metavar='FILE=SCHEMA', default=None,
        help='Override schema detection per file, '
             'e.g. my_data.csv=atlas other.xlsx=glass. '
             'Valid schemas: glass | atlas | earsnet | generic.',
    )
    p.add_argument(
        '--outdir', default=None,
        help='Override base output directory. Default: outputs/',
    )
    p.add_argument(
        '--start-year', dest='start_year', type=int, default=None,
        metavar='YEAR',
        help='First historical year (e.g. 2004 or 2010). '
             'If omitted in real-file mode you will be asked interactively. '
             'In synthetic mode 2010 is always used.',
    )
    return p.parse_args()


# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════

def banner(step: int, title: str):
    print(f"\n{'═'*72}\n  STEP {step}: {title}\n{'═'*72}")


def section(title: str):
    print(f"\n  ── {title}")


def assign_risk_tier_local(score: float) -> str:
    for label, lo, hi in RISK_TIERS:
        if lo <= score < hi:
            return label
    return RISK_TIERS[-1][0]


def _parse_schema_overrides(raw: list | None) -> dict:
    """Parse ['file.csv=glass', 'other.xlsx=atlas'] → {'file.csv': 'glass', ...}"""
    if not raw:
        return {}
    result = {}
    for item in raw:
        if '=' not in item:
            print(f"  ⚠  Ignoring malformed --schemas entry '{item}' (expected FILE=SCHEMA)")
            continue
        fname, schema = item.split('=', 1)
        result[fname.strip()] = schema.strip()
    return result


# ═══════════════════════════════════════════════════════════════════════════
# Main pipeline
# ═══════════════════════════════════════════════════════════════════════════


# ── Start-year helper ─────────────────────────────────────────────────────────

def _peek_oldest_year(file_paths: list) -> int:
    """
    Quickly scan AMR files to find the oldest year present in the data.
    Reads only the year column to keep it fast on large files.
    Returns the oldest year found, or 2010 as a safe fallback.

    Uses fuzzy substring matching (not exact match) so columns like
    'Year Collected', 'Date_Collected_Year', or 'collection year' are
    still recognised -- not just an exact column named 'Year'.
    """
    oldest = None
    year_tokens = ['year', 'survey_year', 'collection_year', 'study_year',
                   'isolate_year', 'time']

    def _find_year_col(columns):
        """Fuzzy match: normalise each column, check if any year token
        appears as a substring (handles 'Year Collected', 'CollectionYear',
        'Year_of_Isolation', etc.)."""
        norm_cols = {re.sub(r'[\s_-]+', '_', c.strip().lower()): c
                    for c in columns}
        # Prefer an exact 'year' match first
        if 'year' in norm_cols:
            return norm_cols['year']
        # Then any column containing a year token
        for norm, orig in norm_cols.items():
            if any(tok in norm for tok in year_tokens):
                return orig
        return None

    for fp in file_paths:
        try:
            ext = fp.lower().split('.')[-1]
            if ext == 'csv':
                sample = pd.read_csv(fp, nrows=5, low_memory=False)
                year_col = _find_year_col(sample.columns)
                if year_col:
                    all_years = pd.read_csv(fp, usecols=[year_col], low_memory=False)
                    yr = int(pd.to_numeric(all_years[year_col],
                                           errors='coerce').dropna().min())
                    oldest = yr if oldest is None else min(oldest, yr)
            else:
                df = pd.read_excel(fp)
                year_col = _find_year_col(df.columns)
                if year_col:
                    yr = int(pd.to_numeric(df[year_col],
                                           errors='coerce').dropna().min())
                    oldest = yr if oldest is None else min(oldest, yr)
        except Exception:
            pass
    return oldest if oldest is not None else 2010


def _determine_start_year(args, file_paths: list) -> int:
    """
    Determine the historical start year through three paths:

    1. --start-year N was passed on the CLI → use N directly (no prompt).
    2. Interactive prompt (real-file mode only):
         a. User answers yes  → asks for the year to type in.
         b. User answers no   → scans the data files and uses the
                                oldest year found in them.
    3. Synthetic mode (no files) → always returns 2010.
    """
    # Path 1: explicit CLI argument
    if args.start_year is not None:
        print(f"  Start year set via --start-year: {args.start_year}")
        return args.start_year

    # Path 3: synthetic mode
    if not file_paths:
        return 2010

    # Path 2: interactive prompt
    print()
    print("  ──────────────────────────────────────────────────────────")
    print("  START YEAR CONFIGURATION")
    print("  ──────────────────────────────────────────────────────────")
    print("  Do you want to set a custom starting year for the analysis?")
    print("  (If no, the oldest year found in your data will be used.)")
    print()

    while True:
        choice = input("  Enter y (yes) or n (no): ").strip().lower()
        if choice in ('y', 'yes'):
            while True:
                try:
                    year_in = int(input("  Enter starting year (e.g. 2010): ").strip())
                    if 1990 <= year_in <= 2023:
                        print(f"  Starting year set to: {year_in}")
                        return year_in
                    else:
                        print("  Please enter a year between 1990 and 2023.")
                except ValueError:
                    print("  Invalid input. Please enter a numeric year.")
        elif choice in ('n', 'no'):
            print("  Scanning data files for oldest year ...")
            oldest = _peek_oldest_year(file_paths)
            print(f"  Oldest year found in data: {oldest}")
            print(f"  Starting year set to: {oldest}")
            return oldest
        else:
            print("  Please type 'y' or 'n'.")


def main():
    args = parse_args()
    t0   = time.time()

    # ── Output directories ────────────────────────────────────────────────
    base_out = args.outdir or OUTPUT_DIR
    fig_dir  = os.path.join(base_out, 'figures')
    res_dir  = os.path.join(base_out, 'results')
    for d in [base_out, fig_dir, res_dir]:
        os.makedirs(d, exist_ok=True)

    print("╔══════════════════════════════════════════════════════════════════╗")
    print("║   AMR Temporal Forecasting — Vivli Challenge 2026               ║")
    print("║   Multi-model framework: RF + Ridge + XGBoost                  ║")
    print("╚══════════════════════════════════════════════════════════════════╝")

    mode = "REAL FILES" if args.amr else "SYNTHETIC"
    print(f"\n  Data mode : {mode}")
    if args.amr:
        for f in args.amr:
            print(f"    * {f}")

    # ── Determine historical start year (before Step 1 needs ALL_YEARS) ──────
    hist_start       = _determine_start_year(args, args.amr or [])
    HISTORICAL_YEARS = list(range(hist_start, 2024))
    ALL_YEARS        = HISTORICAL_YEARS + FORECAST_YEARS
    print(f"\n  Historical range : {hist_start}–2023  "
          f"({len(HISTORICAL_YEARS)} years)")
    print(f"  Forecast range   : 2024–2030")

    # ──────────────────────────────────────────────────────────────────────
    banner(1, "Data Assembly & Harmonisation")
    # ──────────────────────────────────────────────────────────────────────

    if args.amr:
        # ── REAL FILE MODE ────────────────────────────────────────────────
        schema_overrides = _parse_schema_overrides(args.schemas)
        panel = assemble_panel_from_files(
            amr_file_paths    = args.amr,
            years             = ALL_YEARS,
            country_codes     = COUNTRY_CODES,
            wb_indicators     = WB_INDICATORS,
            conflict_strategy = args.conflict,
            force_schemas     = schema_overrides,
            filter_combos     = args.combos,
            filter_countries  = args.countries,
            aware_file_path   = args.aware,
            seed              = 42,
            default_year      = hist_start,
        )

        # Derive dynamic lists from real data
        countries_used = sorted(panel['country'].unique().tolist())
        combos_used    = sorted(panel['combo'].unique().tolist())

        # Build income / region maps for discovered countries
        # (use config maps where available, fall back to 'Unknown')
        income_groups_used = {c: INCOME_GROUPS.get(c, 'Unknown') for c in countries_used}
        regions_used       = {c: REGIONS.get(c, 'Unknown')       for c in countries_used}

        # Derive PATHOGEN_DRUG_COMBOS from actual data for later use
        pd_map = (panel[['combo', 'pathogen', 'drug']]
                  .drop_duplicates()
                  .set_index('combo'))
        combos_tuples = [(pd_map.loc[c, 'pathogen'], pd_map.loc[c, 'drug'])
                          for c in combos_used if c in pd_map.index]
        combo_labels_used = {
            (p, d): COMBO_LABELS.get((p, d), f"{p} / {d}")
            for p, d in combos_tuples
        }

    else:
        # ── SYNTHETIC MODE (original behaviour) ──────────────────────────
        panel = assemble_panel(
            years         = ALL_YEARS,
            countries     = COUNTRIES,
            combos        = PATHOGEN_DRUG_COMBOS,
            country_codes = COUNTRY_CODES,
            wb_indicators = WB_INDICATORS,
            seed          = 42,
        )
        countries_used     = COUNTRIES
        combos_used        = [f"{p}_{d}" for p, d in PATHOGEN_DRUG_COMBOS]
        combos_tuples      = PATHOGEN_DRUG_COMBOS
        income_groups_used = INCOME_GROUPS
        regions_used       = REGIONS
        combo_labels_used  = COMBO_LABELS

    # Print data quality report for real-file mode
    if args.amr:
        from data.multi_file_loader import data_quality_report as dqr
        # Reconstruct AMR-only frame from panel for quality check
        amr_cols = ['country','combo','year','resistance_pct']
        _amr_check = panel[amr_cols].dropna(subset=['resistance_pct'])
        if len(_amr_check):
            print(dqr(_amr_check))

    panel['income_group'] = panel['country'].map(income_groups_used).fillna('Unknown')
    panel['region']       = panel['country'].map(regions_used).fillna('Unknown')

    panel.to_csv(f"{res_dir}/panel_raw.csv", index=False)
    print(f"\n  Panel shape : {panel.shape}")
    print(f"  Countries   : {panel.country.nunique()}")
    print(f"  Combos      : {panel.combo.nunique()}")
    print(f"  Year range  : {panel.year.min()}–{panel.year.max()}")
    print(f"  Resistance  : {panel.resistance_pct.notna().mean()*100:.1f}% coverage")

    # ──────────────────────────────────────────────────────────────────────
    banner(2, "Feature Engineering")
    # ──────────────────────────────────────────────────────────────────────

    panel_feat = build_features(panel, SES_WEIGHTS, SES_NORM)
    ses_norm_fitted = panel_feat.attrs.get('ses_norm', SES_NORM)

    df_train = get_train_ready(panel_feat, ALL_FEATURES,
                                min_year=HISTORICAL_YEARS[0],
                                max_year=HISTORICAL_YEARS[-1])
    print(f"  Train-ready rows: {len(df_train):,}")

    panel_feat.to_csv(f"{res_dir}/panel_features.csv", index=False)
    df_train.to_csv(f"{res_dir}/train_ready.csv", index=False)

    if len(df_train) < 30:
        print("\n  ⚠  Very few training rows — model accuracy will be limited.")
        print("     Consider adding more historical files or broadening year range.")

    # ──────────────────────────────────────────────────────────────────────
    banner(3, "Model Training — RF Cross-Validation + Ridge Baseline")
    # ──────────────────────────────────────────────────────────────────────

    section("Random Forest (TimeSeriesSplit, 5 folds)")
    cv_results = cross_validate_rf(
        df_train, ALL_FEATURES, TARGET, RF_PARAMS, TSCV_N_SPLITS
    )

    rf_model     = cv_results['model']
    ridge_scaler = cv_results['scaler_ridge']
    ridge_model  = cv_results['model_ridge']
    fi           = cv_results['feature_importances']
    cv_mae       = cv_results['mean_mae']
    cv_std       = cv_results['std_mae']

    section("Ridge Coefficients")
    ridge_coefs = get_ridge_coefficients(ALL_FEATURES, ridge_scaler, ridge_model)

    section("Objective 7 — Feature Group Importance")
    fi_temporal = fi[fi.index.isin(TEMPORAL_FEATURES)].sum()
    fi_consump  = fi[fi.index.isin(CONSUMPTION_FEATURES)].sum()
    fi_socioec  = fi[fi.index.isin(SOCIOECONOMIC_FEATURES)].sum()
    fi_total    = fi_temporal + fi_consump + fi_socioec

    print(f"\n  Feature Group Importances:")
    if fi_total > 0:
        print(f"    Temporal autocorrelation : {fi_temporal/fi_total*100:5.1f}%")
        print(f"    Antibiotic consumption   : {fi_consump/fi_total*100:5.1f}%")
        print(f"    Socioeconomic factors    : {fi_socioec/fi_total*100:5.1f}%")
    else:
        print(f"    N/A -- no model was trained (0 train-ready rows).")
    print(f"\n  Top-5 features:")
    for feat, imp in fi.head(5).items():
        print(f"    {feat:<35s}  {imp:.4f}")

    pd.DataFrame({'feature': fi.index, 'importance': fi.values}
                 ).to_csv(f"{res_dir}/feature_importances.csv", index=False)
    pd.DataFrame({'feature': ridge_coefs.index, 'coefficient': ridge_coefs.values}
                 ).to_csv(f"{res_dir}/ridge_coefficients.csv", index=False)

    # panel_hist is needed both for the CSV substitute below and for
    # Step 4 (RF iterative forecast). Define it once here so it is in
    # scope for both uses.
    panel_hist = panel_feat[panel_feat.year.isin(HISTORICAL_YEARS)].copy()

    # When the models are placeholders (0 train-ready rows), the saved CSVs
    # contain all-zero values which are misleading. Append a more meaningful
    # supplementary CSV using real observed resistance data instead.
    is_placeholder = cv_results.get('is_placeholder', False)
    if is_placeholder:
        obs_res = (panel_hist.dropna(subset=['resistance_pct'])
                   .groupby(['country', 'combo'])['resistance_pct']
                   .agg(mean_resistance='mean', n_isolates='count')
                   .reset_index())
        obs_res.to_csv(f"{res_dir}/observed_resistance_summary.csv", index=False)
        print(f"  NOTE: RF/Ridge models are placeholders (0 train-ready rows). "
              f"Observed resistance summary saved to "
              f"observed_resistance_summary.csv instead.")

    # ──────────────────────────────────────────────────────────────────────
    banner(4, "Multi-Step RF Forecasting (2024–2030)")
    # ──────────────────────────────────────────────────────────────────────

    rf_forecasts = iterative_rf_forecast(
        rf_model       = rf_model,
        panel_hist     = panel_hist,
        forecast_years = FORECAST_YEARS,
        ses_weights    = SES_WEIGHTS,
        ses_norm       = ses_norm_fitted,
    )
    rf_forecasts.to_csv(f"{res_dir}/rf_forecasts_2024_2030.csv", index=False)
    print(f"  RF forecasts: {len(rf_forecasts):,} rows")

    # ──────────────────────────────────────────────────────────────────────
    banner(5, "XGBoost Quantile Forecasts (Independent Validation)")
    # ──────────────────────────────────────────────────────────────────────

    # Only pass (country, combo) pairs that actually have historical data.
    # Passing the full Cartesian product (countries x combos) causes the
    # forecaster to iterate empty combinations — the root cause of Step 5
    # taking several minutes despite parallelisation.
    observed_pairs = (panel_hist[['country', 'combo']]
                      .drop_duplicates()
                      .values.tolist())
    obs_countries = list(dict.fromkeys(p[0] for p in observed_pairs))
    obs_combos    = list(dict.fromkeys(p[1] for p in observed_pairs))
    print(f"  Forecaster backend: XGBoost")
    print(f"  Forecasting {len(observed_pairs):,} observed pairs "
          f"({len(obs_countries)} countries x {len(obs_combos)} unique combos)")

    gbm_df = run_gbm_forecasts(
        panel          = panel_hist,
        countries      = obs_countries,
        combos         = obs_combos,
        hist_years     = HISTORICAL_YEARS,
        forecast_years = FORECAST_YEARS,
        interval_width = FORECAST_CI_WIDTH,
        observed_pairs = observed_pairs,
        n_estimators   = GBM_N_ESTIMATORS,
    )
    gbm_df.to_csv(f"{res_dir}/gbm_forecasts.csv", index=False)
    print(f"  GBM forecasts: {len(gbm_df):,} rows")

    # Extract XGBoost feature importances (stored in DataFrame attrs)
    gbm_fi = get_gbm_feature_importances(gbm_df)
    gbm_na = (gbm_fi.name == 'gbm_unavailable' or gbm_fi.sum() == 0)
    if not gbm_fi.empty:
        gbm_fi.reset_index().rename(columns={'index':'feature',
            'gbm_importance':'importance'}).to_csv(
            f"{res_dir}/gbm_feature_importances.csv", index=False)
        if gbm_na:
            print(f"  GBM feature importances: unavailable (zero-variance or "
                  f"naive fallback) -- CSV saved with placeholder zeros.")
            print(f"  See observed_resistance_summary.csv for meaningful data.")
        else:
            print(f"  GBM feature importances saved ({len(gbm_fi)} features)")

    # ──────────────────────────────────────────────────────────────────────
    banner(6, "Country Risk Scoring & Tier Classification (2030)")
    # ──────────────────────────────────────────────────────────────────────

    risk_scores = compute_country_risk_scores(rf_forecasts, forecast_year=2030)
    risk_table  = build_risk_tier_table(
        risk_scores, RISK_TIERS, income_groups_used, regions_used
    )
    api_df = compute_antibiotic_pressure_index(panel_feat)

    print("\n  2030 Country Risk Tiers:")
    print(f"  {'Country':<22} {'Score':>10}  {'Tier':<10}  Region")
    print(f"  {'-'*66}")
    for _, r in risk_table.iterrows():
        print(f"  {r['country']:<22} {r['risk_score_2030']:>8.1f}%   "
              f"{r['risk_tier']:<10}  {r['region']}")

    risk_table.to_csv(f"{res_dir}/country_risk_tiers_2030.csv", index=False)
    api_df.to_csv(f"{res_dir}/antibiotic_pressure_index.csv", index=False)

    # ──────────────────────────────────────────────────────────────────────
    banner(7, "Analytical Questions — Summary Answers")
    # ──────────────────────────────────────────────────────────────────────

    from scipy import stats as sp_stats

    hist_panel = panel_feat[panel_feat.year.isin(HISTORICAL_YEARS)]

    section(f"Q1 — Steepest resistance trajectories ({hist_start}-{HISTORICAL_YEARS[-1]})")

    def _safe_slope(g):
        """
        Compute the OLS trend slope (pp/year) for a country group.
        Returns np.nan instead of crashing when:
          - fewer than 3 valid data points
          - all years are identical (zero variance in X)
          - SVD does not converge (Intel MKL DGELSD error)
          - any other numerical failure
        """
        valid = g.dropna(subset=['year', 'resistance_pct'])
        if len(valid) < 3:
            return np.nan
        yrs = valid['year'].values.astype(float)
        res = valid['resistance_pct'].values.astype(float)
        # Polyfit requires at least 2 distinct X values
        if np.unique(yrs).size < 2:
            return np.nan
        try:
            return float(np.polyfit(yrs, res, 1)[0])
        except Exception:
            # Fallback: simple manual slope via least-squares formula
            try:
                x = yrs - yrs.mean()
                denom = (x ** 2).sum()
                return float((x * res).sum() / denom) if denom > 0 else np.nan
            except Exception:
                return np.nan

    slopes = (hist_panel.groupby('country')
              .apply(_safe_slope)
              .dropna()
              .sort_values(ascending=False))
    for c, s in slopes.head(6).items():
        print(f"    {c:<22}  {s:+.3f} pp/year")

    section("Q2 — Feature group importance ranking")
    if fi_total > 0:
        print(f"    Temporal: {fi_temporal/fi_total*100:.1f}%  |  "
              f"Consumption: {fi_consump/fi_total*100:.1f}%  |  "
              f"Socioeconomic: {fi_socioec/fi_total*100:.1f}%")
    else:
        print(f"    N/A -- no model was trained (0 train-ready rows).")

    section("Q2b -- XGBoost feature importances (modifiable drivers)")
    xgb_na = (gbm_fi.name == "gbm_unavailable" or gbm_fi.sum() == 0)
    if xgb_na:
        print("    XGBoost importances unavailable: training target has near-zero")
        print("    variance (dataset predominantly susceptible / all-zero resistance).")
        print("    Fig 5b shows RF importances as fallback.")
        print("    Top modifiable drivers from Random Forest:")
        mod_rf = fi[[f for f in fi.index if not any(t in f for t in
                     ["lag","rolling","delta","year","pathogen_enc","drug_enc","country_enc"])]]
        for feat, imp in mod_rf.head(5).items():
            print("      {:<35s}  {:.2f}%".format(feat, imp*100))
    elif not gbm_fi.empty:
        temporal_feats = [f for f in gbm_fi.index if any(t in f for t in ["lag","rolling","delta","year"])]
        consump_feats  = [f for f in gbm_fi.index if any(t in f for t in ["DDD","watch","pressure"])]
        identity_feats = [f for f in gbm_fi.index if "_enc" in f]
        socio_feats    = [f for f in gbm_fi.index if f not in temporal_feats+consump_feats+identity_feats]
        print("    XGBoost global model importances (gain-based):")
        print("      Temporal autocorrelation : {:.1f}%".format(gbm_fi[temporal_feats].sum()*100))
        print("      Antibiotic consumption   : {:.1f}%".format(gbm_fi[consump_feats].sum()*100))
        print("      Pathogen/drug identity   : {:.1f}%".format(gbm_fi[identity_feats].sum()*100))
        print("      Socioeconomic factors    : {:.1f}%".format(gbm_fi[socio_feats].sum()*100))
        print("    Top modifiable drivers (non-temporal, non-identity):")
        modifiable = gbm_fi[[f for f in gbm_fi.index if f not in temporal_feats+identity_feats]]
        for feat, imp in modifiable.head(5).items():
            print("      {:<35s}  {:.2f}%".format(feat, imp*100))
    else:
        print("    GBM importances not available.")

    section("Q3 — GDP per capita vs resistance (Pearson r)")
    agg_gdp = (hist_panel.dropna(subset=['gdp_per_capita','resistance_pct'])
               .groupby('country')
               .agg(mean_res=('resistance_pct','mean'),
                    mean_gdp=('gdp_per_capita','mean'))
               .reset_index())
    if len(agg_gdp) >= 3:
        r_gdp, p_gdp = sp_stats.pearsonr(np.log(agg_gdp.mean_gdp + 1), agg_gdp.mean_res)
        print(f"    Pearson r (log-GDP vs resistance): {r_gdp:.3f}  (p={p_gdp:.4f})")
    else:
        r_gdp, p_gdp = np.nan, np.nan
        print("    Not enough countries for correlation.")

    section("Q4 — Watch-group proportion vs resistance")
    valid = hist_panel.dropna(subset=['watch_proportion','resistance_pct'])
    if len(valid) >= 10:
        r_w, p_w = sp_stats.pearsonr(valid.watch_proportion, valid.resistance_pct)
        print(f"    Pearson r: {r_w:.3f}  (p={p_w:.4f})")
    else:
        r_w, p_w = np.nan, np.nan
        print("    Insufficient data for correlation.")

    section("Q5 — 2030 High / Critical countries")
    hc = risk_table[risk_table.risk_tier.isin(['High','Critical'])]
    if hc.empty:
        print("    None projected at High/Critical tier.")
    for _, r in hc.iterrows():
        print(f"    {r['country']:<22}  {r['risk_score_2030']:.1f}%  [{r['risk_tier']}]")

    section("Q6 — Most protective factors (negative Ridge coefficients)")
    for feat, coef in ridge_coefs[ridge_coefs < 0].sort_values().head(5).items():
        print(f"    {feat:<35s}  {coef:+.4f}")

    section("Q7 — Model accuracy")
    if not np.isnan(cv_mae):
        print(f"    RF CV-MAE: {cv_mae:.3f}% +/- {cv_std:.3f}%")
    else:
        print(f"    RF CV-MAE: N/A -- insufficient data for cross-validation.")
    gbm_2030 = (gbm_df[gbm_df.year == 2030]
                .groupby('country')['yhat'].mean()
                .reset_index()
                .rename(columns={'yhat': 'gbm_2030'}))
    dir_df = risk_scores.merge(gbm_2030, on='country', how='inner')
    if len(dir_df) >= 2:
        dir_df['rf_tier'] = dir_df.risk_score_2030.apply(assign_risk_tier_local)
        dir_df['gbm_tier'] = dir_df.gbm_2030.apply(assign_risk_tier_local)
        agree = (dir_df.rf_tier == dir_df.gbm_tier).mean()
        print(f"    RF vs GBM tier agreement: {agree*100:.0f}%")

    # ──────────────────────────────────────────────────────────────────────
    banner(8, "Generating All Visualisations")
    # ──────────────────────────────────────────────────────────────────────

    print("\n  Plotting …")

    plot_resistance_trends(
        panel_feat, combos_tuples, combo_labels_used,
        countries_used, HISTORICAL_YEARS, fig_dir,
    )
    r_ses, p_ses = plot_ses_vs_resistance(
        panel_feat, HISTORICAL_YEARS, income_groups_used, fig_dir,
    )
    print(f"    SES-RS vs resistance: r = {r_ses:.3f}")
    plot_gdp_gradient(panel_feat, HISTORICAL_YEARS, income_groups_used, fig_dir)
    plot_ridge_coefficients(ridge_coefs, fig_dir, panel=panel_hist)
    plot_feature_importance(fi, fig_dir, panel=panel_hist)
    if not gbm_fi.empty:
        plot_gbm_feature_importance(
            gbm_fi=gbm_fi,
            rf_fi=fi,
            figures_dir=fig_dir,
            backend_name=GBM_BACKEND_NAME.capitalize(),
            panel=panel_hist,
        )

    # Use first available combo for GBM plot
    first_combo = combos_used[0] if combos_used else 'E_coli_3GC'
    first_combo_tuple = combos_tuples[0] if combos_tuples else ('E_coli', '3GC')
    plot_gbm_forecasts(
        panel   = panel_hist,
        gbm_df  = gbm_df,
        rf_forecasts  = rf_forecasts,
        countries_to_plot  = countries_used,
        combo_to_plot      = first_combo,
        combo_label        = combo_labels_used.get(first_combo_tuple, first_combo),
        hist_years         = HISTORICAL_YEARS,
        forecast_years     = FORECAST_YEARS,
        figures_dir        = fig_dir,
    )
    plot_risk_tier_heatmap(
        risk_table, rf_forecasts, combos_tuples, RISK_TIERS, fig_dir,
    )
    plot_api_bubble(api_df, income_groups_used, fig_dir, target_year=2022)

    # ──────────────────────────────────────────────────────────────────────
    banner(9, "Final Summary Report")
    # ──────────────────────────────────────────────────────────────────────

    elapsed = time.time() - t0
    _write_summary(
        mode, args, cv_mae, cv_std, r_ses, p_ses,
        r_gdp, p_gdp, risk_table, fi, ridge_coefs,
        fi_temporal, fi_consump, fi_socioec, fi_total,
        res_dir, elapsed,
    )

    print(f"\n{'═'*72}")
    print(f"  ✓  Pipeline complete in {elapsed:.1f}s")
    print(f"  Figures  → {fig_dir}/")
    print(f"  Results  → {res_dir}/")
    print(f"{'═'*72}\n")


# ═══════════════════════════════════════════════════════════════════════════
# Summary report
# ═══════════════════════════════════════════════════════════════════════════

def _write_summary(mode, args, cv_mae, cv_std, r_ses, p_ses,
                   r_gdp, p_gdp, risk_table, fi, ridge_coefs,
                   fi_temporal, fi_consump, fi_socioec, fi_total,
                   res_dir, elapsed):
    lines = [
        "AMR Temporal Forecasting Challenge -- Summary Report",
        "Vivli AMR Surveillance Data Challenge 2026",
        "=" * 62,
        "",
        f"DATA MODE : {mode}",
    ]
    if args.amr:
        lines += [f"  Source files ({len(args.amr)}):"] + [f"    * {f}" for f in args.amr]
        lines.append(f"  Conflict strategy : {args.conflict}")
    lines += [
        "",
        "METHODOLOGY",
        "-" * 40,
        "Primary model  : Random Forest (n=300, max_depth=10)",
        "Baseline model : Ridge Regression (alpha=1.0, standardised)",
        f"Validation     : {GBM_BACKEND_NAME} Quantile Regression (90% CI)",
        "CV strategy    : TimeSeriesSplit (5 folds, no look-ahead)",
        "",
        "MODEL PERFORMANCE",
        "-" * 40,
        f"RF CV-MAE              : {cv_mae:.3f}% +/- {cv_std:.3f}%",
        f"Reference benchmark    : 1.61% +/- 0.64%",
        "",
        "FEATURE GROUP IMPORTANCES (RF)",
        "-" * 40,
        (f"  Temporal autocorrelation : {fi_temporal/fi_total*100:.1f}%"
         if fi_total > 0 else "  Temporal autocorrelation : N/A (no model trained)"),
        (f"  Antibiotic consumption   : {fi_consump/fi_total*100:.1f}%"
         if fi_total > 0 else "  Antibiotic consumption   : N/A (no model trained)"),
        (f"  Socioeconomic factors    : {fi_socioec/fi_total*100:.1f}%"
         if fi_total > 0 else "  Socioeconomic factors    : N/A (no model trained)"),
        "",
        "TOP 8 FEATURES",
        "-" * 40,
    ]
    for feat, imp in fi.head(8).items():
        lines.append(f"  {feat:<35s}  {imp:.4f}")

    lines += [
        "",
        "KEY CORRELATIONS",
        "-" * 40,
        f"  SES-RS vs resistance (Pearson r) : {r_ses:.3f}  (p={p_ses:.4f})",
        f"  log-GDP vs resistance (Pearson r): {r_gdp:.3f}  (p={p_gdp:.4f})"
            if not np.isnan(r_gdp) else "  log-GDP vs resistance: insufficient data",
        "",
        "MOST PROTECTIVE FACTORS (negative Ridge coefficients)",
        "-" * 40,
    ]
    for feat, coef in ridge_coefs[ridge_coefs < 0].sort_values().head(5).items():
        lines.append(f"  {feat:<35s}  {coef:+.4f}")

    lines += [
        "",
        "2030 COUNTRY RISK TIERS",
        "-" * 40,
        f"  {'Country':<22}  {'Score':>8}   {'Tier':<10}  Region",
        f"  {'-'*62}",
    ]
    for _, r in risk_table.iterrows():
        lines.append(f"  {r['country']:<22}  {r['risk_score_2030']:>7.1f}%   "
                     f"{r['risk_tier']:<10}  {r['region']}")

    lines += [
        "",
        "POLICY INTERVENTIONS (top priority per high-risk country)",
        "-" * 40,
    ]
    for _, r in risk_table[risk_table.risk_tier.isin(['High','Critical'])].iterrows():
        lines.append(f"\n  {r['country']} [{r['risk_tier']}]:")
        for ln in r['top_interventions'].split("\n"):
            lines.append(f"    {ln}")

    lines += [
        "",
        "=" * 62,
        f"Pipeline runtime: {elapsed:.1f}s",
        "",
        "OUTPUT FILES",
        "-" * 40,
        f"  {res_dir}/panel_raw.csv",
        f"  {res_dir}/panel_features.csv",
        f"  {res_dir}/rf_forecasts_2024_2030.csv",
        f"  {res_dir}/gbm_forecasts.csv",
        f"  {res_dir}/gbm_feature_importances.csv",
        f"  {res_dir}/country_risk_tiers_2030.csv",
        f"  {res_dir}/feature_importances.csv",
        f"  {res_dir}/ridge_coefficients.csv",
        f"  {res_dir}/antibiotic_pressure_index.csv",
    ]

    path = f"{res_dir}/summary_report.txt"
    # Encode to ASCII (replace unknowns) so the file writes on any Windows locale,
    # then write with utf-8 to preserve the ASCII-safe result cleanly.
    safe_lines = [
        ln.encode('ascii', errors='replace').decode('ascii')
        for ln in lines
    ]
    with open(path, 'w', encoding='utf-8') as f:
        f.write("\n".join(safe_lines))
    print(f"\n  Summary report -> {path}")


if __name__ == "__main__":
    main()
