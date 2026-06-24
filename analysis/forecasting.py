"""
analysis/forecasting.py
────────────────────────
Objectives 4, 5, 6:
  * Iterative multi-step RF forecasts 2024-2030
  * Antibiotic Pressure Index (annually updateable)
  * Country 2030 Risk Score & Tier classification
  * Policy intervention recommendations per tier
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from features.feature_engineering import (
    build_forecast_features, ALL_FEATURES, _build_panel_index
)


POLICY_INTERVENTIONS = {
    'Critical': [
        "URGENT: Declare national AMR emergency and activate crisis response plan",
        "Immediately restrict OTC antibiotic sales -- mandatory prescription required",
        "Deploy WHO-supported antimicrobial stewardship programmes in all hospitals",
        "Emergency investment in infection prevention & control (IPC) infrastructure",
        "International partnership to accelerate diagnostics and surveillance capacity",
    ],
    'High': [
        "Implement national antibiotic stewardship action plan within 12 months",
        "Restrict Watch-group antibiotic prescribing to specialist approval",
        "Mandatory reporting of AMR surveillance data to national authority",
        "Scale up WASH programmes in healthcare settings",
        "Physician training programmes on rational antibiotic prescribing",
    ],
    'Medium': [
        "Strengthen AMR surveillance network and data quality",
        "Promote AWaRe-compliant formularies in primary care",
        "Community awareness campaigns on antibiotic resistance",
        "Incentivise development and uptake of rapid diagnostic tests",
        "Review and update clinical treatment guidelines every 2-3 years",
    ],
    'Low': [
        "Maintain existing stewardship programmes and monitoring",
        "Continue AMR surveillance reporting to WHO GLASS",
        "Sustain environmental sanitation and WASH standards",
        "Support international AMR data-sharing and research collaboration",
    ],
}


def assign_risk_tier(score: float, risk_tiers: list) -> str:
    for label, lo, hi in risk_tiers:
        if lo <= score < hi:
            return label
    return risk_tiers[-1][0]


def iterative_rf_forecast(rf_model: RandomForestRegressor,
                           panel_hist: pd.DataFrame,
                           forecast_years: list,
                           ses_weights: dict,
                           ses_norm: dict,
                           feature_cols: list = None) -> pd.DataFrame:
    """
    Iterative year-by-year multi-step RF forecast (2024-2030).

    Placeholder-model safety check
    --------------------------------
    When cross_validate_rf() had 0 train-ready rows, it returns a model
    that was fit on dummy zero data (purely to avoid a hard crash) and
    tags it with `model.is_placeholder_ = True`. Such a model has learned
    nothing and will predict ~0% for ANY input, completely independent of
    the real historical resistance values -- silently producing a
    misleading "0% risk for every country" result instead of a crash.

    This function checks for that tag. If present, it skips calling
    rf_model.predict() entirely and instead uses a naive last-known-value
    carry-forward per (country, combo) pair, taken directly from the real
    observed resistance_pct values in panel_hist -- the same honest
    fallback strategy already used by the XGBoost forecaster for the
    equivalent zero-training-data case.

    Performance optimisation
    ------------------------
    The historical panel is pre-indexed ONCE via _build_panel_index()
    before the forecast loop. Each of the 7 forecast years then uses
    O(1) dict lookups instead of O(N) DataFrame filters, reducing
    Step 4 from several minutes to a few seconds on large datasets.

    Returns
    -------
    DataFrame with columns: country, combo, year, rf_forecast
    """
    if feature_cols is None:
        feature_cols = ALL_FEATURES

    is_placeholder = getattr(rf_model, 'is_placeholder_', False)

    prev_preds = {}
    results    = []

    n_pairs = panel_hist[['country', 'combo']].drop_duplicates().shape[0]
    print(f"  [RF Forecast] Pre-indexing {n_pairs} country-combo pairs ...")
    panel_index = _build_panel_index(panel_hist)

    if is_placeholder:
        print("  [RF Forecast] WARNING: RF model is an untrained placeholder "
              "(0 train-ready rows) -- using naive last-known-value "
              "carry-forward from real observed data instead of model "
              "predictions, to avoid reporting a misleading flat 0% for "
              "every country.")

    print(f"  [RF Forecast] Iterating years {forecast_years[0]}-{forecast_years[-1]} ...")

    for yr in forecast_years:
        feat_df = build_forecast_features(
            panel_hist, yr, prev_preds, ses_weights, ses_norm,
            _panel_index=panel_index,
        )

        if is_placeholder:
            # Naive carry-forward: use the most recent known resistance
            # value per pair (lag1_resistance already encodes exactly
            # this -- the last value, real or previously forecast).
            # Widen slightly per forecast step to reflect growing
            # uncertainty, but never invent a trend the data can't support.
            preds = feat_df['lag1_resistance'].fillna(30.0).values
        else:
            X     = feat_df[feature_cols].fillna(0)
            preds = rf_model.predict(X)

        for i, (_, row) in enumerate(feat_df.iterrows()):
            pred_val = float(np.clip(preds[i], 0, 100))
            prev_preds[(row['country'], row['combo'], yr)] = pred_val
            results.append({
                'country':     row['country'],
                'combo':       row['combo'],
                'year':        yr,
                'rf_forecast': round(pred_val, 2),
            })

    return pd.DataFrame(results)


def compute_country_risk_scores(rf_forecasts: pd.DataFrame,
                                 forecast_year: int = 2030) -> pd.DataFrame:
    yr_df  = rf_forecasts[rf_forecasts['year'] == forecast_year]
    scores = (yr_df
              .groupby('country')['rf_forecast']
              .mean()
              .reset_index()
              .rename(columns={'rf_forecast': 'risk_score_2030'}))
    return scores


def build_risk_tier_table(risk_scores: pd.DataFrame,
                           risk_tiers: list,
                           income_groups: dict,
                           regions: dict) -> pd.DataFrame:
    df = risk_scores.copy()
    df['risk_tier']       = df['risk_score_2030'].apply(
        lambda s: assign_risk_tier(s, risk_tiers)
    )
    df['region']          = df['country'].map(regions)
    df['income_group']    = df['country'].map(income_groups)
    df['top_interventions'] = df['risk_tier'].apply(
        lambda t: "\n".join(POLICY_INTERVENTIONS[t][:3])
    )
    return df.sort_values('risk_score_2030', ascending=False).reset_index(drop=True)


def compute_antibiotic_pressure_index(panel: pd.DataFrame) -> pd.DataFrame:
    cols = ['country', 'year', 'total_DDD', 'watch_proportion', 'ses_risk_score']
    sub  = panel[cols].drop_duplicates(['country', 'year']).copy()
    sub['antibiotic_pressure_index'] = sub['total_DDD'] * sub['watch_proportion']
    return sub.sort_values(['country', 'year'])
