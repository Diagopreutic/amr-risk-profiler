"""
models/gbm_forecaster.py
─────────────────────────
Version-2 forecasting module.

Forecasting method: XGBoost Quantile Regression
─────────────────────────────────────────────────
A SINGLE global XGBoost model is trained across ALL country-combo
pairs simultaneously.

Why a global GBM model?
─────────────────────────
1. Transfer learning across pairs — resistance dynamics learned from
   well-surveilled countries improve projections for sparse ones.
2. Feature richness — lag features, rolling means, socioeconomic
   indicators, consumption patterns, and pathogen-drug identity all
   used as predictors.
3. Quantile regression — three models (q05/q50/q95) produce honest
   90% prediction intervals directly from the model.
4. Speed — one model fit instead of 5,000+ individual fits.

Backend
────────
XGBoost (primary and only backend).
Falls back to Ridge regression if XGBoost is unavailable.

Forecasting algorithm
──────────────────────
1. Build a supervised training set from historical panel:
   Features: lag1/2/3 resistance, rolling3 mean, delta, year,
             total_DDD, watch_proportion, ses_risk_score,
             gdp_per_capita, physicians, sanitation,
             pathogen_encoded, drug_encoded, country_encoded
   Target: resistance_pct

2. Train three LightGBM models:
   - median model  (alpha=0.50) → point forecast (yhat)
   - lower model   (alpha=0.05) → lower CI bound (yhat_lower)
   - upper model   (alpha=0.95) → upper CI bound (yhat_upper)

3. Iterative year-by-year forecast (2024-2030):
   Predict year T from features built using data up to T-1,
   then feed the prediction back as a lag for year T+1.
"""

import warnings
import logging
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")

# ── Backend selection: XGBoost only ─────────────────────────────────────────

def _get_backend():
    """Return ('xgboost', module) or ('ridge', None) if XGBoost unavailable."""
    try:
        import xgboost as xgb
        return 'xgboost', xgb
    except ImportError:
        pass
    return 'ridge', None


BACKEND_NAME, _BACKEND_MOD = _get_backend()
GBM_BACKEND_NAME = BACKEND_NAME   # public alias used by main.py


# ── Feature columns used for GBM training/prediction ─────────────────────────

GBM_FEATURES = [
    'lag1_resistance', 'lag2_resistance', 'lag3_resistance',
    'rolling3_mean_resistance', 'delta_resistance',
    'year_norm',           # year normalised to [0,1]
    'lag1_total_DDD', 'lag1_watch_proportion', 'antibiotic_pressure_index',
    'ses_risk_score', 'gdp_per_capita', 'physicians', 'sanitation',
    'pathogen_enc', 'drug_enc', 'country_enc',
]


# ── Label encoders (fitted once, reused at forecast time) ────────────────────

class _Encoders:
    def __init__(self):
        self.pathogen = LabelEncoder()
        self.drug     = LabelEncoder()
        self.country  = LabelEncoder()
        self._fitted  = False

    def fit(self, df: pd.DataFrame):
        self.pathogen.fit(df['pathogen'].astype(str))
        self.drug.fit(df['drug'].astype(str))
        self.country.fit(df['country'].astype(str))
        self._fitted = True

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        def _safe_enc(enc, col):
            known = set(enc.classes_)
            return enc.transform(
                [x if x in known else enc.classes_[0]
                 for x in df[col].astype(str)]
            )
        df['pathogen_enc'] = _safe_enc(self.pathogen, 'pathogen')
        df['drug_enc']     = _safe_enc(self.drug,     'drug')
        df['country_enc']  = _safe_enc(self.country,  'country')
        return df


_ENCODERS = _Encoders()


# ── Feature builder from training panel ──────────────────────────────────────

def _build_gbm_train_features(panel: pd.DataFrame,
                               year_min: int, year_max: int) -> pd.DataFrame:
    """
    Build supervised training rows from the historical panel.
    Requires at least lag1 to be non-null (drops first year per pair).
    """
    from features.feature_engineering import (
        build_features, get_train_ready, ALL_FEATURES, TARGET
    )
    from config import SES_WEIGHTS as SW, SES_NORM as SN

    feat_panel = build_features(panel, SW, SN)

    # Add pathogen / drug columns to the FULL panel (not just train_df) so
    # encoders below see every identity that exists, even when the strict
    # train-ready filter removes all rows (e.g. every pair has only 1 year
    # of history, so lag1_resistance is NaN everywhere and 0 rows survive).
    if 'pathogen' not in feat_panel.columns:
        feat_panel['pathogen'] = feat_panel['combo'].str.split('_').str[0]
    if 'drug' not in feat_panel.columns:
        feat_panel['drug'] = feat_panel['combo'].str.split('_').str[-1]

    # Fit label encoders on the FULL panel, not the filtered train_df.
    # This guarantees enc.classes_ is never empty, even when train_df ends
    # up with 0 rows -- the encoders must still be usable at forecast time
    # for every country/pathogen/drug that appears anywhere in the data.
    _ENCODERS.fit(feat_panel)

    df = get_train_ready(feat_panel, ALL_FEATURES, year_min, year_max)

    # Add normalised year feature
    df['year_norm'] = (df['year'] - year_min) / max(year_max - year_min, 1)

    if len(df) > 0:
        df = _ENCODERS.transform(df)
    else:
        # Ensure expected encoded columns still exist on an empty frame so
        # downstream X_train = train_df[GBM_FEATURES] doesn't KeyError.
        for col in ('pathogen_enc', 'drug_enc', 'country_enc'):
            df[col] = pd.Series(dtype=float)

    return df, feat_panel


def _build_gbm_forecast_row(country: str, combo: str, future_year: int,
                              hist_res: list, last_row: dict,
                              year_min: int, year_max: int,
                              ses_weights: dict, ses_norm: dict) -> dict:
    """Build one feature dict for a single (country, combo, year) forecast."""
    from features.feature_engineering import compute_ses_risk_score

    n = len(hist_res)
    lag1 = hist_res[-1] if n >= 1 else np.nan
    lag2 = hist_res[-2] if n >= 2 else np.nan
    lag3 = hist_res[-3] if n >= 3 else np.nan
    rol3 = float(np.mean(hist_res[-3:])) if n >= 1 else lag1
    delt = hist_res[-1] - hist_res[-2] if n >= 2 else 0.0

    gdp   = last_row.get('gdp_per_capita', 5000.0)
    hexp  = last_row.get('health_expenditure', 5.0)
    san   = last_row.get('sanitation', 70.0)
    phys  = last_row.get('physicians', 1.0)
    ddd   = last_row.get('total_DDD', 15.0)
    watch = last_row.get('watch_proportion', 0.30)

    ses = float(compute_ses_risk_score(
        pd.DataFrame([{'gdp_per_capita': gdp, 'health_expenditure': hexp,
                        'sanitation': san, 'physicians': phys}]),
        ses_weights, ses_norm
    ).iloc[0])

    pathogen = combo.split('_')[0]
    drug     = combo.split('_')[-1]

    row = {
        'country': country, 'combo': combo,
        'pathogen': pathogen, 'drug': drug,
        'year': future_year,
        'year_norm': (future_year - year_min) / max(year_max - year_min, 1),
        'lag1_resistance': lag1,
        'lag2_resistance': lag2,
        'lag3_resistance': lag3,
        'rolling3_mean_resistance': rol3,
        'delta_resistance': delt,
        'lag1_total_DDD': ddd,
        'lag1_watch_proportion': watch,
        'antibiotic_pressure_index': ddd * watch,
        'ses_risk_score': ses,
        'gdp_per_capita': gdp,
        'physicians': phys,
        'sanitation': san,
    }
    return row


def _train_xgboost(X_train: np.ndarray, y_train: np.ndarray,
                   feature_names: list, n_estimators: int = 500) -> tuple:
    """
    Train XGBoost models for point forecast + CI.
    XGBoost uses quantile regression via reg:quantileerror objective.
    Returns (model_q05, model_q50, model_q95).
    """
    import xgboost as xgb

    base_params = {
        'objective':       'reg:quantileerror',
        'n_estimators':     n_estimators,
        'learning_rate':    0.05,
        'max_depth':        6,
        'min_child_weight': 3,
        'subsample':        0.8,
        'colsample_bytree': 0.8,
        'reg_alpha':        0.1,
        'reg_lambda':       0.1,
        'random_state':     42,
        'verbosity':        0,
        'n_jobs':          -1,
    }

    models = {}
    for alpha, label in [(0.05, 'q05'), (0.50, 'q50'), (0.95, 'q95')]:
        params = {**base_params, 'quantile_alpha': alpha}
        m = xgb.XGBRegressor(**params)
        m.fit(X_train, y_train)
        models[label] = m

    return models['q05'], models['q50'], models['q95']


def _train_ridge_fallback(X_train: np.ndarray, y_train: np.ndarray) -> tuple:
    """Simple Ridge fallback — no quantile support, uses ±1.645σ envelope."""
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X_train)
    m  = Ridge(alpha=1.0)
    m.fit(Xs, y_train)
    resid_std = float(np.std(y_train - m.predict(Xs)))
    return scaler, m, resid_std


# ── Main public function ──────────────────────────────────────────────────────

def run_gbm_forecasts(panel: pd.DataFrame,
                      countries: list,
                      combos: list,
                      hist_years: list,
                      forecast_years: list,
                      interval_width: float = 0.90,
                      observed_pairs: list = None,
                      n_estimators: int = 500) -> pd.DataFrame:
    """
    Train a global XGBoost model on all historical
    country-combo-year data, then generate iterative forecasts for
    2024-2030 with 90% prediction intervals.

    Parameters
    ----------
    panel          : Historical AMR panel with feature columns.
    countries      : Country names.
    combos         : Combo strings.
    hist_years     : Historical years used for training.
    forecast_years : Years to forecast (2024-2030).
    interval_width : CI width (default 0.90).
    observed_pairs : Pre-filtered observed (country, combo) pairs.
    n_estimators   : Number of GBM trees (default 500).

    Returns
    -------
    DataFrame: country, combo, year, yhat, yhat_lower, yhat_upper, source
    """
    from config import SES_WEIGHTS, SES_NORM
    from features.feature_engineering import _build_panel_index

    print(f"  [GBM] Backend: {BACKEND_NAME}")
    print(f"  [GBM] Building training features from historical panel ...")

    year_min = min(hist_years)
    year_max = max(hist_years)

    # ── Build training dataset ────────────────────────────────────────────────
    train_df, feat_panel = _build_gbm_train_features(panel, year_min, year_max)

    X_train = train_df[GBM_FEATURES].fillna(0).values
    y_train = train_df['resistance_pct'].values

    print(f"  [GBM] Training set: {len(train_df):,} rows x {len(GBM_FEATURES)} features")

    # ── Zero-row guard ───────────────────────────────────────────────────────
    # If there is literally no training data (e.g. every pathogen-drug-country
    # pair has only a single historical year, so lag1_resistance can never be
    # computed), BOTH XGBoost and the Ridge fallback will crash on an empty
    # array (XGBoost: NaN base_score; sklearn: "0 samples" StandardScaler
    # error). In this case skip model fitting entirely and fall back to a
    # naive last-known-value projection per pair, with a conservative widening
    # confidence interval. This is the most honest thing to do when there is
    # genuinely no temporal signal to learn from.
    use_naive = (len(X_train) == 0)
    use_gbm   = False

    if use_naive:
        print("  [GBM] WARNING: 0 training rows available -- every pair has "
              "only a single historical year, so no lag-based features could "
              "be computed.")
        print("  [GBM] Skipping model fitting entirely. Forecasts will use a "
              "naive last-known-value projection with a widening confidence "
              "interval instead of a learned model.")
    else:
        # ── Train models ────────────────────────────────────────────────────
        print(f"  [GBM] Training 3 XGBoost quantile models "
              f"(q05 / q50 / q95, n_estimators={n_estimators}) ...")
        try:
            if BACKEND_NAME == 'xgboost':
                m_low, m_mid, m_high = _train_xgboost(
                    X_train, y_train, GBM_FEATURES, n_estimators)
            else:
                raise ImportError("Neither LightGBM nor XGBoost available")
            use_gbm = True
        except Exception as e:
            print(f"  [GBM] GBM training failed ({e}), using Ridge fallback ...")
            try:
                scaler, m_ridge, resid_std = _train_ridge_fallback(X_train, y_train)
            except Exception as e2:
                print(f"  [GBM] Ridge fallback also failed ({e2}), "
                      f"switching to naive last-known-value projection.")
                use_naive = True

    # ── Build panel index for fast history lookup ─────────────────────────────
    panel_hist = feat_panel[feat_panel['year'].isin(hist_years)].copy()
    panel_index = _build_panel_index(panel_hist)

    # ── Determine iteration pairs ─────────────────────────────────────────────
    if observed_pairs is not None:
        iter_pairs = [(str(p[0]), str(p[1])) for p in observed_pairs]
    else:
        iter_pairs = [(c, combo) for c in countries for combo in combos]

    print(f"  [GBM] Forecasting {len(iter_pairs):,} pairs "
          f"for years {forecast_years[0]}-{forecast_years[-1]} ...")

    # ── Iterative year-by-year forecast ──────────────────────────────────────
    all_years   = sorted(set(hist_years) | set(forecast_years))
    prev_preds  = {}   # {(country, combo, year): predicted_pct}
    records     = []

    for yr in forecast_years:
        feat_rows = []
        pair_keys = []

        for country, combo in iter_pairs:
            entry = panel_index.get((country, combo))
            if entry is None:
                continue

            hist_res = list(entry['resistance'])
            for prev_yr in range(min(forecast_years), yr):
                hist_res.append(
                    prev_preds.get((country, combo, prev_yr),
                                   hist_res[-1] if hist_res else 30.0)
                )

            row = _build_gbm_forecast_row(
                country, combo, yr, hist_res,
                entry['last_row'], year_min, year_max,
                SES_WEIGHTS, SES_NORM
            )
            feat_rows.append(row)
            pair_keys.append((country, combo))

        if not feat_rows:
            continue

        feat_df = pd.DataFrame(feat_rows)
        feat_df  = _ENCODERS.transform(feat_df)
        X_fore   = feat_df[GBM_FEATURES].fillna(0).values

        if use_naive:
            # Naive last-known-value projection: no model, widening CI.
            last_known = np.array([
                row_dict['lag1_resistance'] if not np.isnan(row_dict.get('lag1_resistance', np.nan))
                else 30.0
                for row_dict in feat_rows
            ])
            n_fore_steps = yr - year_max
            margin = last_known.clip(min=5) * 0.15 * (1 + n_fore_steps * 0.2)
            y_mid  = np.clip(last_known, 0, 100)
            y_low  = np.clip(y_mid - margin, 0, 100)
            y_high = np.clip(y_mid + margin, 0, 100)
        elif use_gbm:
            y_mid  = np.clip(m_mid.predict(X_fore),  0, 100)
            y_low  = np.clip(m_low.predict(X_fore),  0, 100)
            y_high = np.clip(m_high.predict(X_fore), 0, 100)
            # Ensure lower <= mid <= upper
            y_low  = np.minimum(y_low,  y_mid)
            y_high = np.maximum(y_high, y_mid)
        else:
            X_sc  = scaler.transform(X_fore)
            y_mid = np.clip(m_ridge.predict(X_sc), 0, 100)
            z     = 1.645
            n_fore = yr - year_max
            margin = z * resid_std * (1 + n_fore * 0.05)
            y_low  = np.clip(y_mid - margin, 0, 100)
            y_high = np.clip(y_mid + margin, 0, 100)

        source_label = ('naive_last_value' if use_naive else BACKEND_NAME)
        for i, (country, combo) in enumerate(pair_keys):
            pred_val = float(y_mid[i])
            prev_preds[(country, combo, yr)] = pred_val
            records.append({
                'country':    country,
                'combo':      combo,
                'year':       yr,
                'yhat':       round(pred_val, 2),
                'yhat_lower': round(float(y_low[i]),  2),
                'yhat_upper': round(float(y_high[i]), 2),
                'source':     source_label,
            })

    # Also add historical fitted values for plotting (years in hist_years)
    hist_source_label = ('naive_last_value_fitted' if use_naive
                         else f'{BACKEND_NAME}_fitted')
    hist_rows = []
    for country, combo in iter_pairs:
        entry = panel_index.get((country, combo))
        if entry is None:
            continue
        subset = panel_hist[(panel_hist.country == country) &
                             (panel_hist.combo   == combo)].sort_values('year')
        for _, row in subset.iterrows():
            hist_rows.append({
                'country':    country,
                'combo':      combo,
                'year':       int(row['year']),
                'yhat':       float(row['resistance_pct'])
                              if pd.notna(row['resistance_pct']) else np.nan,
                'yhat_lower': np.nan,
                'yhat_upper': np.nan,
                'source':     hist_source_label,
            })

    df_out = pd.DataFrame(records + hist_rows)
    src_counts = df_out['source'].value_counts().to_dict()
    print(f"  [GBM] Done. Usage: {src_counts}")

    # ── Extract and attach feature importances from median model ─────────────
    if use_naive:
        gbm_fi = pd.Series(np.zeros(len(GBM_FEATURES)), index=GBM_FEATURES,
                           name='gbm_unavailable')
        print("  [GBM] No model was fitted (naive fallback) -- feature "
              "importances are unavailable. Fig 5b will show RF importances "
              "instead.")
    else:
        gbm_fi = _extract_feature_importances(
            m_mid if use_gbm else None, GBM_FEATURES, BACKEND_NAME
        )
    # Store as DataFrame attribute so caller can retrieve without a second call
    df_out.attrs['gbm_feature_importances'] = gbm_fi

    return df_out


def _extract_feature_importances(model, feature_names: list,
                                  backend: str) -> 'pd.Series':
    """
    Extract normalised feature importances from a fitted XGBoost model.

    Why importances can be all-zero
    ---------------------------------
    XGBoost builds trees by finding splits that reduce prediction error.
    When the training target (resistance_pct) is near-constant — e.g. a
    dataset that is predominantly susceptible (0% resistance) — there is
    no error to reduce, so XGBoost builds zero trees.  All three importance
    types (gain, weight, cover) are then identically 0.  This is not a bug;
    it reflects genuinely uninformative training data for the model.

    Return convention
    -----------------
    - Normal case : pd.Series of normalised importances, sorted descending,
                    name='gbm_importance'.
    - Zero-tree case : pd.Series of zeros, name='gbm_unavailable'.
      Callers check .name == 'gbm_unavailable' to detect this and fall back
      to RF importances in Fig 5b.
    """
    if model is None or backend not in ('xgboost',):
        uniform = 1.0 / len(feature_names)
        return pd.Series([uniform] * len(feature_names),
                         index=feature_names, name='gbm_importance')
    try:
        # --- Primary: gain importance, keys are f0,f1,... in XGBoost --------
        score_dict = model.get_booster().get_score(importance_type='gain')
        raw = np.array([score_dict.get(f'f{i}', 0.0)
                        for i in range(len(feature_names))], dtype=float)

        # --- Fallback 1: weight (number of splits) ---------------------------
        if raw.sum() == 0:
            score_dict = model.get_booster().get_score(importance_type='weight')
            raw = np.array([score_dict.get(f'f{i}', 0.0)
                            for i in range(len(feature_names))], dtype=float)

        # --- Fallback 2: sklearn feature_importances_ ------------------------
        if raw.sum() == 0:
            raw = model.feature_importances_.astype(float)

        # --- Zero-tree case: no splits were made at all ----------------------
        if raw.sum() == 0:
            print(
                "  [GBM] XGBoost feature importances are all zero.\n"
                "  Cause: training target has near-zero variance "
                "(dataset may be predominantly susceptible).\n"
                "  Fig 5b will display Random Forest importances instead."
            )
            return pd.Series(np.zeros(len(feature_names)),
                             index=feature_names,
                             name='gbm_unavailable')

        raw = raw / raw.sum()
        return pd.Series(raw, index=feature_names,
                         name='gbm_importance').sort_values(ascending=False)

    except Exception as e:
        print(f"  [GBM] Could not extract feature importances: {e}")
        return pd.Series(np.zeros(len(feature_names)),
                         index=feature_names,
                         name='gbm_unavailable')


def get_gbm_feature_importances(gbm_df: 'pd.DataFrame') -> 'pd.Series':
    """
    Retrieve GBM feature importances stored in the DataFrame returned by
    run_gbm_forecasts().

    Parameters
    ----------
    gbm_df : DataFrame returned by run_gbm_forecasts()

    Returns
    -------
    pd.Series of normalised feature importances, sorted descending.
    Returns an empty Series if importances were not stored.
    """
    return gbm_df.attrs.get('gbm_feature_importances', pd.Series(dtype=float))
