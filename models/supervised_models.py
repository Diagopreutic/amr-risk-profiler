"""
models/supervised_models.py
────────────────────────────
Objective 3: Train and validate Random Forest (primary) and Ridge regression
             (interpretable baseline) using TimeSeriesSplit cross-validation.

Outputs
───────
  • Trained RF and Ridge models
  • Per-fold MAE table
  • Feature importance (RF)
  • Standardised coefficients (Ridge)
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error
import warnings

warnings.filterwarnings("ignore")


def train_random_forest(X_train: pd.DataFrame,
                        y_train: pd.Series,
                        rf_params: dict) -> RandomForestRegressor:
    """Fit Random Forest on training data."""
    model = RandomForestRegressor(**rf_params)
    model.fit(X_train, y_train)
    return model


def train_ridge(X_train: pd.DataFrame,
                y_train: pd.Series,
                alpha: float = 1.0):
    """Fit a standardised Ridge regression. Returns (scaler, model)."""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)
    model = Ridge(alpha=alpha)
    model.fit(X_scaled, y_train)
    return scaler, model


def cross_validate_rf(df_train: pd.DataFrame,
                      feature_cols: list,
                      target: str,
                      rf_params: dict,
                      n_splits: int = 5) -> dict:
    """
    Time-series cross-validation for Random Forest.

    Panel is sorted by year; folds respect temporal ordering to avoid
    look-ahead bias.

    Graceful degradation on sparse data
    -------------------------------------
    sklearn's TimeSeriesSplit requires n_samples >= n_splits + 1. Real
    surveillance files (especially single-country or narrow-year-range
    extracts) can produce far fewer than 5*2=10 training rows. Rather
    than crashing, this function:
      - n_samples >= 2: reduces n_splits down to fit the available data
        (minimum 1 split), printing a clear warning.
      - n_samples == 1: skips CV entirely, trains directly on the single
        row, reports mean_mae/std_mae as NaN.
      - n_samples == 0: skips CV and training entirely. Returns a
        trivially-fitted fallback model (fit on zeroed dummy data) so
        downstream forecasting steps receive a valid model object instead
        of crashing, with mean_mae/std_mae as NaN. A loud warning explains
        why -- this indicates the supplied file has too little usable
        historical data to forecast from.

    Returns
    -------
    dict with keys:
        fold_maes   : list of per-fold MAE values (empty if CV was skipped)
        mean_mae    : float (NaN if CV was skipped)
        std_mae     : float (NaN if CV was skipped)
        feature_importances : pd.Series (mean across folds, or single-fit/zero)
        model       : final model retrained on all data
        scaler_ridge: StandardScaler (fitted on all data for Ridge)
        model_ridge : Ridge model fitted on all data
    """
    df_sorted  = df_train.sort_values('year').reset_index(drop=True)
    X          = df_sorted[feature_cols].values
    y          = df_sorted[target].values
    n_samples  = len(df_sorted)

    # ── Zero training rows: cannot fit anything meaningful ────────────────────
    if n_samples == 0:
        print("  [RF] WARNING: 0 train-ready rows available -- cannot train "
              "any model on this data.")
        print("  [RF] This usually means the supplied file has too few years "
              "of historical data per pathogen-drug-country combination.")
        print("  [RF] Returning a placeholder model. IMPORTANT: this model "
              "was fit on dummy zero data and learned nothing -- it will "
              "predict ~0% for any input regardless of real historical "
              "values. Downstream forecasting code MUST check the "
              "'is_placeholder' flag and use a naive last-known-value "
              "fallback instead of calling .predict() on this model.")
        dummy_X = np.zeros((2, len(feature_cols)))
        dummy_y = np.zeros(2)
        rf_final = train_random_forest(
            pd.DataFrame(dummy_X, columns=feature_cols), pd.Series(dummy_y), rf_params
        )
        # Tag the model object itself so it carries the warning wherever it
        # travels, even if the caller doesn't propagate the dict flag.
        rf_final.is_placeholder_ = True
        scaler, ridge = train_ridge(
            pd.DataFrame(dummy_X, columns=feature_cols), pd.Series(dummy_y)
        )
        fi_mean = pd.Series(np.zeros(len(feature_cols)), index=feature_cols,
                            name='importance')
        return {
            'fold_maes':            [],
            'mean_mae':             float('nan'),
            'std_mae':              float('nan'),
            'feature_importances':  fi_mean,
            'model':                rf_final,
            'scaler_ridge':         scaler,
            'model_ridge':          ridge,
            'is_placeholder':       True,
        }

    # ── Single training row: fit directly, no CV possible ─────────────────────
    if n_samples == 1:
        print("  [RF] WARNING: Only 1 train-ready row available -- cannot "
              "perform cross-validation (need at least 2). Training directly "
              "on the single available row.")
        rf_final = train_random_forest(
            pd.DataFrame(X, columns=feature_cols), pd.Series(y), rf_params
        )
        scaler, ridge = train_ridge(
            pd.DataFrame(X, columns=feature_cols), pd.Series(y)
        )
        fi_mean = pd.Series(rf_final.feature_importances_, index=feature_cols,
                            name='importance').sort_values(ascending=False)
        return {
            'fold_maes':            [],
            'mean_mae':             float('nan'),
            'std_mae':              float('nan'),
            'feature_importances':  fi_mean,
            'model':                rf_final,
            'scaler_ridge':         scaler,
            'model_ridge':          ridge,
            'is_placeholder':       False,
        }

    # ── Adapt fold count to available data ─────────────────────────────────────
    # TimeSeriesSplit needs n_samples >= n_splits + 1
    effective_splits = min(n_splits, n_samples - 1)
    if effective_splits < n_splits:
        print(f"  [RF] WARNING: Only {n_samples} train-ready rows available; "
              f"reducing CV folds from {n_splits} to {effective_splits} "
              f"to fit the available data. Results will have wider "
              f"uncertainty than a full {n_splits}-fold run.")
    n_splits = effective_splits

    tscv      = TimeSeriesSplit(n_splits=n_splits)
    fold_maes = []
    fi_accum  = np.zeros(len(feature_cols))

    print(f"  [RF] TimeSeriesSplit CV ({n_splits} folds) …")
    for fold, (train_idx, val_idx) in enumerate(tscv.split(X), 1):
        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]

        rf = RandomForestRegressor(**rf_params)
        rf.fit(X_tr, y_tr)
        preds    = rf.predict(X_val)
        mae      = mean_absolute_error(y_val, preds)
        fold_maes.append(mae)
        fi_accum += rf.feature_importances_
        print(f"    Fold {fold}: MAE = {mae:.3f}%")

    fi_mean = pd.Series(fi_accum / n_splits, index=feature_cols).sort_values(ascending=False)

    # Retrain on full dataset
    print("  [RF] Retraining on full dataset …")
    rf_final = train_random_forest(
        pd.DataFrame(X, columns=feature_cols),
        pd.Series(y),
        rf_params
    )

    # Ridge on full dataset
    scaler, ridge = train_ridge(
        pd.DataFrame(X, columns=feature_cols),
        pd.Series(y)
    )

    mean_mae = float(np.mean(fold_maes))
    std_mae  = float(np.std(fold_maes))
    print(f"  [RF] CV-MAE: {mean_mae:.3f}% ± {std_mae:.3f}%")

    return {
        'fold_maes':            fold_maes,
        'mean_mae':             mean_mae,
        'std_mae':              std_mae,
        'feature_importances':  fi_mean,
        'model':                rf_final,
        'scaler_ridge':         scaler,
        'model_ridge':          ridge,
        'is_placeholder':       False,
    }


def get_ridge_coefficients(feature_cols: list,
                           scaler: StandardScaler,
                           model: Ridge) -> pd.Series:
    """
    Return Ridge coefficients (standardised) as a named Series,
    sorted by absolute magnitude.
    """
    coefs = pd.Series(model.coef_, index=feature_cols)
    # Re-scale to standardised units (coefficients already are, since we scaled X)
    return coefs.sort_values(key=lambda x: x.abs(), ascending=False)


def predict_panel(model: RandomForestRegressor,
                  df: pd.DataFrame,
                  feature_cols: list) -> pd.Series:
    """Apply a fitted RF model to a feature DataFrame."""
    X = df[feature_cols].fillna(method='ffill').fillna(0)
    return pd.Series(model.predict(X), index=df.index, name='predicted_resistance')
