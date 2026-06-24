"""
analysis/visualisations.py
───────────────────────────
Generates all publication-quality figures for the AMR challenge:

  Fig 1  — Historical resistance trends by country (line charts, per combo)
  Fig 2  — SES-RS vs % resistance scatter (Pearson r)
  Fig 3  — log GDP per capita vs % resistance (wealth-gradient)
  Fig 4  — Ridge standardised coefficient plot
  Fig 5  — Random Forest feature importance bar chart
  Fig 6  — XGBoost/XGBoost quantile forecast with 90% CI bands
  Fig 7  — 2030 Country risk tier heatmap
  Fig 8  — Antibiotic Pressure Index bubble chart
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from scipy import stats
import warnings, os

warnings.filterwarnings("ignore")

# ── Aesthetic constants ────────────────────────────────────────────────────
PALETTE = sns.color_palette("tab20", 12)
COUNTRY_COLORS = {}   # populated on first call to _get_country_color

TIER_COLORS = {
    'Low':      '#2ECC71',
    'Medium':   '#F39C12',
    'High':     '#E74C3C',
    'Critical': '#8E44AD',
}

INCOME_MARKERS = {'LMI': 'o', 'UMI': 's', 'HI': '^'}

FIG_DPI = 200
FONT_TITLE = 13
FONT_LABEL = 11


def _build_country_colors(countries: list) -> dict:
    """
    Generate one unique colour per country using the HUSL colour space,
    which is specifically designed to produce maximally distinguishable
    colours for large sets (works for 2 to 100+ countries).

    Strategy for readability:
    - Up to 20 countries  : use tab20 (high contrast, familiar palette)
    - 21-40 countries     : interleave two shifted HUSL sequences so
                             adjacent countries in the list are far apart
                             in hue.
    - 41+ countries       : full HUSL with lightness variation to add a
                             second dimension of differentiation.
    """
    n = len(countries)
    if n == 0:
        return {}

    if n <= 20:
        palette = sns.color_palette("tab20", n)
    elif n <= 40:
        # Two interleaved HUSL sequences shifted by half the hue circle
        half   = (n + 1) // 2
        seq_a  = sns.color_palette("husl", half)
        seq_b  = sns.color_palette(
            sns.diverging_palette(15, 195, s=90, l=50, as_cmap=False),
            half
        ) if hasattr(sns, 'diverging_palette') else sns.color_palette("husl", half)
        # Interleave: a0, b0, a1, b1, ...
        palette = []
        for i in range(half):
            palette.append(seq_a[i])
            if len(palette) < n:
                palette.append(seq_b[i] if i < len(seq_b) else seq_a[i])
        palette = palette[:n]
    else:
        # 41+ countries: HUSL with two lightness levels to double capacity
        half    = (n + 1) // 2
        light   = sns.color_palette("husl", half)
        # Darker variants by scaling RGB towards 0.5
        dark    = [tuple(max(0, v * 0.65) for v in c) for c in light]
        palette = []
        for i in range(half):
            palette.append(light[i])
            if len(palette) < n:
                palette.append(dark[i])
        palette = palette[:n]

    return {country: palette[i] for i, country in enumerate(countries)}


def _get_country_color(country, countries):
    """Return the unique colour assigned to `country`."""
    global COUNTRY_COLORS
    # Rebuild if the country list has changed (e.g. real-file mode vs synthetic)
    if not COUNTRY_COLORS or country not in COUNTRY_COLORS:
        COUNTRY_COLORS = _build_country_colors(list(countries))
    return COUNTRY_COLORS.get(country, (0.5, 0.5, 0.5))


def _savefig(fig, path, tight=True):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if tight:
        fig.tight_layout()
    fig.savefig(path, dpi=FIG_DPI, bbox_inches='tight')
    plt.close(fig)
    print(f"    Saved → {path}")


# ── Fig 1: Resistance Trends ──────────────────────────────────────────────
def plot_resistance_trends(panel: pd.DataFrame,
                            combos: list,
                            combo_labels: dict,
                            countries: list,
                            hist_years: list,
                            figures_dir: str,
                            max_combos: int = 12,
                            countries_per_page: int = 12):
    """
    Historical resistance % line charts, split across multiple page files.

    Layout
    ------
    Each saved file shows ALL top combos (rows) but only
    `countries_per_page` countries (unique colour per country).

    Example with 59 countries, countries_per_page=12:
      fig1a_resistance_trends.png  countries  1-12
      fig1b_resistance_trends.png  countries 13-24
      ...
      fig1e_resistance_trends.png  countries 49-59

    No two countries share a colour. Colours are generated with the
    HUSL perceptual palette (_build_country_colors) which scales to
    any number of countries.
    """
    import math
    from matplotlib.gridspec import GridSpec

    print("  [Fig 1] Resistance trends ...")

    plt.rcParams.update({
        'lines.antialiased': True,
        'patch.antialiased': True,
        'text.antialiased':  True,
    })

    hist = panel[panel.year.isin(hist_years)]

    # ── Build unique colour map for ALL countries ─────────────────────────────
    global COUNTRY_COLORS
    COUNTRY_COLORS = _build_country_colors(list(countries))

    # ── Select top combos by data coverage ───────────────────────────────────
    combo_str_to_tuple = {f"{p}_{d}": (p, d) for p, d in combos}
    coverage = (hist.dropna(subset=["resistance_pct"])
                    .groupby("combo")["resistance_pct"].count()
                    .sort_values(ascending=False))
    ranked   = [c for c in coverage.index if c in combo_str_to_tuple]
    ranked  += [c for c in combo_str_to_tuple if c not in ranked]
    top_strs   = ranked[:max_combos]
    top_tuples = [combo_str_to_tuple[c] for c in top_strs]
    n_combos   = len(top_tuples)

    if len(combo_str_to_tuple) > max_combos:
        print(f"    Showing top {max_combos} of {len(combo_str_to_tuple)} "
              f"combos by data coverage.")

    # ── Active countries: those with data in at least one top combo ───────────
    active = set()
    for p, d in top_tuples:
        active.update(
            hist[hist.combo == f"{p}_{d}"]
                .dropna(subset=["resistance_pct"])["country"].unique()
        )
    active_countries  = [c for c in countries if c in active]
    active_countries += [c for c in active if c not in active_countries]
    n_active  = len(active_countries)
    n_pages   = math.ceil(max(1, n_active) / countries_per_page)
    page_keys = "abcdefghijklmnopqrstuvwxyz"

    print(f"    {n_active} active countries -> "
          f"{n_pages} page(s) of up to {countries_per_page} countries each.")

    saved_paths = []

    for page_idx in range(n_pages):
        lo_idx = page_idx * countries_per_page
        hi_idx = min(lo_idx + countries_per_page, n_active)
        page_countries = active_countries[lo_idx:hi_idx]
        n_pc = len(page_countries)

        # Figure dimensions
        cell_h  = 4.8
        fig_w   = 9.0
        ncol_leg = min(4, n_pc)
        leg_rows = math.ceil(n_pc / ncol_leg)
        legend_h = leg_rows * 0.38 + 0.30
        fig_h    = n_combos * cell_h + legend_h + 0.70

        fig = plt.figure(figsize=(fig_w, fig_h), dpi=FIG_DPI)
        gs  = GridSpec(
            n_combos + 1, 1,
            figure=fig,
            height_ratios=[cell_h] * n_combos + [legend_h],
            hspace=0.60,
        )
        plot_axes = [fig.add_subplot(gs[i, 0]) for i in range(n_combos)]
        ax_leg    = fig.add_subplot(gs[n_combos, 0])
        ax_leg.axis("off")

        for ax, (pathogen, drug) in zip(plot_axes, top_tuples):
            combo = f"{pathogen}_{drug}"
            sub   = hist[hist.combo == combo]

            # Detect single-observation combos — only 1 unique year of data
            # across all countries. A line chart makes no sense here, so
            # switch to large annotated scatter points with resistance values
            # printed on each dot, and add a subtitle explaining the situation.
            all_years_in_combo = sub.dropna(subset=['resistance_pct'])['year'].nunique()
            single_obs = (all_years_in_combo <= 1)

            for country in page_countries:
                c_data = sub[sub.country == country].sort_values("year")
                if c_data["resistance_pct"].notna().sum() < 1:
                    continue
                color = COUNTRY_COLORS.get(country, (0.5, 0.5, 0.5))
                if single_obs:
                    # Single observation: large dot + resistance value annotated
                    for _, row in c_data.dropna(subset=['resistance_pct']).iterrows():
                        ax.scatter(row['year'], row['resistance_pct'],
                                   s=200, color=color, edgecolors='white',
                                   linewidths=1.2, zorder=5, label=country)
                        ax.annotate(f"{row['resistance_pct']:.0f}%",
                                    (row['year'], row['resistance_pct']),
                                    xytext=(8, 0), textcoords='offset points',
                                    fontsize=8, color=color, fontweight='bold',
                                    va='center')
                else:
                    ax.plot(c_data["year"], c_data["resistance_pct"],
                            marker="o", markersize=5, linewidth=2.2,
                            markeredgewidth=0.5, markeredgecolor="white",
                            solid_capstyle="round", solid_joinstyle="round",
                            color=color, label=country, zorder=3)

            title_str = combo_labels.get((pathogen, drug), combo)
            if single_obs:
                title_str += "\n(Single observation — trend analysis not possible)"
            ax.set_title(title_str, fontsize=FONT_TITLE - 1,
                         fontweight="bold", pad=5)
            ax.set_xlabel("Year", fontsize=FONT_LABEL - 1)
            ax.set_ylabel("% Non-Susceptible", fontsize=FONT_LABEL - 1)
            ax.tick_params(axis="both", labelsize=9)
            ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True, nbins=6))

            page_vals = (sub[sub.country.isin(page_countries)]
                         ["resistance_pct"].dropna())
            if len(page_vals):
                y_lo = max(0,   page_vals.min() - 5)
                y_hi = min(100, page_vals.max() + 10)
                ax.set_ylim(y_lo, y_hi)
                if y_hi > 50:
                    ax.axhline(50, color="red", linestyle="--",
                               linewidth=1.0, alpha=0.40, zorder=1)
            else:
                ax.set_ylim(0, 100)
                ax.text(0.5, 0.5, "No data for this country group",
                        transform=ax.transAxes, ha="center", va="center",
                        fontsize=9, color="grey", style="italic")

            ax.grid(axis="y", alpha=0.22, linewidth=0.7, zorder=0)
            ax.set_facecolor("#FAFAFA")

        # Legend: one patch per country, all unique colours
        handles = [
            mpatches.Patch(
                color=COUNTRY_COLORS.get(c, (0.5, 0.5, 0.5)), label=c
            )
            for c in page_countries
        ]
        ax_leg.legend(
            handles=handles, loc="center", ncol=ncol_leg,
            fontsize=9, frameon=True, framealpha=0.92,
            edgecolor="#CCCCCC",
            title=f"Countries — page {page_idx + 1}/{n_pages}  "
                  f"(#{lo_idx + 1}-{hi_idx})",
            title_fontsize=9,
            bbox_to_anchor=(0.5, 0.5),
            bbox_transform=ax_leg.transAxes,
        )

        pg_lbl = page_keys[page_idx] if page_idx < len(page_keys) else str(page_idx+1)
        fig.suptitle(
            f"AMR Historical Resistance Trends 2010-2023"
            f"  —  Page {page_idx+1}/{n_pages}  "
            f"(countries {lo_idx+1}-{hi_idx} of {n_active})",
            fontsize=FONT_TITLE, fontweight="bold", y=1.005,
        )

        fname = (
            f"{figures_dir}/fig1{pg_lbl}_resistance_trends.png"
            if n_pages > 1
            else f"{figures_dir}/fig1_resistance_trends.png"
        )
        fig.savefig(fname, dpi=FIG_DPI, bbox_inches="tight")
        plt.close(fig)
        print(f"    Saved -> {fname}")
        saved_paths.append(fname)

    return saved_paths


def plot_ses_vs_resistance(panel: pd.DataFrame,
                            hist_years: list,
                            income_groups: dict,
                            figures_dir: str):
    """Scatter: SES Risk Score vs mean resistance %; annotated with Pearson r."""
    print("  [Fig 2] SES-RS vs resistance …")
    hist = panel[panel.year.isin(hist_years)]
    agg  = (hist.groupby('country')
                .agg(mean_res=('resistance_pct', 'mean'),
                     mean_ses=('ses_risk_score', 'mean'))
                .reset_index())
    agg['income'] = agg['country'].map(income_groups)

    fig, ax = plt.subplots(figsize=(9, 6))

    if len(agg) < 2:
        # Not enough countries for a correlation -- draw what we have
        # (a single point, or nothing) with a clear explanatory note
        # instead of crashing on pearsonr/polyfit.
        print(f"    WARNING: only {len(agg)} country available -- "
              f"skipping Pearson correlation (needs >= 2).")
        r, p = float('nan'), float('nan')
        for ig, marker in INCOME_MARKERS.items():
            sub = agg[agg.income == ig]
            if len(sub):
                ax.scatter(sub['mean_ses'], sub['mean_res'],
                           marker=marker, s=90, label=ig,
                           color=[_get_country_color(c, list(income_groups.keys()))
                                  for c in sub.country], edgecolors='k', linewidths=0.4)
        for _, row in agg.iterrows():
            ax.annotate(row['country'], (row['mean_ses'], row['mean_res']),
                        fontsize=7.5, xytext=(4, 2), textcoords='offset points')
        ax.set_xlabel('Composite SES Risk Score (0-1)', fontsize=FONT_LABEL)
        ax.set_ylabel('Mean % Non-Susceptible', fontsize=FONT_LABEL)
        ax.set_title(f'Socioeconomic Risk vs AMR Resistance\n'
                     f'Insufficient countries for correlation (n={len(agg)})',
                     fontsize=FONT_TITLE, fontweight='bold', color='#B03A2E')
        ax.text(0.5, 0.5,
                f'Only {len(agg)} country in dataset --\n'
                f'Pearson correlation requires at least 2.',
                transform=ax.transAxes, ha='center', va='center',
                fontsize=10, color='#B03A2E', style='italic',
                bbox=dict(boxstyle='round,pad=0.4', facecolor='#FDEDEC',
                          edgecolor='#B03A2E', alpha=0.85))
        if len(agg):
            ax.legend(title='Income Group', fontsize=9)
        ax.grid(alpha=0.3)
        _savefig(fig, f"{figures_dir}/fig2_ses_vs_resistance.png")
        return r, p

    r, p = stats.pearsonr(agg['mean_ses'], agg['mean_res'])

    for ig, marker in INCOME_MARKERS.items():
        sub = agg[agg.income == ig]
        ax.scatter(sub['mean_ses'], sub['mean_res'],
                   marker=marker, s=90, label=ig,
                   color=[_get_country_color(c, list(income_groups.keys()))
                          for c in sub.country], edgecolors='k', linewidths=0.4)

    for _, row in agg.iterrows():
        ax.annotate(row['country'], (row['mean_ses'], row['mean_res']),
                    fontsize=7.5, xytext=(4, 2), textcoords='offset points')

    # Regression line
    m_fit, b_fit = np.polyfit(agg['mean_ses'], agg['mean_res'], 1)
    xs = np.linspace(agg['mean_ses'].min(), agg['mean_ses'].max(), 100)
    ax.plot(xs, m_fit * xs + b_fit, 'r--', linewidth=1.5, alpha=0.8)

    ax.set_xlabel('Composite SES Risk Score (0–1)', fontsize=FONT_LABEL)
    ax.set_ylabel('Mean % Non-Susceptible (2010–2023)', fontsize=FONT_LABEL)
    ax.set_title(f'Socioeconomic Risk vs AMR Resistance\nPearson r = {r:.3f}  (p = {p:.4f})',
                 fontsize=FONT_TITLE, fontweight='bold')
    ax.legend(title='Income Group', fontsize=9)
    ax.grid(alpha=0.3)
    _savefig(fig, f"{figures_dir}/fig2_ses_vs_resistance.png")
    return r, p


# ── Fig 3: GDP Gradient ───────────────────────────────────────────────────
def plot_gdp_gradient(panel: pd.DataFrame,
                       hist_years: list,
                       income_groups: dict,
                       figures_dir: str):
    """Scatter: log GDP per capita vs mean % resistance."""
    print("  [Fig 3] GDP gradient …")
    hist = panel[panel.year.isin(hist_years)]
    agg  = (hist.groupby('country')
                .agg(mean_res=('resistance_pct', 'mean'),
                     mean_gdp=('gdp_per_capita', 'mean'))
                .reset_index())
    agg['income'] = agg['country'].map(income_groups)
    agg['log_gdp'] = np.log10(agg['mean_gdp'])

    fig, ax = plt.subplots(figsize=(9, 6))

    if len(agg) < 2:
        print(f"    WARNING: only {len(agg)} country available -- "
              f"skipping Pearson correlation (needs >= 2).")
        r, p = float('nan'), float('nan')
        for ig, marker in INCOME_MARKERS.items():
            sub = agg[agg.income == ig]
            if len(sub):
                ax.scatter(sub['log_gdp'], sub['mean_res'],
                           marker=marker, s=90, label=ig,
                           color=[_get_country_color(c, list(income_groups.keys()))
                                  for c in sub.country], edgecolors='k', linewidths=0.4)
        for _, row in agg.iterrows():
            ax.annotate(row['country'], (row['log_gdp'], row['mean_res']),
                        fontsize=7.5, xytext=(4, 2), textcoords='offset points')
        ax.set_xlabel('log\u2081\u2080 GDP per Capita (USD)', fontsize=FONT_LABEL)
        ax.set_ylabel('Mean % Non-Susceptible', fontsize=FONT_LABEL)
        ax.set_title(f'Wealth-Resistance Gradient\n'
                     f'Insufficient countries for correlation (n={len(agg)})',
                     fontsize=FONT_TITLE, fontweight='bold', color='#B03A2E')
        ax.text(0.5, 0.5,
                f'Only {len(agg)} country in dataset --\n'
                f'Pearson correlation requires at least 2.',
                transform=ax.transAxes, ha='center', va='center',
                fontsize=10, color='#B03A2E', style='italic',
                bbox=dict(boxstyle='round,pad=0.4', facecolor='#FDEDEC',
                          edgecolor='#B03A2E', alpha=0.85))
        if len(agg):
            ax.legend(title='Income Group', fontsize=9)
        ax.grid(alpha=0.3)
        _savefig(fig, f"{figures_dir}/fig3_gdp_gradient.png")
        return r, p

    r, p = stats.pearsonr(agg['log_gdp'], agg['mean_res'])

    for ig, marker in INCOME_MARKERS.items():
        sub = agg[agg.income == ig]
        ax.scatter(sub['log_gdp'], sub['mean_res'],
                   marker=marker, s=90, label=ig,
                   color=[_get_country_color(c, list(income_groups.keys()))
                          for c in sub.country], edgecolors='k', linewidths=0.4)

    for _, row in agg.iterrows():
        ax.annotate(row['country'], (row['log_gdp'], row['mean_res']),
                    fontsize=7.5, xytext=(4, 2), textcoords='offset points')

    m_fit, b_fit = np.polyfit(agg['log_gdp'], agg['mean_res'], 1)
    xs = np.linspace(agg['log_gdp'].min(), agg['log_gdp'].max(), 100)
    ax.plot(xs, m_fit * xs + b_fit, 'r--', linewidth=1.5, alpha=0.8)

    ax.set_xlabel('log₁₀ GDP per Capita (USD)', fontsize=FONT_LABEL)
    ax.set_ylabel('Mean % Non-Susceptible (2010–2023)', fontsize=FONT_LABEL)
    ax.set_title(f'Wealth–Resistance Gradient\nPearson r = {r:.3f}  (p = {p:.4f})',
                 fontsize=FONT_TITLE, fontweight='bold')
    ax.legend(title='Income Group', fontsize=9)
    ax.grid(alpha=0.3)
    _savefig(fig, f"{figures_dir}/fig3_gdp_gradient.png")
    return r, p


# ── Fig 4: Ridge Coefficients ─────────────────────────────────────────────
def plot_ridge_coefficients(coef_series: pd.Series, figures_dir: str,
                             panel: pd.DataFrame = None):
    """Horizontal bar chart of standardised Ridge coefficients.

    When the model is a placeholder (all coefficients near zero because
    the Ridge was trained on dummy data), the bar chart would be visually
    empty. In that case, this function substitutes a meaningful alternative:
    the actual mean observed resistance % per country-combo from the real
    panel data, giving the user informative output rather than a blank plot.
    """
    print("  [Fig 4] Ridge coefficients ...")
    is_zero = coef_series.abs().sum() < 1e-6

    fig, ax = plt.subplots(figsize=(10, 7))

    if is_zero and panel is not None:
        # Substitute: show actual mean resistance per country from real data
        obs = panel.dropna(subset=['resistance_pct'])
        if len(obs):
            mean_r = (obs.groupby('country')['resistance_pct']
                        .mean()
                        .sort_values(ascending=True))
            colors_bar = ['#E74C3C' if v > 65 else '#F39C12' if v > 45
                          else '#F7DC6F' if v > 25 else '#2ECC71'
                          for v in mean_r.values]
            bars = ax.barh(mean_r.index, mean_r.values,
                           color=colors_bar, edgecolor='k', linewidth=0.4,
                           height=0.6)
            for bar, val in zip(bars, mean_r.values):
                ax.text(val + 0.8, bar.get_y() + bar.get_height() / 2,
                        f'{val:.1f}%', va='center', fontsize=9,
                        fontweight='bold')
            ax.set_xlabel('Mean Observed Resistance % (2010-2023)',
                          fontsize=FONT_LABEL)
            ax.set_title(
                'Observed Mean Resistance by Country\n'
                '(Ridge coefficient chart unavailable: insufficient '
                'multi-year data to train the model)',
                fontsize=FONT_TITLE, fontweight='bold', color='#7D3C98')
            ax.set_xlim(0, min(105, mean_r.max() * 1.15))
            ax.axvline(65, color='#E74C3C', linestyle='--',
                       linewidth=1, alpha=0.6, label='Critical threshold 65%')
            ax.axvline(25, color='#2ECC71', linestyle='--',
                       linewidth=1, alpha=0.6, label='Low threshold 25%')
            ax.legend(fontsize=8)
            ax.text(0.98, 0.02,
                    'Ridge requires 2+ years per combination to estimate '
                    'protective vs risk-driving factors.\n'
                    'Supply multi-year surveillance data to enable this analysis.',
                    transform=ax.transAxes, ha='right', va='bottom',
                    fontsize=8, color='#7D3C98', style='italic',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='#F9EBFF',
                              edgecolor='#7D3C98', alpha=0.85))
        else:
            ax.text(0.5, 0.5, 'No observed resistance data available.',
                    transform=ax.transAxes, ha='center', va='center',
                    fontsize=11, color='grey')
    else:
        coef = coef_series.sort_values()
        colors = ['#E74C3C' if v > 0 else '#2ECC71' for v in coef.values]
        bars = ax.barh(coef.index, coef.values, color=colors, edgecolor='k',
                       linewidth=0.4, height=0.7)
        ax.axvline(0, color='black', linewidth=1.0)
        ax.set_xlabel('Standardised Coefficient (Ridge α=1.0)',
                      fontsize=FONT_LABEL)
        ax.set_title('Ridge Regression: Protective vs Resistance-Driving Factors',
                     fontsize=FONT_TITLE, fontweight='bold')
        pos_patch = mpatches.Patch(color='#E74C3C', label='Resistance-driving (+)')
        neg_patch = mpatches.Patch(color='#2ECC71', label='Protective (−)')
        ax.legend(handles=[pos_patch, neg_patch], fontsize=9)

    ax.grid(axis='x', alpha=0.3)
    _savefig(fig, f"{figures_dir}/fig4_ridge_coefficients.png")


# ── Fig 5: RF Feature Importance ─────────────────────────────────────────
def plot_feature_importance(fi_series: pd.Series, figures_dir: str,
                             panel: pd.DataFrame = None):
    """Bar chart of RF feature importances (top 15).

    When the model is a placeholder (all importances = 0 because the RF
    was trained on dummy data), substitutes a meaningful alternative:
    the actual mean observed resistance % per pathogen-drug combination,
    giving informative output instead of a blank bar chart.
    """
    print("  [Fig 5] RF feature importance ...")
    is_zero = fi_series.abs().sum() < 1e-6

    fig, ax = plt.subplots(figsize=(10, 7))

    if is_zero and panel is not None:
        obs = panel.dropna(subset=['resistance_pct'])
        if len(obs):
            mean_r = (obs.groupby('combo')['resistance_pct']
                        .mean()
                        .sort_values(ascending=True)
                        .tail(15))
            colors_bar = ['#E74C3C' if v > 65 else '#F39C12' if v > 45
                          else '#F7DC6F' if v > 25 else '#2ECC71'
                          for v in mean_r.values]
            bars = ax.barh(mean_r.index, mean_r.values,
                           color=colors_bar, edgecolor='k',
                           linewidth=0.4, height=0.65)
            for bar, val in zip(bars, mean_r.values):
                ax.text(val + 0.8, bar.get_y() + bar.get_height() / 2,
                        f'{val:.1f}%', va='center', fontsize=8,
                        fontweight='bold')
            ax.set_xlabel('Mean Observed Resistance % (2010-2023)',
                          fontsize=FONT_LABEL)
            ax.set_title(
                'Observed Resistance by Pathogen-Drug Combination\n'
                '(RF feature importance unavailable: insufficient '
                'multi-year data to train the model)',
                fontsize=FONT_TITLE, fontweight='bold', color='#7D3C98')
            ax.set_xlim(0, min(105, mean_r.max() * 1.15))
            ax.text(0.98, 0.02,
                    'RF feature importance requires 2+ years per combination.\n'
                    'Supply multi-year surveillance data to enable this analysis.',
                    transform=ax.transAxes, ha='right', va='bottom',
                    fontsize=8, color='#7D3C98', style='italic',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='#F9EBFF',
                              edgecolor='#7D3C98', alpha=0.85))
        else:
            ax.text(0.5, 0.5, 'No observed resistance data available.',
                    transform=ax.transAxes, ha='center', va='center',
                    fontsize=11, color='grey')
    else:
        top = fi_series.head(15).sort_values()
        colors = []
        for feat in top.index:
            if any(t in feat for t in ['lag', 'rolling', 'delta']):
                colors.append('#3498DB')
            elif any(t in feat for t in ['DDD', 'watch', 'pressure']):
                colors.append('#E67E22')
            else:
                colors.append('#9B59B6')
        ax.barh(top.index, top.values, color=colors, edgecolor='k',
                linewidth=0.4, height=0.7)
        ax.set_xlabel('Mean Decrease in Impurity (Feature Importance)',
                      fontsize=FONT_LABEL)
        ax.set_title('Random Forest Feature Importances\n'
                     '(Mean across 5 CV folds)',
                     fontsize=FONT_TITLE, fontweight='bold')
        patches = [
            mpatches.Patch(color='#3498DB', label='Temporal autocorrelation'),
            mpatches.Patch(color='#E67E22', label='Antibiotic consumption'),
            mpatches.Patch(color='#9B59B6', label='Socioeconomic'),
        ]
        ax.legend(handles=patches, fontsize=9)

    ax.grid(axis='x', alpha=0.3)
    _savefig(fig, f"{figures_dir}/fig5_feature_importance.png")


# ── Fig 5b: XGBoost Feature Importance ──────────────────────────────────────
# -- Fig 5b: XGBoost Feature Importance ------------------------------------
def plot_gbm_feature_importance(gbm_fi, rf_fi, figures_dir,
                                backend_name='XGBoost',
                                panel: pd.DataFrame = None):
    """
    Fig 5b: Feature importance with cascading fallbacks.

    Normal case (multi-year data):
      Left  — XGBoost gain importances
      Right — RF vs XGBoost side-by-side with modifiable drivers highlighted

    XGBoost zero-variance (e.g. single-drug dataset):
      Left  — RF importances (with annotation)
      Right — RF modifiable driver ranking

    Both models placeholder (0 train-ready rows, e.g. single-year file):
      Both panels — real observed resistance % per combo from panel data,
      colour-coded by risk tier, with a clear explanation. This is the
      most informative output possible when no temporal model can be trained.
    """
    print('  [Fig 5b] XGBoost feature importances + RF comparison ...')

    def _colour(feat):
        if any(t in feat for t in ['lag', 'rolling', 'delta', 'year']):
            return '#2980B9'
        if any(t in feat for t in ['DDD', 'watch', 'pressure']):
            return '#E67E22'
        if feat in ['pathogen_enc', 'drug_enc', 'country_enc']:
            return '#27AE60'
        return '#8E44AD'

    xgb_na  = (gbm_fi.name == 'gbm_unavailable' or gbm_fi.sum() == 0)
    rf_na   = (rf_fi.abs().sum() < 1e-6)
    both_na = xgb_na and rf_na

    fig, axes = plt.subplots(1, 2, figsize=(18, 8))

    # ── Both models are placeholders: show observed resistance data ───────────
    if both_na:
        note_txt = (
            'Both XGBoost and Random Forest are placeholders' + chr(10) +
            '(0 train-ready rows: dataset has only one year of data,' + chr(10) +
            'so no lag features could be computed).' + chr(10) + chr(10) +
            'Feature importance analysis requires' + chr(10) +
            'at least 2 years of historical data.' + chr(10) + chr(10) +
            'Showing observed resistance % from real data instead.'
        )
        for ax_idx, ax in enumerate(axes):
            ax.set_facecolor('#FAFAFA')
            if panel is not None:
                obs = panel.dropna(subset=['resistance_pct'])
                if len(obs):
                    if ax_idx == 0:
                        # Left: mean resistance per combo
                        mean_r = (obs.groupby('combo')['resistance_pct']
                                    .mean().sort_values(ascending=True))
                        x_vals, y_lbls = mean_r.values, list(mean_r.index)
                        xlabel = 'Mean Observed Resistance %'
                        title  = ('Observed Resistance by Pathogen-Drug' +
                                  chr(10) + 'Combination')
                    else:
                        # Right: mean resistance per country
                        mean_r = (obs.groupby('country')['resistance_pct']
                                    .mean().sort_values(ascending=True))
                        x_vals, y_lbls = mean_r.values, list(mean_r.index)
                        xlabel = 'Mean Observed Resistance %'
                        title  = ('Observed Resistance by Country' +
                                  chr(10) + '(All Pathogen-Drug Combinations)')

                    tier_colors = ['#8E44AD' if v > 65 else '#E74C3C' if v > 45
                                   else '#F39C12' if v > 25 else '#2ECC71'
                                   for v in x_vals]
                    bars = ax.barh(range(len(y_lbls)), x_vals,
                                   color=tier_colors, edgecolor='k',
                                   linewidth=0.4, height=0.65)
                    ax.set_yticks(range(len(y_lbls)))
                    ax.set_yticklabels(y_lbls, fontsize=9)
                    for bar, val in zip(bars, x_vals):
                        ax.text(val + 0.8,
                                bar.get_y() + bar.get_height() / 2,
                                '{:.1f}%'.format(val),
                                va='center', fontsize=8, fontweight='bold')
                    ax.set_xlim(0, min(105, max(x_vals) * 1.15) if len(x_vals) else 10)
                    ax.set_xlabel(xlabel, fontsize=FONT_LABEL)
                    ax.set_title(title, fontsize=FONT_TITLE,
                                 fontweight='bold', color='#7D3C98')
                    ax.grid(axis='x', alpha=0.25, linewidth=0.7)
                    # Tier legend
                    tier_patches = [
                        mpatches.Patch(color='#8E44AD', label='Critical (>65%)'),
                        mpatches.Patch(color='#E74C3C', label='High (45-65%)'),
                        mpatches.Patch(color='#F39C12', label='Medium (25-45%)'),
                        mpatches.Patch(color='#2ECC71', label='Low (<25%)'),
                    ]
                    ax.legend(handles=tier_patches, fontsize=8, loc='lower right')

            # Explanatory annotation on right panel
            if ax_idx == 1:
                ax.text(0.5, 0.02, note_txt,
                        transform=ax.transAxes, ha='center', va='bottom',
                        fontsize=8, color='#7D3C98', style='italic',
                        bbox=dict(boxstyle='round,pad=0.5', facecolor='#F9EBFF',
                                  edgecolor='#7D3C98', alpha=0.88))

        fig.suptitle(
            'Feature Importance - Modifiable Drivers of AMR' + chr(10) +
            'Observed resistance shown (models unavailable: single-year data)',
            fontsize=FONT_TITLE + 1, fontweight='bold', color='#7D3C98', y=1.02
        )
        _savefig(fig, '{}/fig5b_gbm_feature_importance.png'.format(figures_dir))
        return

    # ── Normal paths (at least RF importances available) ─────────────────────
    ax = axes[0]
    top_left = (rf_fi if xgb_na else gbm_fi).head(15).sort_values()
    clr = [_colour(f) for f in top_left.index]
    bars = ax.barh(top_left.index, top_left.values,
                   color=clr, edgecolor='k', linewidth=0.4, height=0.7)
    for bar, val in zip(bars, top_left.values):
        ax.text(val + 0.002, bar.get_y() + bar.get_height() / 2,
                '{:.1f}%'.format(val * 100),
                va='center', fontsize=8, color='#333333')
    if xgb_na:
        ax.set_xlabel('Normalised RF Importance', fontsize=FONT_LABEL)
        ttl = ('Random Forest Feature Importances' + chr(10) +
               '(XGBoost unavailable: near-constant target)')
        ax.set_title(ttl, fontsize=FONT_TITLE, fontweight='bold',
                     color='#7D3C98')
        note = chr(10).join([
            'XGBoost built zero trees.',
            'Training target has near-zero variance',
            '(predominantly susceptible data).',
            '', 'RF importances shown instead.'])
        ax.text(0.97, 0.03, note, transform=ax.transAxes,
                ha='right', va='bottom', fontsize=9,
                color='#7D3C98', style='italic',
                bbox=dict(boxstyle='round,pad=0.4', facecolor='#F9EBFF',
                          edgecolor='#7D3C98', alpha=0.85))
    else:
        ax.set_xlabel('Normalised Gain Importance', fontsize=FONT_LABEL)
        ttl = ('XGBoost Feature Importances' + chr(10) +
               '(Global model - all countries & combos)')
        ax.set_title(ttl, fontsize=FONT_TITLE, fontweight='bold')
    ax.set_xlim(0, max(top_left.values.max() * 1.18, 0.01))
    ax.grid(axis='x', alpha=0.25, linewidth=0.7)
    ax.set_facecolor('#FAFAFA')
    patches = [
        mpatches.Patch(color='#2980B9', label='Temporal / year'),
        mpatches.Patch(color='#E67E22', label='Antibiotic consumption'),
        mpatches.Patch(color='#27AE60', label='Pathogen / drug / country id'),
        mpatches.Patch(color='#8E44AD', label='Socioeconomic'),
    ]
    ax.legend(handles=patches, fontsize=8, loc='lower right')

    ax2 = axes[1]
    if xgb_na:
        top_rf = rf_fi.head(15).sort_values()
        clr2 = [_colour(f) for f in top_rf.index]
        ax2.barh(range(len(top_rf)), top_rf.values,
                 color=clr2, edgecolor='k', linewidth=0.4, height=0.7)
        ax2.set_yticks(range(len(top_rf)))
        ax2.set_yticklabels(top_rf.index, fontsize=9)
        ax2.set_xlabel('Normalised RF Importance', fontsize=FONT_LABEL)
        ttl2 = ('Random Forest - Modifiable Driver Ranking' + chr(10) +
                '(XGBoost comparison unavailable)')
        ax2.set_title(ttl2, fontsize=FONT_TITLE, fontweight='bold')
        mod_idx = [i for i, f in enumerate(top_rf.index)
                   if not any(t in f for t in
                              ['lag', 'rolling', 'delta', 'year',
                               'pathogen_enc', 'drug_enc', 'country_enc'])]
    else:
        shared = list(dict.fromkeys(
            list(gbm_fi.head(12).index) + list(rf_fi.head(12).index)))
        shared = [f for f in shared
                  if f in gbm_fi.index and f in rf_fi.index][:15]
        srev = list(reversed(shared))
        gv = [float(gbm_fi.get(f, 0)) for f in srev]
        rv = [float(rf_fi.get(f, 0)) for f in srev]
        yp, h = np.arange(len(srev)), 0.35
        ax2.barh(yp + h/2, gv, h, color='#2980B9', alpha=0.85,
                 edgecolor='k', linewidth=0.4, label='XGBoost')
        ax2.barh(yp - h/2, rv, h, color='#E74C3C', alpha=0.85,
                 edgecolor='k', linewidth=0.4, label='Random Forest')
        ax2.set_yticks(yp)
        ax2.set_yticklabels(srev, fontsize=9)
        ax2.set_xlabel('Normalised Importance', fontsize=FONT_LABEL)
        ttl2 = ('RF vs XGBoost - Shared Feature Comparison' + chr(10) +
                '(Modifiable drivers highlighted)')
        ax2.set_title(ttl2, fontsize=FONT_TITLE, fontweight='bold')
        ax2.legend(fontsize=9)
        mod_idx = [i for i, f in enumerate(srev)
                   if not any(t in f for t in
                              ['lag', 'rolling', 'delta', 'year',
                               'pathogen_enc', 'drug_enc', 'country_enc'])]
    for i in mod_idx:
        ax2.axhspan(i - 0.5, i + 0.5,
                    color='#FFF3CD', alpha=0.45, zorder=0)
    ax2.text(0.98, 0.01, 'Yellow = modifiable policy drivers',
             transform=ax2.transAxes, fontsize=8,
             ha='right', va='bottom', color='#7D6608',
             bbox=dict(boxstyle='round,pad=0.3', facecolor='#FFF3CD',
                       edgecolor='#D4AC0D', alpha=0.8))
    ax2.grid(axis='x', alpha=0.25, linewidth=0.7)
    ax2.set_facecolor('#FAFAFA')
    lbl = ('RF (XGBoost zero-variance)' if xgb_na else 'XGBoost global model')
    sup = ('Feature Importance - Modifiable Drivers of AMR' + chr(10) +
           'Left: ' + lbl + '  |  Right: modifiable driver ranking')
    fig.suptitle(sup, fontsize=FONT_TITLE + 1, fontweight='bold', y=1.02)
    _savefig(fig, '{}/fig5b_gbm_feature_importance.png'.format(figures_dir))



    def _colour(feat):
        if any(t in feat for t in ['lag', 'rolling', 'delta', 'year']):
            return '#2980B9'
        if any(t in feat for t in ['DDD', 'watch', 'pressure']):
            return '#E67E22'
        if feat in ['pathogen_enc', 'drug_enc', 'country_enc']:
            return '#27AE60'
        return '#8E44AD'

    xgb_na = (gbm_fi.name == 'gbm_unavailable' or gbm_fi.sum() == 0)
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))

    # Left panel
    ax = axes[0]
    top_left = (rf_fi if xgb_na else gbm_fi).head(15).sort_values()
    clr = [_colour(f) for f in top_left.index]
    bars = ax.barh(top_left.index, top_left.values,
                   color=clr, edgecolor='k', linewidth=0.4, height=0.7)
    for bar, val in zip(bars, top_left.values):
        ax.text(val + 0.002, bar.get_y() + bar.get_height() / 2,
                '{:.1f}%'.format(val * 100),
                va='center', fontsize=8, color='#333333')
    if xgb_na:
        ax.set_xlabel('Normalised RF Importance', fontsize=FONT_LABEL)
        ttl = ('Random Forest Feature Importances' + chr(10) +
               '(XGBoost unavailable: near-constant target)')
        ax.set_title(ttl, fontsize=FONT_TITLE, fontweight='bold',
                     color='#7D3C98')
        note = (chr(10).join([
            'XGBoost built zero trees.',
            'Training target has near-zero variance',
            '(predominantly susceptible data).',
            '',
            'RF importances shown instead.']))
        ax.text(0.97, 0.03, note, transform=ax.transAxes,
                ha='right', va='bottom', fontsize=9,
                color='#7D3C98', style='italic',
                bbox=dict(boxstyle='round,pad=0.4', facecolor='#F9EBFF',
                          edgecolor='#7D3C98', alpha=0.85))
    else:
        ax.set_xlabel('Normalised Gain Importance', fontsize=FONT_LABEL)
        ttl = ('XGBoost Feature Importances' + chr(10) +
               '(Global model - all countries & combos)')
        ax.set_title(ttl, fontsize=FONT_TITLE, fontweight='bold')
    ax.set_xlim(0, max(top_left.values.max() * 1.18, 0.01))
    ax.grid(axis='x', alpha=0.25, linewidth=0.7)
    ax.set_facecolor('#FAFAFA')
    patches = [
        mpatches.Patch(color='#2980B9', label='Temporal / year'),
        mpatches.Patch(color='#E67E22', label='Antibiotic consumption'),
        mpatches.Patch(color='#27AE60', label='Pathogen / drug / country id'),
        mpatches.Patch(color='#8E44AD', label='Socioeconomic'),
    ]
    ax.legend(handles=patches, fontsize=8, loc='lower right')

    # Right panel
    ax2 = axes[1]
    if xgb_na:
        top_rf = rf_fi.head(15).sort_values()
        clr2 = [_colour(f) for f in top_rf.index]
        ax2.barh(range(len(top_rf)), top_rf.values,
                 color=clr2, edgecolor='k', linewidth=0.4, height=0.7)
        ax2.set_yticks(range(len(top_rf)))
        ax2.set_yticklabels(top_rf.index, fontsize=9)
        ax2.set_xlabel('Normalised RF Importance', fontsize=FONT_LABEL)
        ttl2 = ('Random Forest - Modifiable Driver Ranking' + chr(10) +
                '(XGBoost comparison unavailable)')
        ax2.set_title(ttl2, fontsize=FONT_TITLE, fontweight='bold')
        mod_idx = [i for i, f in enumerate(top_rf.index)
                   if not any(t in f for t in
                              ['lag', 'rolling', 'delta', 'year',
                               'pathogen_enc', 'drug_enc', 'country_enc'])]
    else:
        shared = list(dict.fromkeys(
            list(gbm_fi.head(12).index) + list(rf_fi.head(12).index)))
        shared = [f for f in shared
                  if f in gbm_fi.index and f in rf_fi.index][:15]
        srev = list(reversed(shared))
        gv = [float(gbm_fi.get(f, 0)) for f in srev]
        rv = [float(rf_fi.get(f, 0)) for f in srev]
        yp, h = np.arange(len(srev)), 0.35
        ax2.barh(yp + h/2, gv, h, color='#2980B9', alpha=0.85,
                 edgecolor='k', linewidth=0.4, label='XGBoost')
        ax2.barh(yp - h/2, rv, h, color='#E74C3C', alpha=0.85,
                 edgecolor='k', linewidth=0.4, label='Random Forest')
        ax2.set_yticks(yp)
        ax2.set_yticklabels(srev, fontsize=9)
        ax2.set_xlabel('Normalised Importance', fontsize=FONT_LABEL)
        ttl2 = ('RF vs XGBoost - Shared Feature Comparison' + chr(10) +
                '(Modifiable drivers highlighted)')
        ax2.set_title(ttl2, fontsize=FONT_TITLE, fontweight='bold')
        ax2.legend(fontsize=9)
        mod_idx = [i for i, f in enumerate(srev)
                   if not any(t in f for t in
                              ['lag', 'rolling', 'delta', 'year',
                               'pathogen_enc', 'drug_enc', 'country_enc'])]
    for i in mod_idx:
        ax2.axhspan(i - 0.5, i + 0.5,
                    color='#FFF3CD', alpha=0.45, zorder=0)
    ax2.text(0.98, 0.01, 'Yellow = modifiable policy drivers',
             transform=ax2.transAxes, fontsize=8,
             ha='right', va='bottom', color='#7D6608',
             bbox=dict(boxstyle='round,pad=0.3', facecolor='#FFF3CD',
                       edgecolor='#D4AC0D', alpha=0.8))
    ax2.grid(axis='x', alpha=0.25, linewidth=0.7)
    ax2.set_facecolor('#FAFAFA')
    lbl = ('RF (XGBoost zero-variance)' if xgb_na else 'XGBoost global model')
    sup = ('Feature Importance - Modifiable Drivers of AMR' + chr(10) +
           'Left: ' + lbl + '  |  Right: modifiable driver ranking')
    fig.suptitle(sup, fontsize=FONT_TITLE + 1, fontweight='bold', y=1.02)
    _savefig(fig, '{}/fig5b_gbm_feature_importance.png'.format(figures_dir))


def plot_gbm_forecasts(panel: pd.DataFrame,
                        gbm_df: pd.DataFrame,
                        rf_forecasts: pd.DataFrame,
                        countries_to_plot: list,
                        combo_to_plot: str,
                        combo_label: str,
                        hist_years: list,
                        forecast_years: list,
                        figures_dir: str):
    """XGBoost 90% CI quantile forecasts + RF overlay."""
    print("  [Fig 6] XGBoost quantile forecast bands ...")

    # ── Auto-select the best combo if the requested one has no forecast data ──
    fore_df = gbm_df[gbm_df.year.isin(forecast_years)]
    combos_with_forecasts = fore_df.dropna(subset=["yhat"])["combo"].unique()

    if combo_to_plot not in combos_with_forecasts:
        # Pick combo with highest mean forecast resistance AND >= 3 countries
        # to avoid selecting a rare combo with only 1 high-resistance country.
        def _best_combo(min_countries):
            return (fore_df.dropna(subset=["yhat"])
                    .groupby("combo")
                    .agg(n_countries=("country", "nunique"),
                         mean_r=("yhat", "mean"))
                    .query(f"n_countries >= {min_countries}")
                    .sort_values("mean_r", ascending=False))
        best = _best_combo(3)
        if best.empty:
            best = _best_combo(1)  # relax threshold
        if len(best):
            combo_to_plot = best.index[0]
            combo_label   = combo_to_plot.replace("_", " / ")
            print(f"    Auto-selected combo: {combo_to_plot} "
                  f"(mean forecast R% = {best.iloc[0]['mean_r']:.1f}%)")
        else:
            print("    No forecast data available for Fig 6.")
            return

    # ── Filter to countries that actually have forecasts for this combo ───────
    has_forecast = (fore_df[(fore_df.combo == combo_to_plot) &
                             fore_df.yhat.notna()]
                    ["country"].unique().tolist())
    plot_countries = [c for c in countries_to_plot if c in has_forecast]
    if not plot_countries:
        plot_countries = has_forecast[:16]
    if not plot_countries:
        print("    No countries with forecast data for this combo.")
        return

    n    = len(plot_countries)
    cols = min(4, n)
    rows = max(1, (n + cols - 1) // cols)
    fig, axes = plt.subplots(rows, cols,
                             figsize=(max(22, cols * 6.5), max(5, rows * 6.5)),
                             squeeze=False)
    axes = axes.flatten()

    # Global rendering quality
    plt.rcParams.update({
        'lines.antialiased': True,
        'patch.antialiased': True,
        'figure.dpi': 200,
    })

    for ax, country in zip(axes, plot_countries):
        # GBM rows for this country + combo
        p_sub  = gbm_df[(gbm_df.country == country) &
                         (gbm_df.combo   == combo_to_plot)]
        p_hist = p_sub[p_sub.year.isin(hist_years)].sort_values("year")
        p_fore = p_sub[p_sub.year.isin(forecast_years)].sort_values("year")

        # Fitted line (historical)
        if not p_hist.empty and p_hist["yhat"].notna().any():
            ax.plot(p_hist["year"], p_hist["yhat"],
                    color="#2980B9", linewidth=2.5,
                    solid_capstyle="round", solid_joinstyle="round",
                    label="GBM (fitted)", zorder=3)

        # Forecast line + CI band
        if not p_fore.empty:
            ax.plot(p_fore["year"], p_fore["yhat"],
                    color="#2980B9", linewidth=2.5, linestyle="--",
                    dash_capstyle="round",
                    label="GBM (forecast)", zorder=3)
            ci_ok = (p_fore["yhat_lower"].notna() &
                     p_fore["yhat_upper"].notna())
            if ci_ok.any():
                ax.fill_between(p_fore.loc[ci_ok, "year"],
                                p_fore.loc[ci_ok, "yhat_lower"],
                                p_fore.loc[ci_ok, "yhat_upper"],
                                alpha=0.20, color="#2980B9",
                                linewidth=0, label="90% CI", zorder=2)

        # Observed scatter — larger, visible dots
        act = (panel[(panel.country == country) &
                     (panel.combo   == combo_to_plot) &
                     (panel.year.isin(hist_years))]
               .sort_values("year"))
        if not act.empty and act["resistance_pct"].notna().any():
            ax.scatter(act["year"], act["resistance_pct"],
                       color="#1A1A2E", s=45, zorder=5,
                       edgecolors="white", linewidths=0.6,
                       label="Observed")

        # RF forecast overlay
        rf_sub = rf_forecasts[(rf_forecasts.country == country) &
                               (rf_forecasts.combo   == combo_to_plot)]
        if not rf_sub.empty:
            ax.plot(rf_sub["year"], rf_sub["rf_forecast"],
                    color="#E74C3C", linewidth=2.2, linestyle=(0, (4, 2)),
                    solid_capstyle="round",
                    label="RF forecast", zorder=3)

        ax.axvline(2023.5, color="#AAAAAA", linestyle=":", linewidth=1.2, zorder=1)
        ax.set_title(country, fontsize=FONT_TITLE - 1, fontweight="bold", pad=5)
        ax.tick_params(axis='both', labelsize=9)
        ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True, nbins=6))
        # Auto-scale Y: collect all values in this subplot
        _all_vals = []
        for _line in ax.get_lines():
            _yd = [v for v in _line.get_ydata() if v is not None and not np.isnan(v)]
            _all_vals.extend(_yd)
        for _coll in ax.collections:
            try:
                _paths = _coll.get_paths()
                for _p in _paths:
                    _all_vals.extend(_p.vertices[:, 1].tolist())
            except Exception:
                pass
        if _all_vals:
            _lo = max(0,   min(_all_vals) - 5)
            _hi = min(100, max(_all_vals) + 8)
            ax.set_ylim(_lo, _hi)
        else:
            ax.set_ylim(0, 100)
        ax.set_xlabel("Year", fontsize=9)
        ax.set_ylabel("% Non-Susceptible", fontsize=9)
        ax.grid(alpha=0.2, linewidth=0.7, zorder=0)
        ax.set_facecolor("#FAFAFA")

    # Hide unused subplots
    for ax in axes[n:]:
        ax.set_visible(False)

    # Shared legend
    handles, labels = next(
        (ax.get_legend_handles_labels()
         for ax in axes if ax.get_lines()), ([], [])
    )
    if handles:
        fig.legend(handles, labels, loc="lower center", ncol=min(5, len(handles)),
                   fontsize=9, bbox_to_anchor=(0.5, -0.03))

    fig.suptitle(
        f"XGBoost Quantile Forecasts (90% CI) + RF Projections (2024-2030)\n"
        f"{combo_label}  |  {len(plot_countries)} countries",
        fontsize=FONT_TITLE + 1, fontweight="bold", y=1.02
    )
    _savefig(fig, f"{figures_dir}/fig6_gbm_forecasts.png")

# ── Fig 7: 2030 Risk Tier Heatmap ────────────────────────────────────────
def plot_risk_tier_heatmap(risk_table: pd.DataFrame,
                            rf_forecasts: pd.DataFrame,
                            combos: list,
                            risk_tiers: list,
                            figures_dir: str,
                            max_combos_per_page: int = 20):
    """
    Country x combo heatmap coloured by 2030 risk tier.

    When there are more than max_combos_per_page combos, the figure is
    split into multiple pages (fig7a, fig7b, ...) so each stays readable.
    Only combos that have at least one non-Low forecast are shown —
    all-Low combos are uninformative and skipped.
    """
    print("  [Fig 7] 2030 risk tier heatmap ...")
    tier_to_num = {'Low': 1, 'Medium': 2, 'High': 3, 'Critical': 4}

    def score_to_tier(s):
        for label, lo, hi in risk_tiers:
            if lo <= s < hi:
                return label
        return risk_tiers[-1][0]

    fore_2030  = rf_forecasts[rf_forecasts.year == 2030].copy()
    combo_list = [f"{p}_{d}" for p, d in combos]
    # Use combos actually present in forecasts
    combo_list = [c for c in combo_list if c in fore_2030['combo'].unique()]
    countries  = risk_table['country'].tolist()

    # Build full matrix: countries x combos
    matrix = pd.DataFrame(index=countries, columns=combo_list, dtype=float)
    fore_map = fore_2030.set_index(['country', 'combo'])['rf_forecast']
    for c in countries:
        for combo in combo_list:
            val = fore_map.get((c, combo), None)
            if val is not None:
                matrix.loc[c, combo] = tier_to_num[score_to_tier(float(val))]
    matrix = matrix.fillna(0).astype(float)   # 0 = no data (grey)

    # Keep only combos with at least one non-Low (> 1) forecast
    informative = [c for c in combo_list
                   if (matrix[c] > 1).any()]
    if not informative:
        informative = combo_list   # fallback: show all if none qualify
    print(f"    Showing {len(informative)} informative combos "
          f"(with at least one Medium/High/Critical country) "
          f"out of {len(combo_list)} total.")

    # Sort countries by mean risk score (highest first)
    countries_sorted = (risk_table
                        .sort_values('risk_score_2030', ascending=False)
                        ['country'].tolist())
    # Filter to countries in matrix
    countries_sorted = [c for c in countries_sorted if c in matrix.index]

    from matplotlib.colors import ListedColormap, BoundaryNorm
    cmap  = ListedColormap(['#DDDDDD', '#2ECC71', '#F39C12', '#E74C3C', '#8E44AD'])
    bounds = [0, 0.5, 1.5, 2.5, 3.5, 4.5]
    norm  = BoundaryNorm(bounds, cmap.N)

    # Split combos into pages
    pages = [informative[i: i + max_combos_per_page]
             for i in range(0, len(informative), max_combos_per_page)]
    suffix_map = {0: 'a', 1: 'b', 2: 'c', 3: 'd', 4: 'e'}

    for page_idx, page_combos in enumerate(pages):
        n_c = len(countries_sorted)
        n_k = len(page_combos)
        cell_w = max(0.55, 14 / n_k)
        cell_h = max(0.30, 12 / n_c)
        fig_w  = min(32, max(12, n_k * cell_w + 4))
        fig_h  = min(40, max(8,  n_c * cell_h + 2))

        fig, ax = plt.subplots(figsize=(fig_w, fig_h))

        sub_matrix = matrix.loc[countries_sorted, page_combos].values
        im = ax.imshow(sub_matrix, cmap=cmap, norm=norm, aspect='auto')

        # X-axis: combo labels
        ax.set_xticks(range(n_k))
        x_labels = [c.replace('_', '\n', 1) for c in page_combos]
        ax.set_xticklabels(x_labels,
                           fontsize=max(6, min(9, 100 // n_k)),
                           rotation=30, ha='right')

        # Y-axis: country labels
        ax.set_yticks(range(n_c))
        ax.set_yticklabels(countries_sorted,
                           fontsize=max(7, min(10, 120 // n_c)))

        # Annotate cells: show resistance % only for non-Low cells
        for i, country in enumerate(countries_sorted):
            for j, combo in enumerate(page_combos):
                raw_val = fore_map.get((country, combo), None)
                if raw_val is not None and float(raw_val) >= 25:
                    ax.text(j, i, f"{float(raw_val):.0f}%",
                            ha='center', va='center',
                            fontsize=max(5, min(7, 80 // max(n_k, n_c))),
                            color='white', fontweight='bold')

        cbar = fig.colorbar(im, ax=ax, ticks=[0, 1, 2, 3, 4], shrink=0.5)
        cbar.ax.set_yticklabels(
            ['No data', 'Low', 'Medium', 'High', 'Critical'], fontsize=9)

        page_label = (f" (page {page_idx+1}/{len(pages)})"
                      if len(pages) > 1 else "")
        ax.set_title(
            f'Projected 2030 AMR Risk Tiers{page_label}\n'
            f'Country x Pathogen-Drug Combination  (RF Forecast)',
            fontsize=FONT_TITLE, fontweight='bold'
        )
        fig.tight_layout()

        suffix = suffix_map.get(page_idx, str(page_idx + 1))
        fname  = (f"{figures_dir}/fig7_risk_tier_heatmap.png"
                  if len(pages) == 1
                  else f"{figures_dir}/fig7{suffix}_risk_tier_heatmap.png")
        _savefig(fig, fname)



# ── Fig 8: Antibiotic Pressure Index Bubble ───────────────────────────────
def plot_api_bubble(api_df: pd.DataFrame,
                    income_groups: dict,
                    figures_dir: str,
                    target_year: int = 2022):
    """Bubble chart: DDD vs Watch% sized by SES-RS."""
    print("  [Fig 8] Antibiotic Pressure Index bubble chart …")
    sub = api_df[api_df.year == target_year].copy()
    sub['income'] = sub['country'].map(income_groups)
    sub['bubble_size'] = sub['ses_risk_score'] * 800 + 80

    fig, ax = plt.subplots(figsize=(10, 7))
    for ig, marker in INCOME_MARKERS.items():
        s = sub[sub.income == ig]
        ax.scatter(s['total_DDD'], s['watch_proportion'] * 100,
                   s=s['bubble_size'], marker=marker, alpha=0.75,
                   label=ig, edgecolors='k', linewidths=0.5)

    for _, row in sub.iterrows():
        ax.annotate(row['country'],
                    (row['total_DDD'], row['watch_proportion'] * 100),
                    fontsize=7.5, xytext=(4, 3), textcoords='offset points')

    ax.set_xlabel('Total Antibiotic Consumption (DDD/1000/day)', fontsize=FONT_LABEL)
    ax.set_ylabel('Watch-Group Proportion (%)', fontsize=FONT_LABEL)
    ax.set_title(f'Antibiotic Pressure Index — {target_year}\n'
                 '(Bubble size ∝ SES Risk Score)',
                 fontsize=FONT_TITLE, fontweight='bold')
    ax.legend(title='Income Group', fontsize=9)
    ax.grid(alpha=0.3)
    _savefig(fig, f"{figures_dir}/fig8_api_bubble.png")
