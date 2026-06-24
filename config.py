"""
AMR Temporal Forecasting Challenge — Configuration
Vivli AMR Surveillance Data Challenge 2026
"""

# ── Countries ────────────────────────────────────────────────────────────────
COUNTRIES = [
    'Nigeria', 'Kenya', 'South Africa',
    'India', 'Indonesia',
    'China',
    'Brazil', 'Mexico',
    'Germany', 'France', 'United Kingdom', 'United States'
]

# ISO-3166 alpha-2 codes used for World Bank API
COUNTRY_CODES = {
    'Nigeria': 'NG', 'Kenya': 'KE', 'South Africa': 'ZA',
    'India': 'IN', 'Indonesia': 'ID', 'China': 'CN',
    'Brazil': 'BR', 'Mexico': 'MX',
    'Germany': 'DE', 'France': 'FR',
    'United Kingdom': 'GB', 'United States': 'US'
}

# Income-group classification (World Bank tiers)
INCOME_GROUPS = {
    'Nigeria': 'LMI', 'Kenya': 'LMI', 'South Africa': 'UMI',
    'India': 'LMI', 'Indonesia': 'LMI', 'China': 'UMI',
    'Brazil': 'UMI', 'Mexico': 'UMI',
    'Germany': 'HI', 'France': 'HI', 'United Kingdom': 'HI', 'United States': 'HI'
}

# Geographic region
REGIONS = {
    'Nigeria': 'Sub-Saharan Africa', 'Kenya': 'Sub-Saharan Africa',
    'South Africa': 'Sub-Saharan Africa',
    'India': 'South/SE Asia', 'Indonesia': 'South/SE Asia',
    'China': 'East Asia',
    'Brazil': 'Latin America', 'Mexico': 'Latin America',
    'Germany': 'Western Europe', 'France': 'Western Europe',
    'United Kingdom': 'Western Europe', 'United States': 'North America'
}

# ── Pathogen–Drug Combinations ───────────────────────────────────────────────
PATHOGEN_DRUG_COMBOS = [
    ('E_coli', '3GC'),
    ('E_coli', 'FQ'),
    ('E_coli', 'CARB'),
    ('K_pneumoniae', 'CARB'),
    ('K_pneumoniae', '3GC'),
    ('S_aureus', 'MRSA'),
]

COMBO_LABELS = {
    ('E_coli', '3GC'):         'E. coli / 3rd-Gen Cephalosporins (ESBL)',
    ('E_coli', 'FQ'):          'E. coli / Fluoroquinolones',
    ('E_coli', 'CARB'):        'E. coli / Carbapenems (Last-resort)',
    ('K_pneumoniae', 'CARB'):  'K. pneumoniae / Carbapenems (CRE)',
    ('K_pneumoniae', '3GC'):   'K. pneumoniae / 3rd-Gen Cephalosporins',
    ('S_aureus', 'MRSA'):      'S. aureus / Methicillin (MRSA)',
}

# ── Time Windows ─────────────────────────────────────────────────────────────
HISTORICAL_YEARS  = list(range(2010, 2024))   # 2010–2023
FORECAST_YEARS    = list(range(2024, 2031))   # 2024–2030
ALL_YEARS         = HISTORICAL_YEARS + FORECAST_YEARS

# ── World Bank Indicator Codes ────────────────────────────────────────────────
WB_INDICATORS = {
    'gdp_per_capita':   'NY.GDP.PCAP.CD',
    'health_expenditure':'SH.XPD.CHEX.GD.ZS',
    'sanitation':       'SH.STA.SMSS.ZS',
    'physicians':       'SH.MED.PHYS.ZS',
    'hospital_beds':    'SH.MED.BEDS.ZS',
    'urbanisation':     'SP.URB.TOTL.IN.ZS',
    'agricultural_land':'AG.LND.AGRI.ZS',
    'gini':             'SI.POV.GINI',
    'water_access':     'SH.H2O.BASW.ZS',
}

# ── Risk Tier Thresholds (% resistance) ──────────────────────────────────────
RISK_TIERS = [
    ('Low',      0,   25),
    ('Medium',  25,   45),
    ('High',    45,   65),
    ('Critical',65,  100),
]

RISK_TIER_COLORS = {
    'Low':      '#2ECC71',
    'Medium':   '#F39C12',
    'High':     '#E74C3C',
    'Critical': '#8E44AD',
}

# ── SES-RS Weights (sum = 1.0) ───────────────────────────────────────────────
SES_WEIGHTS = {
    'gdp':        0.35,
    'sanitation': 0.25,
    'health_exp': 0.20,
    'physicians': 0.20,
}

# Normalisation denominators for SES-RS
SES_NORM = {
    'max_gdp_log': None,   # computed at runtime from data
    'health_exp_denom': 20.0,
    'physicians_denom':  5.0,
}

# ── Model Hyper-parameters ───────────────────────────────────────────────────
RF_PARAMS = {
    'n_estimators':    300,
    'max_depth':        10,
    'min_samples_leaf':  4,
    'random_state':     42,
    'n_jobs':           -1,
}

RIDGE_ALPHA      = 1.0
TSCV_N_SPLITS    = 5
# Confidence interval width for GBM quantile forecasts
FORECAST_CI_WIDTH = 0.90
# Number of trees per quantile model (increase for larger datasets)
GBM_N_ESTIMATORS  = 500

# ── Output Paths ─────────────────────────────────────────────────────────────
OUTPUT_DIR   = 'outputs'
FIGURES_DIR  = 'outputs/figures'
RESULTS_DIR  = 'outputs/results'
