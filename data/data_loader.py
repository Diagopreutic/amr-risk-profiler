"""
data/data_loader.py
────────────────────
Handles three data sources:

  1. Synthetic AMR panel  — mimics GLASS / Vivli isolate-level aggregates
     (real GLASS requires WHO download; Vivli requires registration)
  2. World Bank socioeconomic indicators  — live API fetch with fallback
  3. Synthetic WHO AWaRe DDD consumption  — mimics AWaRe country reports

All three converge to a single harmonised panel:
  country × pathogen_drug × year  with resistance %, consumption, and SES cols.
"""

import os
import numpy as np
import pandas as pd
import requests
import warnings
from typing import Optional

warnings.filterwarnings("ignore")

# ── Realistic seed parameters for synthetic AMR generation ──────────────────
# Approximate 2023 resistance % from published GLASS / meta-analysis data
# Format: (median_2023, annual_trend_slope, noise_sd)
AMR_PARAMS = {
    # (pathogen, drug)  →  { country: (base_2023, annual_delta, noise) }
    ('E_coli', '3GC'): {
        'India': (72, 0.4, 2.0), 'Nigeria': (64, 0.6, 2.5), 'Kenya': (58, 0.5, 2.2),
        'South Africa': (48, 0.3, 1.8), 'Indonesia': (62, 0.5, 2.0), 'China': (52, 0.2, 1.5),
        'Brazil': (44, 0.3, 1.8), 'Mexico': (42, 0.3, 1.6), 'Germany': (12, 0.1, 0.8),
        'France': (14, 0.1, 0.9), 'United Kingdom': (13, 0.1, 0.8), 'United States': (18, 0.15, 1.0),
    },
    ('E_coli', 'FQ'): {
        'India': (68, 0.35, 2.0), 'Nigeria': (54, 0.45, 2.2), 'Kenya': (52, 0.4, 2.0),
        'South Africa': (44, 0.25, 1.8), 'Indonesia': (58, 0.4, 1.9), 'China': (62, 0.2, 1.5),
        'Brazil': (36, 0.2, 1.6), 'Mexico': (38, 0.25, 1.5), 'Germany': (18, 0.12, 0.9),
        'France': (22, 0.15, 1.0), 'United Kingdom': (19, 0.1, 0.9), 'United States': (24, 0.18, 1.1),
    },
    ('E_coli', 'CARB'): {
        'India': (14, 0.5, 1.2), 'Nigeria': (9, 0.4, 1.0), 'Kenya': (7, 0.35, 0.9),
        'South Africa': (5, 0.25, 0.7), 'Indonesia': (10, 0.4, 1.0), 'China': (18, 0.45, 1.2),
        'Brazil': (8, 0.3, 0.9), 'Mexico': (7, 0.25, 0.8), 'Germany': (1.5, 0.05, 0.3),
        'France': (1.8, 0.06, 0.3), 'United Kingdom': (1.2, 0.04, 0.25), 'United States': (2.5, 0.1, 0.4),
    },
    ('K_pneumoniae', 'CARB'): {
        'India': (22, 0.6, 1.5), 'Nigeria': (14, 0.5, 1.2), 'Kenya': (10, 0.4, 1.0),
        'South Africa': (8, 0.35, 0.9), 'Indonesia': (16, 0.5, 1.3), 'China': (24, 0.55, 1.5),
        'Brazil': (18, 0.45, 1.2), 'Mexico': (12, 0.4, 1.0), 'Germany': (4, 0.15, 0.5),
        'France': (5, 0.18, 0.5), 'United Kingdom': (3.5, 0.12, 0.45), 'United States': (10, 0.3, 0.8),
    },
    ('K_pneumoniae', '3GC'): {
        'India': (64, 0.4, 2.0), 'Nigeria': (56, 0.5, 2.2), 'Kenya': (52, 0.45, 2.0),
        'South Africa': (46, 0.3, 1.8), 'Indonesia': (58, 0.42, 2.0), 'China': (54, 0.25, 1.6),
        'Brazil': (48, 0.35, 1.9), 'Mexico': (45, 0.3, 1.7), 'Germany': (17, 0.12, 1.0),
        'France': (20, 0.15, 1.0), 'United Kingdom': (16, 0.1, 0.9), 'United States': (24, 0.18, 1.1),
    },
    ('S_aureus', 'MRSA'): {
        'India': (38, 0.3, 1.8), 'Nigeria': (28, 0.3, 1.6), 'Kenya': (25, 0.25, 1.5),
        'South Africa': (22, 0.2, 1.4), 'Indonesia': (32, 0.28, 1.7), 'China': (34, 0.22, 1.5),
        'Brazil': (24, 0.2, 1.4), 'Mexico': (26, 0.22, 1.5), 'Germany': (12, -0.1, 0.8),
        'France': (18, -0.05, 0.9), 'United Kingdom': (10, -0.2, 0.7), 'United States': (28, -0.15, 1.2),
    },
}

# Approximate DDD/1000/day and Watch-group proportions for 2023 (AWaRe proxy)
AWARE_PARAMS = {
    # country: (total_DDD_2023, watch_prop_2023, annual_delta_DDD, annual_delta_watch)
    'India':          (20.0, 0.42, 0.4, 0.005), 'Nigeria':        (14.0, 0.38, 0.3, 0.004),
    'Kenya':          (12.0, 0.35, 0.25, 0.004), 'South Africa':   (15.0, 0.32, 0.2, 0.003),
    'Indonesia':      (17.0, 0.40, 0.35, 0.004), 'China':          (22.0, 0.44, 0.3, 0.003),
    'Brazil':         (16.0, 0.34, 0.2, 0.003), 'Mexico':         (15.0, 0.33, 0.18, 0.003),
    'Germany':        (10.0, 0.22, 0.05, 0.001), 'France':         (12.0, 0.24, 0.05, 0.001),
    'United Kingdom': (11.0, 0.20, 0.04, 0.001), 'United States':  (18.0, 0.28, 0.1, 0.002),
}


def generate_synthetic_amr(years: list, countries: list,
                            combos: list, seed: int = 42) -> pd.DataFrame:
    """
    Generate a realistic synthetic AMR panel that mirrors aggregated GLASS data.
    
    Returns
    -------
    DataFrame with columns:
        country, pathogen, drug, combo, year, resistance_pct
    """
    rng   = np.random.default_rng(seed)
    rows  = []

    for (pathogen, drug) in combos:
        for country in countries:
            params = AMR_PARAMS[(pathogen, drug)][country]
            base_2023, slope, noise_sd = params
            for year in years:
                delta = year - 2023
                pct   = base_2023 + slope * delta + rng.normal(0, noise_sd)
                pct   = float(np.clip(pct, 0.5, 98.0))
                rows.append({
                    'country':        country,
                    'pathogen':       pathogen,
                    'drug':           drug,
                    'combo':          f"{pathogen}_{drug}",
                    'year':           year,
                    'resistance_pct': round(pct, 2),
                })

    return pd.DataFrame(rows)


def generate_synthetic_aware(years: list, countries: list, seed: int = 42) -> pd.DataFrame:
    """
    Generate synthetic WHO AWaRe DDD consumption data.

    Returns
    -------
    DataFrame with columns:
        country, year, total_DDD, watch_proportion
    """
    rng  = np.random.default_rng(seed + 1)
    rows = []

    for country in countries:
        base_ddd, base_watch, delta_ddd, delta_watch = AWARE_PARAMS[country]
        for year in years:
            d = year - 2023
            ddd   = float(np.clip(base_ddd   + delta_ddd   * d + rng.normal(0, 0.4), 5, 40))
            watch = float(np.clip(base_watch + delta_watch * d + rng.normal(0, 0.01), 0.05, 0.70))
            rows.append({'country': country, 'year': year,
                         'total_DDD': round(ddd, 2),
                         'watch_proportion': round(watch, 4)})
    return pd.DataFrame(rows)


# ── World Bank fetcher ───────────────────────────────────────────────────────
# ── World Bank fetcher ────────────────────────────────────────────────────────
# Cache path: fetched data is saved here so subsequent runs never need the API.
WB_CACHE_PATH = os.path.join(os.path.dirname(__file__), 'wb_cache.csv')
# Cache TTL in days (refresh if older than this)
WB_CACHE_TTL_DAYS = 30


def _wb_fetch_one(iso2: str, wb_code: str, ind_name: str,
                   start_year: int, end_year: int) -> list:
    """
    Fetch a single indicator for a single country.
    Returns a list of {country_code, year, ind_name} dicts, or [].

    Fetching one country at a time avoids proxy/gateway rejections that
    occur when multiple ISO-2 codes are joined by semicolons in the URL
    path (HTTP 502 from Microsoft ISA / Forefront proxies).
    """
    import time
    url = (
        f"https://api.worldbank.org/v2/country/{iso2}"
        f"/indicator/{wb_code}"
        f"?date={start_year}:{end_year}&format=json&per_page=500"
    )
    for attempt in range(1, 4):
        try:
            resp = requests.get(url, timeout=30)
            if resp.status_code == 502:
                # 502 from proxy — brief pause then retry
                time.sleep(attempt * 3)
                continue
            if resp.status_code != 200:
                return []
            payload = resp.json()
            if len(payload) >= 2 and payload[1]:
                return [
                    {
                        "country_code": r["countryiso3code"],
                        "year":         int(r["date"]),
                        ind_name:       float(r["value"]),
                    }
                    for r in payload[1]
                    if r.get("value") is not None
                ]
            return []
        except requests.exceptions.Timeout:
            time.sleep(attempt * 5)
        except requests.exceptions.ConnectionError:
            return []
        except Exception:
            return []
    return []


def _wb_fetch_all_indicators(country_iso2_list: list,
                              wb_indicators: dict,
                              start_year: int,
                              end_year: int,
                              max_workers: int = 10) -> pd.DataFrame:
    """
    Fetch ALL World Bank indicators for all countries IN PARALLEL.

    Uses ThreadPoolExecutor to fire up to `max_workers` requests
    simultaneously, reducing fetch time from ~450s to ~45s for 59 countries.
    Each request is one (country, indicator) pair to avoid proxy issues
    with semicolons in URLs.

    Returns a wide DataFrame: country_code (ISO-3), year, <indicator cols...>
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    n_countries  = len(country_iso2_list)
    n_indicators = len(wb_indicators)
    total_calls  = n_countries * n_indicators

    # Build all (iso2, ind_name, wb_code) tasks
    tasks = [
        (iso2, ind_name, wb_code)
        for iso2 in country_iso2_list
        for ind_name, wb_code in wb_indicators.items()
    ]

    all_records = []
    done = 0
    last_printed = 0

    def _fetch_task(args):
        iso2, ind_name, wb_code = args
        return _wb_fetch_one(iso2, wb_code, ind_name, start_year, end_year)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_fetch_task, t): t for t in tasks}
        for future in as_completed(futures):
            iso2, ind_name, _ = futures[future]
            done += 1
            try:
                records = future.result()
                all_records.extend(records)
            except Exception:
                pass
            # Print progress every ~10% of total calls
            pct = int(done / total_calls * 10)
            if pct > last_printed:
                last_printed = pct
                print(f"    [WB] {done}/{total_calls} calls done "
                      f"({done/total_calls*100:.0f}%) ...", flush=True)

    # Final count by country
    if all_records:
        done_countries = len(set(r["country_code"] for r in all_records))
        print(f"    [WB] Complete: {len(all_records)} records "
              f"from {done_countries} country codes.", flush=True)

    if not all_records:
        return pd.DataFrame()

    # Pivot flat records -> wide DataFrame
    df_long  = pd.DataFrame(all_records)
    ind_cols = list(wb_indicators.keys())
    all_wide = None
    for ind_name in ind_cols:
        if ind_name not in df_long.columns:
            continue
        df_ind = (df_long[["country_code", "year", ind_name]]
                  .dropna(subset=[ind_name])
                  .drop_duplicates(["country_code", "year"]))
        if all_wide is None:
            all_wide = df_ind
        else:
            all_wide = all_wide.merge(df_ind, on=["country_code", "year"], how="outer")

    return all_wide if all_wide is not None else pd.DataFrame()

# Fallback values for World Bank indicators (median estimates, circa 2019-2022)
WB_FALLBACK = {
    'Nigeria':        {'gdp_per_capita': 2100,  'health_expenditure': 3.8,  'sanitation': 42.0, 'physicians': 0.4,  'hospital_beds': 0.5,  'urbanisation': 52.0, 'gini': 35.0, 'water_access': 71.0},
    'Kenya':          {'gdp_per_capita': 2010,  'health_expenditure': 4.5,  'sanitation': 53.0, 'physicians': 0.2,  'hospital_beds': 1.4,  'urbanisation': 28.0, 'gini': 40.8, 'water_access': 58.0},
    'South Africa':   {'gdp_per_capita': 6100,  'health_expenditure': 8.5,  'sanitation': 75.0, 'physicians': 0.9,  'hospital_beds': 2.3,  'urbanisation': 67.0, 'gini': 63.0, 'water_access': 88.0},
    'India':          {'gdp_per_capita': 2400,  'health_expenditure': 3.0,  'sanitation': 64.0, 'physicians': 0.7,  'hospital_beds': 0.5,  'urbanisation': 35.0, 'gini': 35.7, 'water_access': 93.0},
    'Indonesia':      {'gdp_per_capita': 4400,  'health_expenditure': 2.9,  'sanitation': 79.0, 'physicians': 0.4,  'hospital_beds': 1.0,  'urbanisation': 57.0, 'gini': 38.2, 'water_access': 90.0},
    'China':          {'gdp_per_capita': 12600, 'health_expenditure': 5.4,  'sanitation': 85.0, 'physicians': 2.0,  'hospital_beds': 4.3,  'urbanisation': 63.0, 'gini': 38.5, 'water_access': 97.0},
    'Brazil':         {'gdp_per_capita': 8800,  'health_expenditure': 9.9,  'sanitation': 88.0, 'physicians': 2.3,  'hospital_beds': 2.1,  'urbanisation': 87.0, 'gini': 48.9, 'water_access': 98.0},
    'Mexico':         {'gdp_per_capita': 10100, 'health_expenditure': 5.5,  'sanitation': 90.0, 'physicians': 2.4,  'hospital_beds': 1.4,  'urbanisation': 80.0, 'gini': 45.4, 'water_access': 97.0},
    'Germany':        {'gdp_per_capita': 48200, 'health_expenditure': 11.2, 'sanitation': 99.0, 'physicians': 4.5,  'hospital_beds': 8.0,  'urbanisation': 77.0, 'gini': 31.9, 'water_access': 100.0},
    'France':         {'gdp_per_capita': 43000, 'health_expenditure': 11.1, 'sanitation': 98.0, 'physicians': 3.4,  'hospital_beds': 5.7,  'urbanisation': 81.0, 'gini': 31.6, 'water_access': 100.0},
    'United Kingdom': {'gdp_per_capita': 46500, 'health_expenditure': 10.2, 'sanitation': 99.0, 'physicians': 3.0,  'hospital_beds': 2.5,  'urbanisation': 84.0, 'gini': 35.1, 'water_access': 100.0},
    'United States':  {'gdp_per_capita': 65200, 'health_expenditure': 16.8, 'sanitation': 100.0,'physicians': 2.9,  'hospital_beds': 2.9,  'urbanisation': 83.0, 'gini': 41.5, 'water_access': 99.0},
}

# Annual growth rates for synthetic extension of WB data
WB_GROWTH = {
    'gdp_per_capita': 0.025, 'health_expenditure': 0.005,
    'sanitation': 0.004,     'physicians': 0.008,
    'hospital_beds': 0.003,  'urbanisation': 0.004,
    'gini': -0.001,          'water_access': 0.003,
}

WB_NOISE_SD = {
    'gdp_per_capita': 0.03, 'health_expenditure': 0.01,
    'sanitation': 0.005,    'physicians': 0.02,
    'hospital_beds': 0.01,  'urbanisation': 0.005,
    'gini': 0.005,          'water_access': 0.003,
}


def _build_synthetic_wb(years: list, countries: list, seed: int = 99) -> pd.DataFrame:
    """Build a synthetic World Bank panel when API is unavailable."""
    rng  = np.random.default_rng(seed)
    rows = []
    indics = list(WB_FALLBACK['Germany'].keys())

    # Generic fallback values for countries not in WB_FALLBACK
    _GENERIC_WB = {
        'gdp_per_capita': 15000, 'health_expenditure': 6.0,
        'sanitation': 75.0, 'physicians': 1.5, 'hospital_beds': 2.5,
        'urbanisation': 60.0, 'agricultural_land': 35.0,
        'gini': 38.0, 'water_access': 90.0,
    }
    for country in countries:
        base = WB_FALLBACK.get(country, _GENERIC_WB).copy()
        for year in years:
            d = year - 2022
            row = {'country': country, 'year': year}
            for ind in indics:
                growth = WB_GROWTH.get(ind, 0)
                noise  = rng.normal(0, WB_NOISE_SD.get(ind, 0.01))
                val    = base[ind] * ((1 + growth + noise) ** d)
                row[ind] = round(float(max(val, 0.01)), 3)
            rows.append(row)
    return pd.DataFrame(rows)


# Complete ISO-2 → ISO-3 → country-name lookup table.
# The WB API accepts ISO-2 in the URL but returns ISO-3 in countryiso3code.
# We build the reverse map (ISO-3 → name) once here so load_worldbank_data
# can map API results straight to country names without any extra HTTP calls.
_ISO3_TO_NAME = {
    'NGA':'Nigeria',       'KEN':'Kenya',         'ZAF':'South Africa',
    'IND':'India',         'IDN':'Indonesia',      'CHN':'China',
    'BRA':'Brazil',        'MEX':'Mexico',         'DEU':'Germany',
    'FRA':'France',        'GBR':'United Kingdom', 'USA':'United States',
    'JPN':'Japan',         'AUS':'Australia',      'CAN':'Canada',
    'RUS':'Russia',        'KOR':'South Korea',    'THA':'Thailand',
    'VNM':'Vietnam',       'PHL':'Philippines',    'PAK':'Pakistan',
    'BGD':'Bangladesh',    'EGY':'Egypt',          'ETH':'Ethiopia',
    'TZA':'Tanzania',      'UGA':'Uganda',         'GHA':'Ghana',
    'SEN':'Senegal',       'CIV':'Ivory Coast',    'CMR':'Cameroon',
    'ARG':'Argentina',     'COL':'Colombia',       'PER':'Peru',
    'CHL':'Chile',         'ESP':'Spain',          'ITA':'Italy',
    'PRT':'Portugal',      'NLD':'Netherlands',    'SWE':'Sweden',
    'NOR':'Norway',        'DNK':'Denmark',        'FIN':'Finland',
    'POL':'Poland',        'UKR':'Ukraine',        'TUR':'Turkey',
    'SAU':'Saudi Arabia',  'IRN':'Iran',           'IRQ':'Iraq',
    'ISR':'Israel',        'MYS':'Malaysia',       'SGP':'Singapore',
    'NZL':'New Zealand',   'ZWE':'Zimbabwe',       'ZMB':'Zambia',
    'MAR':'Morocco',       'DZA':'Algeria',        'TUN':'Tunisia',
    'LKA':'Sri Lanka',     'NPL':'Nepal',          'MMR':'Myanmar',
    'KHM':'Cambodia',      'LAO':'Laos',           'MNG':'Mongolia',
    'CRI':'Costa Rica',    'PAN':'Panama',         'GTM':'Guatemala',
    'HND':'Honduras',      'SLV':'El Salvador',    'NIC':'Nicaragua',
    'BOL':'Bolivia',       'PRY':'Paraguay',       'URY':'Uruguay',
    'ECU':'Ecuador',       'VEN':'Venezuela',
}

# Also build ISO-2 → ISO-3 for use by _wb_fetch_indicator
_ISO2_TO_ISO3 = {
    'NG':'NGA','KE':'KEN','ZA':'ZAF','IN':'IND','ID':'IDN','CN':'CHN',
    'BR':'BRA','MX':'MEX','DE':'DEU','FR':'FRA','GB':'GBR','US':'USA',
    'JP':'JPN','AU':'AUS','CA':'CAN','RU':'RUS','KR':'KOR','TH':'THA',
    'VN':'VNM','PH':'PHL','PK':'PAK','BD':'BGD','EG':'EGY','ET':'ETH',
    'TZ':'TZA','UG':'UGA','GH':'GHA','SN':'SEN','CI':'CIV','CM':'CMR',
    'AR':'ARG','CO':'COL','PE':'PER','CL':'CHL','ES':'ESP','IT':'ITA',
    'PT':'PRT','NL':'NLD','SE':'SWE','NO':'NOR','DK':'DNK','FI':'FIN',
    'PL':'POL','UA':'UKR','TR':'TUR','SA':'SAU','IR':'IRN','IQ':'IRQ',
    'IL':'ISR','MY':'MYS','SG':'SGP','NZ':'NZL','ZW':'ZWE','ZM':'ZMB',
    'MA':'MAR','DZ':'DZA','TN':'TUN','LK':'LKA','NP':'NPL','MM':'MMR',
    'KH':'KHM','LA':'LAO','MN':'MNG','CR':'CRI','PA':'PAN','GT':'GTM',
    'HN':'HND','SV':'SLV','NI':'NIC','BO':'BOL','PY':'PRY','UY':'URY',
    'EC':'ECU','VE':'VEN',
}

# ── Comprehensive country name → ISO-2 mapping ───────────────────────────────
# Covers 150+ country name variants found in surveillance datasets.
# Used by assemble_panel_from_files so every discovered country gets
# a valid ISO-2 code for the World Bank API — not just the first 2 letters.
_COUNTRY_NAME_TO_ISO2 = {
    # Core 12 pipeline countries
    'Nigeria': 'NG', 'Kenya': 'KE', 'South Africa': 'ZA',
    'India': 'IN', 'Indonesia': 'ID', 'China': 'CN',
    'Brazil': 'BR', 'Mexico': 'MX', 'Germany': 'DE',
    'France': 'FR', 'United Kingdom': 'GB', 'United States': 'US',
    # Europe
    'Albania': 'AL', 'Austria': 'AT', 'Belarus': 'BY', 'Belgium': 'BE',
    'Bosnia And Herzegovina': 'BA', 'Bulgaria': 'BG', 'Croatia': 'HR',
    'Cyprus': 'CY', 'Czech Republic': 'CZ', 'Czechia': 'CZ',
    'Denmark': 'DK', 'Estonia': 'EE', 'Finland': 'FI', 'Georgia': 'GE',
    'Greece': 'GR', 'Hungary': 'HU', 'Iceland': 'IS', 'Ireland': 'IE',
    'Italy': 'IT', 'Kosovo': 'XK', 'Latvia': 'LV', 'Lithuania': 'LT',
    'Luxembourg': 'LU', 'Malta': 'MT', 'Moldova': 'MD',
    'Montenegro': 'ME', 'Netherlands': 'NL', 'North Macedonia': 'MK',
    'Norway': 'NO', 'Poland': 'PL', 'Portugal': 'PT', 'Romania': 'RO',
    'Russia': 'RU', 'Russian Federation': 'RU',
    'Serbia': 'RS', 'Slovakia': 'SK', 'Slovenia': 'SI', 'Spain': 'ES',
    'Sweden': 'SE', 'Switzerland': 'CH', 'Turkey': 'TR', 'Turkiye': 'TR',
    'Ukraine': 'UA',
    # Americas
    'Argentina': 'AR', 'Bolivia': 'BO', 'Canada': 'CA', 'Chile': 'CL',
    'Colombia': 'CO', 'Costa Rica': 'CR', 'Cuba': 'CU',
    'Dominican Republic': 'DO', 'Ecuador': 'EC', 'El Salvador': 'SV',
    'Guatemala': 'GT', 'Honduras': 'HN', 'Jamaica': 'JM',
    'Nicaragua': 'NI', 'Panama': 'PA', 'Paraguay': 'PY', 'Peru': 'PE',
    'Puerto Rico': 'PR', 'Trinidad And Tobago': 'TT', 'Uruguay': 'UY',
    'Venezuela': 'VE',
    # Middle East
    'Bahrain': 'BH', 'Egypt': 'EG', 'Iran': 'IR',
    'Iran, Islamic Rep.': 'IR', 'Iraq': 'IQ', 'Israel': 'IL',
    'Jordan': 'JO', 'Kuwait': 'KW', 'Lebanon': 'LB', 'Libya': 'LY',
    'Morocco': 'MA', 'Oman': 'OM', 'Qatar': 'QA', 'Saudi Arabia': 'SA',
    'Syria': 'SY', 'Tunisia': 'TN', 'United Arab Emirates': 'AE',
    'Yemen': 'YE',
    # Asia-Pacific
    'Afghanistan': 'AF', 'Australia': 'AU', 'Azerbaijan': 'AZ',
    'Bangladesh': 'BD', 'Bhutan': 'BT', 'Cambodia': 'KH',
    'Hong Kong': 'HK', 'Japan': 'JP', 'Kazakhstan': 'KZ',
    'Korea, South': 'KR', 'South Korea': 'KR', 'Republic Of Korea': 'KR',
    'Korea': 'KR', 'Kyrgyzstan': 'KG', 'Laos': 'LA', 'Malaysia': 'MY',
    'Maldives': 'MV', 'Mongolia': 'MN', 'Myanmar': 'MM', 'Nepal': 'NP',
    'New Zealand': 'NZ', 'Pakistan': 'PK', 'Philippines': 'PH',
    'Singapore': 'SG', 'Sri Lanka': 'LK', 'Taiwan': 'TW',
    'Tajikistan': 'TJ', 'Thailand': 'TH', 'Uzbekistan': 'UZ',
    'Vietnam': 'VN', 'Viet Nam': 'VN',
    # Africa
    'Algeria': 'DZ', 'Angola': 'AO', 'Cameroon': 'CM',
    'Democratic Republic Of The Congo': 'CD', 'Dr Congo': 'CD',
    'Ethiopia': 'ET', 'Ghana': 'GH', 'Ivory Coast': 'CI',
    'Madagascar': 'MG', 'Malawi': 'MW', 'Mali': 'ML', 'Mozambique': 'MZ',
    'Namibia': 'NA', 'Niger': 'NE', 'Rwanda': 'RW', 'Senegal': 'SN',
    'Sudan': 'SD', 'Tanzania': 'TZ', 'Uganda': 'UG', 'Zambia': 'ZM',
    'Zimbabwe': 'ZW',
}


def load_worldbank_data(years: list, countries: list,
                        country_codes: dict, wb_indicators: dict) -> pd.DataFrame:
    """
    Return World Bank socioeconomic indicators for all countries and years.

    Data strategy (in priority order)
    ----------------------------------
    1. LOCAL CACHE  (data/wb_cache.csv)
       If the cache file exists and is less than WB_CACHE_TTL_DAYS old,
       load it instantly — no network call at all.
       This means the API is only ever contacted once every 30 days.

    2. LIVE API FETCH
       If no cache (or cache is stale), fetch from the World Bank API.
       Results are saved to the cache immediately so the next run is fast.
       Each indicator is retried up to 3 times with increasing delay
       (5 s, 10 s, 15 s) to handle transient timeouts.

    3. SYNTHETIC FALLBACK
       If the API is genuinely unreachable, fall back to evidence-based
       synthetic values and warn the user.
    """
    import time as _time

    # ── Identify countries we have ISO codes for ──────────────────────
    known   = {c: country_codes[c] for c in countries if c in country_codes}
    unknown = [c for c in countries if c not in country_codes]
    if unknown:
        print(f"  [WB] No ISO code for: {unknown} — synthetic values used.")

    # Extend name map with any country whose ISO-2 is known but not in _ISO3_TO_NAME
    name_map = dict(_ISO3_TO_NAME)
    for name, iso2 in known.items():
        iso3 = _ISO2_TO_ISO3.get(iso2, iso2)
        if iso3 not in name_map:
            name_map[iso3] = name

    # ── Try cache first ───────────────────────────────────────────────
    hist_years = [y for y in years if y <= 2023]
    cache_df   = _load_wb_cache(known, hist_years, wb_indicators)

    if cache_df is not None:
        print(f"  [data_loader] World Bank data loaded from local cache "
              f"({WB_CACHE_PATH}).")
    else:
        # ── Live API fetch ────────────────────────────────────────────
        if not known:
            print("  [data_loader] No ISO codes → using synthetic WB data.")
            return _build_and_extend_synthetic(years, countries, wb_indicators)

        print(f"  [data_loader] Fetching {len(wb_indicators)} indicators "
              f"for {len(known)} countries from World Bank API ...")
        iso2_list = list(known.values())
        raw = _wb_fetch_all_indicators(
            iso2_list, wb_indicators,
            min(hist_years), max(hist_years)
        )

        if raw.empty:
            print("  [data_loader] World Bank API unavailable "
                  "-> using synthetic WB data.")
            return _build_and_extend_synthetic(years, countries, wb_indicators)

        # Map ISO-3 -> country name
        raw["country"] = raw["country_code"].map(name_map)
        raw = raw.dropna(subset=["country"])

        if raw.empty:
            print("  [data_loader] ISO mapping produced no rows "
                  "-> using synthetic WB data.")
            return _build_and_extend_synthetic(years, countries, wb_indicators)

        ind_cols = [c for c in raw.columns
                    if c not in ("country_code", "country", "year")]
        cache_df = raw[["country", "year"] + ind_cols].copy()

        # Add sentinel rows (NaN values) for countries that returned no data
        # from the WB API.  Without this, the cache validator sees them as
        # "missing" on every subsequent run and triggers a full re-fetch.
        fetched_countries = set(cache_df["country"].unique())
        no_data_countries = [c for c in known.keys()
                             if c not in fetched_countries]
        if no_data_countries:
            sentinel_rows = pd.DataFrame([
                {"country": c, "year": hist_years[0]}
                for c in no_data_countries
            ])
            for col in ind_cols:
                sentinel_rows[col] = np.nan
            cache_df = pd.concat([cache_df, sentinel_rows], ignore_index=True)
            print(f"  [WB] {len(no_data_countries)} countries have no WB data "
                  f"(stored as sentinel): {no_data_countries}")

        # Save to cache
        cache_df.to_csv(WB_CACHE_PATH, index=False)
        print(f"  [data_loader] World Bank API: fetched "
              f"{len(ind_cols)} indicators for {len(fetched_countries)} "
              f"countries. Saved to cache -> {WB_CACHE_PATH}")

    # ── Extend to full year range + fill all gaps ─────────────────────
    return _build_and_extend_wb(cache_df, years, countries, wb_indicators)


def _load_wb_cache(known: dict, hist_years: list,
                   wb_indicators: dict):
    """
    Return cached WB DataFrame if the cache file is valid, else None.
    Valid = file exists, is < WB_CACHE_TTL_DAYS old, and covers the needed
    countries and indicator columns.
    """
    import time as _time

    if not os.path.exists(WB_CACHE_PATH):
        return None

    # Check age
    age_days = (_time.time() - os.path.getmtime(WB_CACHE_PATH)) / 86400
    if age_days > WB_CACHE_TTL_DAYS:
        print(f"  [WB] Cache is {age_days:.0f} days old (>{WB_CACHE_TTL_DAYS}) "
              f"— will refresh from API.")
        return None

    try:
        df = pd.read_csv(WB_CACHE_PATH)
    except Exception as e:
        print(f"  [WB] Could not read cache: {e}")
        return None

    # Check required columns are present
    needed_cols = {"country", "year"} | set(wb_indicators.keys())
    if not needed_cols.issubset(set(df.columns)):
        missing = needed_cols - set(df.columns)
        print(f"  [WB] Cache missing columns {missing} — will refresh.")
        return None

    # Check that needed countries are covered.
    # A country is considered "covered" if it appears in the cache at all —
    # even as a sentinel row with NaN values (meaning the WB API has no data
    # for it).  Only countries that are completely absent from the cache
    # (i.e. never been attempted) should trigger a refresh.
    cached_countries = set(df["country"].unique())
    needed_countries = set(known.keys())
    truly_missing = needed_countries - cached_countries
    if truly_missing:
        print(f"  [WB] Cache missing countries {truly_missing} — will refresh.")
        return None

    return df


def _build_and_extend_synthetic(years, countries, wb_indicators):
    """Convenience: build pure synthetic panel for all years."""
    synth = _build_synthetic_wb(years, countries)
    # Align columns with wb_indicators keys
    for col in wb_indicators.keys():
        if col not in synth.columns:
            synth[col] = np.nan
    return synth


def _build_and_extend_wb(cache_df: pd.DataFrame, years: list,
                          countries: list, wb_indicators: dict) -> pd.DataFrame:
    """
    Take the (possibly partial) fetched/cached DataFrame and:
      - Expand to cover ALL countries x ALL years
      - Fill forecast years and any NaN cells from _build_synthetic_wb
      - Smooth remaining gaps with linear interpolation
    """
    ind_cols = list(wb_indicators.keys())

    # Full skeleton: every country x every year
    skeleton = pd.DataFrame(
        [(c, y) for c in countries for y in years],
        columns=["country", "year"],
    )
    merged = skeleton.merge(
        cache_df[["country", "year"] +
                 [c for c in ind_cols if c in cache_df.columns]],
        on=["country", "year"], how="left"
    )

    # Fill NaN cells from synthetic baseline
    synth     = _build_synthetic_wb(years, countries)
    synth_idx = synth.set_index(["country", "year"])

    for col in ind_cols:
        if col not in merged.columns:
            merged[col] = np.nan
        missing_mask = merged[col].isna()
        if missing_mask.any() and col in synth_idx.columns:
            def _fill(r):
                key = (r["country"], r["year"])
                return synth_idx.loc[key, col] if key in synth_idx.index else np.nan
            merged.loc[missing_mask, col] = (
                merged.loc[missing_mask].apply(_fill, axis=1)
            )

    # Final smoothing
    merged = merged.sort_values(["country", "year"])
    for col in ind_cols:
        if col in merged.columns:
            merged[col] = (
                merged.groupby("country")[col]
                .transform(lambda x:
                    x.interpolate(method="linear", limit=4).ffill().bfill())
            )

    return merged.reset_index(drop=True)

def _fetch_iso3_map(iso2_list: list) -> dict:
    """
    Fetch ISO-2 → ISO-3 country code mapping from the World Bank API.
    Returns dict {iso2: iso3}. Falls back to pycountry or a hardcoded table.
    """
    # Hardcoded table for the most common codes (fast, no extra API call)
    HARDCODED = {
        'NG': 'NGA', 'KE': 'KEN', 'ZA': 'ZAF', 'IN': 'IND', 'ID': 'IDN',
        'CN': 'CHN', 'BR': 'BRA', 'MX': 'MEX', 'DE': 'DEU', 'FR': 'FRA',
        'GB': 'GBR', 'US': 'USA', 'JP': 'JPN', 'AU': 'AUS', 'CA': 'CAN',
        'RU': 'RUS', 'KR': 'KOR', 'TH': 'THA', 'VN': 'VNM', 'PH': 'PHL',
        'PK': 'PAK', 'BD': 'BGD', 'EG': 'EGY', 'ET': 'ETH', 'TZ': 'TZA',
        'UG': 'UGA', 'GH': 'GHA', 'SN': 'SEN', 'CI': 'CIV', 'CM': 'CMR',
        'AR': 'ARG', 'CO': 'COL', 'PE': 'PER', 'VE': 'VEN', 'CL': 'CHL',
        'ES': 'ESP', 'IT': 'ITA', 'PT': 'PRT', 'NL': 'NLD', 'SE': 'SWE',
        'NO': 'NOR', 'DK': 'DNK', 'FI': 'FIN', 'PL': 'POL', 'UA': 'UKR',
        'TR': 'TUR', 'SA': 'SAU', 'IR': 'IRN', 'IQ': 'IRQ', 'IL': 'ISR',
        'MY': 'MYS', 'SG': 'SGP', 'NZ': 'NZL', 'ZW': 'ZWE', 'ZM': 'ZMB',
    }
    result = {}
    for iso2 in iso2_list:
        if iso2 in HARDCODED:
            result[iso2] = HARDCODED[iso2]
        else:
            # Try a lightweight WB API call for unknown codes
            try:
                url  = f"https://api.worldbank.org/v2/country/{iso2}?format=json"
                resp = requests.get(url, timeout=8)
                data = resp.json()
                if len(data) >= 2 and data[1]:
                    result[iso2] = data[1][0].get('id', iso2)
                else:
                    result[iso2] = iso2   # fallback: use iso2 as-is
            except Exception:
                result[iso2] = iso2
    return result


def assemble_panel(years: list, countries: list, combos: list,
                   country_codes: dict, wb_indicators: dict,
                   seed: int = 42) -> pd.DataFrame:
    """
    Assemble the full harmonised panel:
        country × combo × year
    with columns:
        resistance_pct, total_DDD, watch_proportion,
        gdp_per_capita, health_expenditure, sanitation,
        physicians, hospital_beds, urbanisation,
        gini, water_access
    """
    print("[data_loader] Generating synthetic AMR surveillance data …")
    df_amr = generate_synthetic_amr(years, countries, combos, seed=seed)

    print("[data_loader] Generating synthetic WHO AWaRe consumption data …")
    df_aware = generate_synthetic_aware(years, countries, seed=seed)

    print("[data_loader] Fetching World Bank socioeconomic indicators …")
    df_wb = load_worldbank_data(years, countries, country_codes, wb_indicators)

    # Merge AWaRe into AMR panel (country × year)
    df = df_amr.merge(df_aware, on=['country', 'year'], how='left')

    # Merge World Bank indicators (country × year)
    df = df.merge(df_wb, on=['country', 'year'], how='left')

    # Forward-fill minor WB gaps (≤2 years)
    wb_cols = list(WB_FALLBACK['Germany'].keys())
    df = df.sort_values(['country', 'combo', 'year'])
    for col in wb_cols:
        if col in df.columns:
            df[col] = (df.groupby(['country', 'combo'])[col]
                         .transform(lambda x: x.interpolate(method='linear', limit=2)
                                               .ffill().bfill()))

    df = df.reset_index(drop=True)
    print(f"[data_loader] Panel assembled: {len(df):,} rows "
          f"({df.country.nunique()} countries × "
          f"{df.combo.nunique()} combos × "
          f"{df.year.nunique()} years)")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# New entry point: assemble panel from real AMR files
# ══════════════════════════════════════════════════════════════════════════════

def assemble_panel_from_files(
    amr_file_paths: list,
    years: list,
    country_codes: dict,
    wb_indicators: dict,
    conflict_strategy: str = 'mean',
    force_schemas: dict = None,
    filter_combos: list = None,
    filter_countries: list = None,
    aware_file_path: str = None,
    seed: int = 42,
    default_year: int = None,
) -> pd.DataFrame:
    """
    Assemble the full harmonised panel using REAL AMR surveillance files.

    This is the multi-file equivalent of assemble_panel(). It:
      1. Loads and harmonises all AMR files via multi_file_loader
      2. Derives the country list and combo list from the data itself
      3. Fills AWaRe consumption (from file if provided, else synthetic)
      4. Fills World Bank socioeconomic indicators (API with synthetic fallback)
      5. Forward-fills minor gaps and returns the canonical panel

    Parameters
    ----------
    amr_file_paths    : List of CSV/Excel AMR surveillance files.
    years             : Full year range to cover (historical + forecast).
    country_codes     : Dict mapping country name → ISO-2 code (for WB API).
    wb_indicators     : Dict mapping indicator name → World Bank code.
    conflict_strategy : 'mean' | 'median' | 'max' | 'first' | 'last'
    force_schemas     : Optional per-file schema overrides.
    filter_combos     : Keep only these pathogen-drug combos (None = all).
    filter_countries  : Keep only these countries (None = all found in files).
    aware_file_path   : Optional path to a WHO AWaRe CSV (country, year,
                        total_DDD, watch_proportion). Falls back to synthetic.
    seed              : Random seed for synthetic fallback data.
    default_year      : Required only if any input file uses a yearless
                        schema (currently: SPIDAAR patient-level
                        point-prevalence files, which have no internal
                        Year/Date column). Applied uniformly to every row
                        of such files.

    Returns
    -------
    Canonical panel DataFrame (same schema as assemble_panel output).
    """
    from .multi_file_loader import load_amr_files, ingestion_report

    # ── Step 1: Load and harmonise AMR files ─────────────────────────────────
    print("[data_loader] Loading real AMR surveillance files …")
    df_amr = load_amr_files(
        file_paths        = amr_file_paths,
        conflict_strategy = conflict_strategy,
        force_schemas     = force_schemas,
        filter_combos     = filter_combos,
        filter_countries  = filter_countries,
        year_range        = (min(years), max(years)),
        default_year      = default_year,
    )
    print(ingestion_report(df_amr))

    # Derive countries and combos from actual data
    countries_found = sorted(df_amr['country'].unique().tolist())
    combos_found    = sorted(df_amr['combo'].unique().tolist())
    print(f"\n[data_loader] Discovered {len(countries_found)} countries and "
          f"{len(combos_found)} pathogen-drug combos from files.")

    # ── Step 2: Expand panel — only for (country, combo) pairs with real data ──
    # Only expand years for pairs that actually appear in the file.
    # This avoids a massive sparse panel (e.g. 59 x 207 x 21 = 256k rows
    # with 94% empty) that wastes memory and produces meaningless forecasts
    # for pathogen-country pairs with zero surveillance data.
    actual_pairs = (df_amr[['country', 'combo']]
                    .drop_duplicates()
                    .values.tolist())
    print(f"  [data_loader] Expanding {len(actual_pairs)} observed "
          f"country-combo pairs across {len(years)} years ...")

    idx_rows = [(c, combo, yr)
                for c, combo in actual_pairs
                for yr in years]
    df_full  = pd.DataFrame(idx_rows, columns=['country', 'combo', 'year'])

    # Map combo back to pathogen + drug
    combo_to_pd = (df_amr[['combo', 'pathogen', 'drug']]
                   .drop_duplicates()
                   .set_index('combo'))
    df_full['pathogen'] = df_full['combo'].map(combo_to_pd['pathogen'])
    df_full['drug']     = df_full['combo'].map(combo_to_pd['drug'])

    # Merge actual resistance values
    df_amr_slim = df_amr[['country', 'combo', 'year',
                            'resistance_pct', 'n_isolates', 'source_files']]
    df_full = df_full.merge(df_amr_slim, on=['country', 'combo', 'year'], how='left')

    # ── Step 3: AWaRe consumption ─────────────────────────────────────────────
    if aware_file_path and os.path.exists(aware_file_path):
        print(f"[data_loader] Loading AWaRe data from: {aware_file_path}")
        df_aware = _load_aware_file(aware_file_path, countries_found, years)
    else:
        print("[data_loader] No AWaRe file supplied → using synthetic consumption data.")
        # For countries in file but not in AWARE_PARAMS, generate generic values
        df_aware = _synthetic_aware_flexible(years, countries_found, seed=seed)

    df_full = df_full.merge(df_aware, on=['country', 'year'], how='left')

    # ── Step 4: World Bank indicators ─────────────────────────────────────────
    print("[data_loader] Fetching World Bank socioeconomic indicators …")
    # Build country_codes for discovered countries.
    # Priority: (1) caller-supplied country_codes, (2) comprehensive built-in
    # _COUNTRY_NAME_TO_ISO2 map, (3) skip (no fallback to c[:2] which gives
    # invalid codes like 'KO' for 'Korea, South').
    cc_ext = {}
    no_iso = []
    for c in countries_found:
        if c in country_codes:
            cc_ext[c] = country_codes[c]
        elif c in _COUNTRY_NAME_TO_ISO2:
            cc_ext[c] = _COUNTRY_NAME_TO_ISO2[c]
        else:
            no_iso.append(c)
    if no_iso:
        print(f"  [WB] No ISO code for {no_iso} — synthetic WB values will be used.")
    df_wb  = load_worldbank_data(years, countries_found, cc_ext, wb_indicators)
    df_full = df_full.merge(df_wb, on=['country', 'year'], how='left')

    # ── Step 5: Interpolate gaps ──────────────────────────────────────────────
    wb_cols = list(WB_FALLBACK.get(countries_found[0] if countries_found else 'Germany', {}).keys())
    # Use the first country's WB_FALLBACK keys as reference, or all WB indicator names
    wb_col_candidates = list(wb_indicators.keys())
    df_full = df_full.sort_values(['country', 'combo', 'year'])

    for col in wb_col_candidates + ['total_DDD', 'watch_proportion']:
        if col in df_full.columns:
            df_full[col] = (df_full.groupby(['country', 'combo'])[col]
                            .transform(lambda x:
                                x.interpolate(method='linear', limit=2)
                                 .ffill().bfill()))

    df_full = df_full.reset_index(drop=True)
    print(f"[data_loader] Panel assembled: {len(df_full):,} rows "
          f"({df_full.country.nunique()} countries × "
          f"{df_full.combo.nunique()} combos × "
          f"{df_full.year.nunique()} years) "
          f"| Coverage: {df_full.resistance_pct.notna().mean()*100:.1f}% non-null")
    return df_full


# ── AWaRe file loader (real file path) ───────────────────────────────────────
def _load_aware_file(path: str, countries: list, years: list) -> pd.DataFrame:
    """
    Load a WHO AWaRe CSV/Excel.
    Expected columns: country, year, total_DDD, watch_proportion
    """
    if path.endswith('.csv'):
        raw = pd.read_csv(path)
    else:
        raw = pd.read_excel(path)

    col_country = None
    col_year    = None
    col_ddd     = None
    col_watch   = None

    for c in raw.columns:
        cl = c.lower().replace(' ', '_')
        if 'country' in cl:
            col_country = c
        elif 'year' in cl:
            col_year = c
        elif 'ddd' in cl or 'consumption' in cl:
            col_ddd = c
        elif 'watch' in cl or 'proportion' in cl:
            col_watch = c

    if not all([col_country, col_year, col_ddd, col_watch]):
        raise ValueError(
            f"AWaRe file '{path}' must have columns: country, year, "
            f"total_DDD (or 'consumption'), watch_proportion (or 'watch'). "
            f"Found: {list(raw.columns)}"
        )

    df = raw[[col_country, col_year, col_ddd, col_watch]].copy()
    df.columns = ['country', 'year', 'total_DDD', 'watch_proportion']
    df['country'] = df['country'].astype(str).str.strip()
    df['year']    = pd.to_numeric(df['year'], errors='coerce').astype('Int64')
    df['total_DDD']         = pd.to_numeric(df['total_DDD'], errors='coerce')
    df['watch_proportion']  = pd.to_numeric(df['watch_proportion'], errors='coerce')
    return df.dropna()


# ── Flexible synthetic AWaRe for arbitrary country lists ─────────────────────
_AWARE_DEFAULT = (14.0, 0.32, 0.20, 0.003)   # generic defaults

def _synthetic_aware_flexible(years: list, countries: list, seed: int = 42) -> pd.DataFrame:
    """Generate synthetic AWaRe data for any set of countries."""
    rng  = np.random.default_rng(seed + 1)
    rows = []
    for country in countries:
        params = AWARE_PARAMS.get(country, _AWARE_DEFAULT)
        base_ddd, base_watch, delta_ddd, delta_watch = params
        for year in years:
            d     = year - 2023
            ddd   = float(np.clip(base_ddd   + delta_ddd   * d + rng.normal(0, 0.4), 5, 40))
            watch = float(np.clip(base_watch + delta_watch * d + rng.normal(0, 0.01), 0.05, 0.70))
            rows.append({'country': country, 'year': year,
                         'total_DDD': round(ddd, 2),
                         'watch_proportion': round(watch, 4)})
    return pd.DataFrame(rows)
