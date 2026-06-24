#!/usr/bin/env python3
"""
fetch_worldbank_cache.py
─────────────────────────
Pre-fetches World Bank socioeconomic data and saves it to data/wb_cache.csv.
After this, the pipeline loads from cache instantly on every run.

Usage
-----
  # Default: fetch the 12 built-in pipeline countries
  python fetch_worldbank_cache.py

  # Fetch countries discovered from your actual data file(s)
  python fetch_worldbank_cache.py --from-file your_data.csv
  python fetch_worldbank_cache.py --from-file file1.csv file2.xlsx

  # Force refresh even if a valid cache exists
  python fetch_worldbank_cache.py --force
  python fetch_worldbank_cache.py --from-file data.csv --force
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))


def main():
    parser = argparse.ArgumentParser(
        description="Pre-fetch World Bank data cache for AMR pipeline.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument('--force', action='store_true',
                        help='Re-download even if a valid cache already exists.')
    parser.add_argument('--from-file', nargs='+', metavar='FILE', default=None,
                        help='Discover countries from your AMR data file(s) '
                             'and fetch WB data for all of them.')
    parser.add_argument('--countries', nargs='+', default=None,
                        help='Explicit country list (overrides --from-file).')
    args = parser.parse_args()

    from config import COUNTRIES, COUNTRY_CODES, WB_INDICATORS, ALL_YEARS
    from data.data_loader import (
        WB_CACHE_PATH, WB_CACHE_TTL_DAYS,
        _wb_fetch_all_indicators, _ISO3_TO_NAME, _ISO2_TO_ISO3,
        _COUNTRY_NAME_TO_ISO2,
    )
    import pandas as pd
    import numpy as np

    # ── Determine country list ────────────────────────────────────────────────
    if args.countries:
        countries = args.countries
        print(f"  Using explicit country list ({len(countries)} countries)")

    elif args.from_file:
        # Discover countries from actual data files
        print(f"  Discovering countries from {len(args.from_file)} file(s) ...")
        from data.multi_file_loader import load_amr_files
        try:
            df_amr = load_amr_files(args.from_file)
            countries = sorted(df_amr['country'].unique().tolist())
            print(f"  Found {len(countries)} countries in data files.")
        except Exception as e:
            print(f"  ERROR reading files: {e}")
            print("  Falling back to default 12 countries.")
            countries = COUNTRIES
    else:
        countries = COUNTRIES

    # ── Build ISO-2 map for discovered countries ──────────────────────────────
    # Merge built-in COUNTRY_CODES with comprehensive _COUNTRY_NAME_TO_ISO2 map
    cc_merged = {**{c: COUNTRY_CODES[c] for c in COUNTRY_CODES}, **_COUNTRY_NAME_TO_ISO2}
    known   = {c: cc_merged[c] for c in countries if c in cc_merged}
    unknown = [c for c in countries if c not in cc_merged]
    if unknown:
        print(f"  No ISO code for: {unknown} (will use synthetic WB values)")

    print("=" * 62)
    print("  World Bank Cache Pre-Fetcher")
    print("=" * 62)
    print(f"  Countries to fetch : {len(known)}")
    print(f"  Indicators         : {len(WB_INDICATORS)}")
    print(f"  Cache path         : {WB_CACHE_PATH}")
    print(f"  Cache TTL          : {WB_CACHE_TTL_DAYS} days")
    print(f"  Parallel workers   : 10")

    # ── Check existing cache ──────────────────────────────────────────────────
    if not args.force and os.path.exists(WB_CACHE_PATH):
        age_days = (time.time() - os.path.getmtime(WB_CACHE_PATH)) / 86400
        if age_days < WB_CACHE_TTL_DAYS:
            df_existing = pd.read_csv(WB_CACHE_PATH)
            cached_countries = set(df_existing['country'].unique())
            needed_countries = set(known.keys())
            truly_missing = needed_countries - cached_countries
            if not truly_missing:
                print(f"\n  Valid cache found ({age_days:.1f} days old, "
                      f"{len(df_existing):,} rows, "
                      f"{df_existing.country.nunique()} countries).")
                print("  All required countries are cached.")
                print("  Use --force to re-download.")
                return
            else:
                print(f"\n  Cache exists but missing {len(truly_missing)} "
                      f"countries: {sorted(truly_missing)}")
                print("  Fetching missing countries only ...")
                # Only fetch the missing ones
                known = {c: iso for c, iso in known.items()
                         if c in truly_missing}

    # ── Build name map ────────────────────────────────────────────────────────
    name_map = dict(_ISO3_TO_NAME)
    for name, iso2 in known.items():
        iso3 = _ISO2_TO_ISO3.get(iso2, iso2)
        if iso3 not in name_map:
            name_map[iso3] = name

    iso2_list  = list(known.values())
    hist_years = [y for y in ALL_YEARS if y <= 2023]

    # ── Fetch ─────────────────────────────────────────────────────────────────
    print(f"\n  Fetching {len(WB_INDICATORS)} indicators x "
          f"{len(iso2_list)} countries "
          f"({len(iso2_list) * len(WB_INDICATORS)} total calls) ...\n")
    t0  = time.time()
    raw = _wb_fetch_all_indicators(
        iso2_list, WB_INDICATORS,
        min(hist_years), max(hist_years),
        max_workers=10,
    )
    elapsed = time.time() - t0

    if raw.empty:
        print("\n  ERROR: API returned no data.")
        print("  Check your internet connection.")
        print("  Run 'python check_worldbank.py' to diagnose.")
        sys.exit(1)

    # ── Map ISO-3 -> country name ─────────────────────────────────────────────
    raw['country'] = raw['country_code'].map(name_map)
    raw = raw.dropna(subset=['country'])

    ind_cols = [c for c in raw.columns if c not in ('country_code','country','year')]
    cache_new = raw[['country', 'year'] + ind_cols].copy()

    # Add sentinels for countries with no WB data
    fetched_set     = set(cache_new['country'].unique())
    no_data         = [c for c in known.keys() if c not in fetched_set]
    if no_data:
        sentinel_rows = pd.DataFrame([
            {'country': c, 'year': hist_years[0]} for c in no_data
        ])
        for col in ind_cols:
            sentinel_rows[col] = np.nan
        cache_new = pd.concat([cache_new, sentinel_rows], ignore_index=True)
        print(f"\n  {len(no_data)} countries have no WB data (sentinel stored): "
              f"{no_data}")

    # ── Merge with existing cache if we only fetched missing countries ────────
    if os.path.exists(WB_CACHE_PATH) and not args.force:
        try:
            df_old = pd.read_csv(WB_CACHE_PATH)
            cache_new = pd.concat([df_old, cache_new], ignore_index=True)
            cache_new = cache_new.drop_duplicates(
                subset=['country', 'year'], keep='last'
            )
        except Exception:
            pass

    # ── Save ──────────────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(WB_CACHE_PATH) or '.', exist_ok=True)
    cache_new.to_csv(WB_CACHE_PATH, index=False)

    print(f"\n  SUCCESS in {elapsed:.1f}s")
    print(f"  Rows saved     : {len(cache_new):,}")
    print(f"  Countries      : {sorted(cache_new.country.unique())}")
    print(f"  Year range     : {cache_new.year.min()}-{cache_new.year.max()}")
    print(f"  Cache saved to : {WB_CACHE_PATH}")
    print()
    print("  The pipeline will now use this cache automatically.")
    print("=" * 62)


if __name__ == '__main__':
    main()
