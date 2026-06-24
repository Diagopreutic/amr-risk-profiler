"""
data/multi_file_loader.py
──────────────────────────────────────────────────────────────────────────────
Loads, validates, and harmonises ANY number of AMR surveillance files into the
single canonical panel used by the rest of the pipeline:

    country × pathogen_drug_combo × year  →  resistance_pct

Supported input formats
───────────────────────
• CSV  (.csv)
• Excel (.xlsx / .xls)

Supported schema families (auto-detected by column inspection)
──────────────────────────────────────────────────────────────
1. GLASS-style      WHO GLASS country-aggregated exports
                    Key cols: Country, Pathogen, Antibiotic[Class], Year,
                              Resistance[_pct | Percentage | Rate]
2. ATLAS-style      Pfizer ATLAS / similar isolate-level datasets
                    Key cols: Country, Organism[Species], Antibiotic[Agent],
                              Interpretation[SIR | Category], Year[Survey_Year]
3. EARS-Net-style   ECDC EARS-Net country/pathogen/antibiotic aggregates
                    Key cols: RegionName[CountryName], Bacteria[Microorganism],
                              Antibiotic[AntibioticGroup], Value[Percentage],
                              Time[Year]
4. Generic          Any file that has recognisable country / pathogen / drug /
                    year / resistance columns (fuzzy-matched)

Conflict resolution across files
─────────────────────────────────
When multiple files report the same (country, combo, year):
  • Default  → weighted mean by isolate count (if available), else simple mean
  • Strategy → configurable: 'mean' | 'median' | 'max' | 'first' | 'last'
  Priority order follows the list of files passed in (first = highest priority
  for 'first'/'last' strategies).

Isolate-level files (ATLAS-style)
──────────────────────────────────
Isolate rows with a SIR interpretation column are aggregated to:
    resistance_pct = 100 × (count R) / (count R + I + S)
before merging with already-aggregated sources.

Output
──────
A DataFrame with mandatory columns:
    country, pathogen, drug, combo, year, resistance_pct, n_isolates, source_files

Optional provenance columns retained if present:
    region, specimen_type, setting (hospital / community)
"""

import os
import re
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


# ══════════════════════════════════════════════════════════════════════════════
# Canonical vocabulary maps
# ══════════════════════════════════════════════════════════════════════════════

# Pathogen normalisation  (raw string → canonical key)
PATHOGEN_MAP: dict[str, str] = {
    # E. coli
    r'e[\.\s_-]*coli|escherichia[\s_-]*coli': 'E_coli',
    # K. pneumoniae
    r'k[\.\s_-]*pneumoniae|klebsiella[\s_-]*pneumoniae': 'K_pneumoniae',
    # S. aureus / MRSA
    r's[\.\s_-]*aureus|staphylococcus[\s_-]*aureus|mrsa': 'S_aureus',
    # A. baumannii
    r'a[\.\s_-]*baumannii|acinetobacter[\s_-]*baumannii': 'A_baumannii',
    r'acinetobacter[\s_-]+spp\.?$|acinetobacter[\s_-]+species': 'Acinetobacter_spp',
    r'acinetobacter[\s_-]+\w+': 'A_baumannii',
    # P. aeruginosa
    r'p[\.\s_-]*aeruginosa|pseudomonas[\s_-]*aeruginosa': 'P_aeruginosa',
    # Enterococcus
    r'e[\.\s_-]*faecium|enterococcus[\s_-]*faecium': 'E_faecium',
    r'e[\.\s_-]*faecalis|enterococcus[\s_-]*faecalis': 'E_faecalis',
    # Salmonella
    r'salmonella[\s_-]*typhi': 'S_typhi',
    r'non-typhoidal[\s_-]*salmonella|nts|salmonella[\s_-]*spp': 'Salmonella_NTS',
    # Streptococcus
    r's[\.\s_-]*pneumoniae|streptococcus[\s_-]*pneumoniae': 'S_pneumoniae',
    # Neisseria
    r'n[\.\s_-]*gonorrhoeae|neisseria[\s_-]*gonorrhoeae': 'N_gonorrhoeae',
    # Enterobacterales / Serratia genus (all species collapse to genus level)
    r'serratia[\s_-]*marcescens': 'S_marcescens',
    r'serratia[\s_-]+\w+': 'S_marcescens',
    r'enterobacter[\s_-]*cloacae': 'E_cloacae',
    r'enterobacter[\s_-]*aerogenes|klebsiella[\s_-]*aerogenes': 'E_aerogenes',
    r'enterobacter[\s_-]+\w+': 'E_cloacae',
    # Proteus / Providencia / Morganella
    r'proteus[\s_-]*mirabilis': 'P_mirabilis',
    r'proteus[\s_-]+\w+': 'P_mirabilis',
    r'morganella[\s_-]*morganii': 'M_morganii',
    r'providencia[\s_-]+\w+': 'Providencia_spp',
    # Citrobacter
    r'citrobacter[\s_-]*freundii': 'C_freundii',
    r'citrobacter[\s_-]+\w+': 'C_freundii',
    # Haemophilus
    r'haemophilus[\s_-]*influenzae': 'H_influenzae',
    # Stenotrophomonas
    r'stenotrophomonas[\s_-]*maltophilia': 'S_maltophilia',
    # Burkholderia
    r'burkholderia[\s_-]*cepacia': 'B_cepacia',
}

# Drug / antibiotic class normalisation  (raw → canonical key)
DRUG_MAP: dict[str, str] = {
    # 3rd-gen cephalosporins
    r'3rd[\s_-]*gen[\s_-]*ceph|3gc|third[\s_-]*gen|cephalosporin[\s_-]*(3|iii|third)'
    r'|ceftriaxone|cefotaxime|ceftazidime|cefixime': '3GC',
    # Carbapenems
    r'carbapenem|imipenem|meropenem|ertapenem|doripenem|carb': 'CARB',
    # Fluoroquinolones
    r'fluoroquinolone|quinolone|ciprofloxacin|levofloxacin|ofloxacin|fq': 'FQ',
    # Methicillin / MRSA
    r'methicillin|oxacillin|mrsa': 'MRSA',
    # Aminoglycosides
    r'aminoglycoside|gentamicin|amikacin|tobramycin': 'AMG',
    # Colistin / polymyxin
    r'colistin|polymyxin': 'COL',
    # Vancomycin
    r'vancomycin|glycopeptide': 'VAN',
    # Ampicillin / penicillins
    r'ampicillin|amoxicillin|penicillin': 'AMP',
    # Trimethoprim-sulfamethoxazole
    r'trimethoprim|cotrimoxazole|tmp[\s_-]*smx|tmp': 'TMP_SMX',
    # Tetracyclines
    r'tetracycline|doxycycline|minocycline': 'TET',
    # Azithromycin / macrolides
    r'azithromycin|macrolide|erythromycin': 'MAC',
    # Extended-spectrum beta-lactamases (ESBL phenotype)
    r'esbl': '3GC',
}

# SIR interpretation → binary non-susceptible flag
SIR_RESISTANT = {'R', 'r', 'resistant', 'Resistant', 'I', 'i',
                 'intermediate', 'Intermediate', 'NS', 'Non-susceptible'}
SIR_SUSCEPTIBLE = {'S', 's', 'susceptible', 'Susceptible'}

# Country name aliases → canonical names used in config.py
COUNTRY_ALIASES: dict[str, str] = {
    'usa': 'United States', 'united states of america': 'United States',
    'us': 'United States', 'u.s.': 'United States', 'u.s.a.': 'United States',
    'uk': 'United Kingdom', 'great britain': 'United Kingdom',
    'england': 'United Kingdom', 'gb': 'United Kingdom',
    'russian federation': 'Russia', 'russian fed.': 'Russia',
    'korea, rep.': 'South Korea', 'republic of korea': 'South Korea',
    'south korea': 'South Korea',
    'viet nam': 'Vietnam', 'vietnam': 'Vietnam',
    'iran, islamic rep.': 'Iran', 'islamic republic of iran': 'Iran',
    'tanzania': 'Tanzania', 'united republic of tanzania': 'Tanzania',
    "cote d'ivoire": 'Ivory Coast', 'côte d\'ivoire': 'Ivory Coast',
    'democratic republic of the congo': 'DR Congo', 'drc': 'DR Congo',
    'republic of south africa': 'South Africa',
}





# ══════════════════════════════════════════════════════════════════════════════
# ATLAS-SIR format  (DrugName / DrugName_I column pairs)
# ══════════════════════════════════════════════════════════════════════════════

DRUG_FULLNAME_MAP: dict[str, str] = {
    # 3rd-gen cephalosporins
    'ceftriaxone': '3GC', 'cefotaxime': '3GC', 'ceftazidime': '3GC',
    'cefixime': '3GC', 'cefpodoxime': '3GC', 'ceftibuten': '3GC',
    'cefepime': '3GC', 'cefoperazone sulbactam': '3GC',
    'cefoperazone/sulbactam': '3GC', 'cefiderocol': '3GC',
    'ceftazidime avibactam': '3GC', 'ceftazidime/avibactam': '3GC',
    'ceftolozane tazobactam': '3GC', 'ceftolozane/tazobactam': '3GC',
    'aztreonam avibactam': '3GC', 'aztreonam/avibactam': '3GC',
    # Carbapenems
    'imipenem': 'CARB', 'meropenem': 'CARB', 'ertapenem': 'CARB',
    'doripenem': 'CARB',
    'meropenem vaborbactam': 'CARB', 'meropenem/vaborbactam': 'CARB',
    # Fluoroquinolones
    'ciprofloxacin': 'FQ', 'levofloxacin': 'FQ', 'moxifloxacin': 'FQ',
    'ofloxacin': 'FQ', 'gatifloxacin': 'FQ', 'norfloxacin': 'FQ',
    # Methicillin / MRSA marker
    'oxacillin': 'MRSA', 'methicillin': 'MRSA',
    # Aminoglycosides
    'gentamicin': 'AMG', 'amikacin': 'AMG', 'tobramycin': 'AMG',
    'netilmicin': 'AMG',
    # Penicillins / beta-lactams
    'ampicillin': 'AMP', 'amoxicillin': 'AMP', 'amoxycillin': 'AMP',
    'amoxycillin clavulanate': 'AMP', 'amoxicillin clavulanate': 'AMP',
    'amoxycillin/clavulanate': 'AMP', 'amoxicillin/clavulanate': 'AMP',
    'ampicillin sulbactam': 'AMP', 'ampicillin/sulbactam': 'AMP',
    'piperacillin tazobactam': 'AMP', 'piperacillin/tazobactam': 'AMP',
    'sulbactam': 'AMP', 'penicillin': 'AMP',
    # Colistin / polymyxins
    'colistin': 'COL', 'polymyxin b': 'COL', 'polymyxin': 'COL',
    # Glycopeptides
    'vancomycin': 'VAN', 'teicoplanin': 'VAN',
    # TMP-SMX
    'trimethoprim sulfa': 'TMP_SMX', 'trimethoprim/sulfa': 'TMP_SMX',
    'trimethoprim sulfamethoxazole': 'TMP_SMX',
    'trimethoprim/sulfamethoxazole': 'TMP_SMX', 'cotrimoxazole': 'TMP_SMX',
    'trimethoprim': 'TMP_SMX',
    # Tetracyclines
    'tetracycline': 'TET', 'doxycycline': 'TET', 'minocycline': 'TET',
    'tigecycline': 'TET',
    # Macrolides
    'azithromycin': 'MAC', 'clarithromycin': 'MAC', 'erythromycin': 'MAC',
    # Other
    'linezolid': 'LZD', 'daptomycin': 'DAP',
    'metronidazole': 'OTHER', 'clindamycin': 'OTHER',
    'quinupristin dalfopristin': 'OTHER', 'aztreonam': 'OTHER',
    'cefoxitin': 'OTHER', 'ceftaroline': 'OTHER',
}

_SIR_RESISTANT_VALUES = {
    'resistant', 'r', 'intermediate', 'i', 'non-susceptible', 'ns',
}


def _detect_sir_columns(df: pd.DataFrame) -> list:
    """
    Scan for DrugName_I interpretation columns whose drug name maps
    to a known canonical code (excluding OTHER).
    Returns list of (drug_name, interp_col, canonical_code) triples.
    """
    result = []
    for col in df.columns:
        if not col.endswith('_I'):
            continue
        drug_name = col[:-2].strip()
        canonical = DRUG_FULLNAME_MAP.get(drug_name.lower().strip())
        if canonical and canonical != 'OTHER':
            result.append((drug_name, col, canonical))
    return result


def _parse_atlas_sir(df: pd.DataFrame, source_name: str) -> pd.DataFrame:
    """
    Parse ATLAS/VIVLI files where each drug has DrugName and DrugName_I columns.
    Uses the _I (pre-computed SIR) interpretation column as the primary
    resistance indicator — more reliable than raw MIC + manual breakpoints.
    Aggregates to country x pathogen x drug x year -> resistance_pct.
    """
    country_col  = _find_col(df, ['Country', 'CountryName', 'Nation'])
    pathogen_col = _find_col(df, ['Species', 'Organism', 'Pathogen',
                                   'Bacteria', 'Microorganism'])
    year_col     = _find_col(df, ['Year', 'SurveyYear', 'CollectionYear'])

    missing = [n for n, c in [('Country', country_col),
                               ('Species/Organism', pathogen_col),
                               ('Year', year_col)] if c is None]
    if missing:
        raise ValueError(
            f"[ATLAS-SIR parser] Cannot find columns {missing} "
            f"in '{source_name}'.\n"
            f"Columns present: {list(df.columns[:12])} ..."
        )

    sir_pairs = _detect_sir_columns(df)
    if not sir_pairs:
        raise ValueError(
            f"[ATLAS-SIR parser] No interpretable DrugName_I columns found "
            f"in '{source_name}'.\n"
            f"_I columns present: "
            f"{[c for c in df.columns if c.endswith('_I')][:8]}"
        )

    print(f"    SIR pairs : {len(sir_pairs)} drug-interpretation pairs detected")

    rows_all = []
    for drug_name, interp_col, canonical in sir_pairs:
        sub = df[[country_col, pathogen_col, year_col, interp_col]].copy()
        sub.columns = ['country', 'pathogen', 'year', 'sir']
        sub = sub[sub['sir'].notna() &
                  (sub['sir'].astype(str).str.strip() != '')]
        if sub.empty:
            continue
        sub['drug'] = canonical
        sub['non_susceptible'] = (
            sub['sir'].astype(str).str.strip().str.lower()
            .isin(_SIR_RESISTANT_VALUES)
        ).astype(int)
        rows_all.append(sub[['country','pathogen','drug','year','non_susceptible']])

    if not rows_all:
        raise ValueError(
            f"[ATLAS-SIR parser] All _I columns were empty in '{source_name}'."
        )

    long_df = pd.concat(rows_all, ignore_index=True)
    agg = (long_df
           .groupby(['country', 'pathogen', 'drug', 'year'])
           .agg(resistance_pct=('non_susceptible', lambda x: x.mean() * 100),
                n_isolates=('non_susceptible', 'count'))
           .reset_index())
    return agg


# ══════════════════════════════════════════════════════════════════════════════
# MIC-format support
# ══════════════════════════════════════════════════════════════════════════════

# Map of MIC column abbreviations (as they appear in file headers) to
# (canonical_drug_key, EUCAST_R_breakpoint_mg_L)
# Breakpoint = lowest MIC value classified as Resistant (i.e. MIC > breakpoint → R)
# Sources: EUCAST 2024 clinical breakpoints v14.0; CLSI M100-Ed34 where EUCAST absent.
# Format:  col_suffix → (canonical_drug, R_breakpoint_mg_L)
MIC_COL_MAP: dict[str, tuple[str, float]] = {
    # Cephalosporins
    'CAZ':   ('3GC',   8.0),   # Ceftazidime          EUCAST Enterobacterales R>8
    'CTX':   ('3GC',   2.0),   # Cefotaxime           EUCAST R>2
    'CRO':   ('3GC',   2.0),   # Ceftriaxone          EUCAST R>2
    'FEP':   ('3GC',   8.0),   # Cefepime             EUCAST R>8
    'CFP':   ('3GC',   8.0),   # Cefoperazone         CLSI R>32 (use 8 conserv.)
    'CTT':   ('3GC',   8.0),   # Ceftazidime/other
    # Carbapenems
    'IPM':   ('CARB',  8.0),   # Imipenem             EUCAST R>8
    'MEM':   ('CARB',  8.0),   # Meropenem            EUCAST R>8
    'ETP':   ('CARB',  1.0),   # Ertapenem            EUCAST R>1
    'DOR':   ('CARB',  2.0),   # Doripenem            EUCAST R>2
    'IMP':   ('CARB',  8.0),   # Imipenem alt abbrev
    # Fluoroquinolones
    'CIP':   ('FQ',    0.5),   # Ciprofloxacin        EUCAST R>0.5
    'LVX':   ('FQ',    1.0),   # Levofloxacin         EUCAST R>1
    'OFX':   ('FQ',    1.0),   # Ofloxacin            EUCAST R>1
    'NOR':   ('FQ',    0.5),   # Norfloxacin          CLSI R>4 (use 0.5 conserv.)
    'MOX':   ('FQ',    0.5),   # Moxifloxacin
    # Methicillin / MRSA
    'OXA':   ('MRSA',  2.0),   # Oxacillin            EUCAST S.aureus R>2
    'MET':   ('MRSA',  4.0),   # Methicillin
    # Aminoglycosides
    'GM':    ('AMG',   4.0),   # Gentamicin           EUCAST R>4
    'TOB':   ('AMG',   4.0),   # Tobramycin           EUCAST R>4
    'AMK':   ('AMG',  16.0),   # Amikacin             EUCAST R>16
    'NET':   ('AMG',   8.0),   # Netilmicin
    # Colistin / polymyxins
    'CL':    ('COL',   2.0),   # Colistin             EUCAST R>2
    'COL':   ('COL',   2.0),   # Colistin alt
    'PMB':   ('COL',   2.0),   # Polymyxin B
    # Glycopeptides / vancomycin
    'VAN':   ('VAN',   2.0),   # Vancomycin           EUCAST S.aureus R>2
    'TEC':   ('VAN',   2.0),   # Teicoplanin
    # Penicillins / beta-lactams
    'AMP':   ('AMP',   8.0),   # Ampicillin           EUCAST R>8
    'AMX':   ('AMP',   8.0),   # Amoxicillin
    'TZP':   ('AMP',  16.0),   # Piperacillin-tazobactam EUCAST R>16
    'TIM':   ('AMP',  16.0),   # Ticarcillin-clavulanate
    'PIP':   ('AMP',  16.0),   # Piperacillin
    # Tetracyclines
    'TET':   ('TET',   8.0),   # Tetracycline         EUCAST R>8
    'MIN':   ('TET',   4.0),   # Minocycline          EUCAST R>4
    'MI':    ('TET',   4.0),   # Minocycline alt
    'DOX':   ('TET',   4.0),   # Doxycycline
    # Trimethoprim / sulfonamides
    'SXT':   ('TMP_SMX', 4.0), # TMP-SMX              EUCAST R>4
    'TMP':   ('TMP_SMX', 4.0), # Trimethoprim
    'SMX':   ('TMP_SMX', 4.0), # Sulfamethoxazole
    # Macrolides
    'ERY':   ('MAC',   1.0),   # Erythromycin
    'AZI':   ('MAC',   2.0),   # Azithromycin
    'CLR':   ('MAC',   1.0),   # Clarithromycin
    # Ceftazidime-avibactam (novel)
    'CZAV':  ('3GC',   8.0),   # Ceftazidime-avibactam
    'CZA':   ('3GC',   8.0),
    # Chloramphenicol
    'C':     ('OTHER', 8.0),   # Chloramphenicol — mapped to OTHER
    'CHL':   ('OTHER', 8.0),
    # Rifampicin
    'RIF':   ('OTHER', 0.5),
    # Linezolid
    'LZD':   ('OTHER', 4.0),
    # Fosfomycin
    'FOS':   ('OTHER',32.0),
}

# ── Full drug-name MIC breakpoints (no abbreviation/suffix required) ──────────
# Covers files where columns are named directly after the drug
# (e.g. 'Cefiderocol', 'Ceftazidime/ Avibactam', 'Polymyxin B MIC (mcg/ml)')
# rather than using an abbreviation + '_MIC' suffix.
# Keys are normalised via _normalise_drug_colname() before lookup.
FULL_NAME_MIC_BREAKPOINTS: dict[str, tuple[str, float]] = {
    # Single agents
    'cefiderocol':              ('3GC',  2.0),
    'meropenem':                ('CARB', 8.0),
    'imipenem':                 ('CARB', 8.0),
    'doripenem':                ('CARB', 2.0),
    'ertapenem':                ('CARB', 1.0),
    'ciprofloxacin':            ('FQ',   0.5),
    'levofloxacin':             ('FQ',   1.0),
    'colistin':                 ('COL',  2.0),
    'polymyxin b':              ('COL',  2.0),
    'cefepime':                 ('3GC',  8.0),
    'ceftazidime':              ('3GC',  8.0),
    'ceftriaxone':              ('3GC',  2.0),
    'minocycline':              ('TET',  4.0),
    'tigecycline':              ('TET',  0.5),
    'tetracycline':             ('TET',  8.0),
    'gentamicin':               ('AMG',  4.0),
    'amikacin':                 ('AMG', 16.0),
    'tobramycin':               ('AMG',  4.0),
    'ampicillin':                ('AMP',  8.0),
    'vancomycin':                ('VAN',  2.0),
    'azithromycin':              ('MAC',  2.0),
    'aztreonam':                  ('OTHER', 16.0),
    # Beta-lactam / beta-lactamase-inhibitor combinations
    'ampicillin sulbactam':       ('AMP',  8.0),
    'piperacillin tazobactam':    ('AMP', 16.0),
    'meropenem vaborbactam':      ('CARB', 8.0),
    'imipenem relebactam':        ('CARB', 2.0),
    'ceftazidime avibactam':      ('3GC',  8.0),
    'aztreonam avibactam':        ('3GC',  4.0),
    'ceftolozane tazobactam':     ('3GC',  4.0),
    'cefoperazone sulbactam':     ('3GC',  8.0),
    'trimethoprim sulfamethoxazole': ('TMP_SMX', 4.0),
    'trimethoprim sulfa':         ('TMP_SMX', 4.0),
}


def _normalise_drug_colname(col: str) -> str:
    """
    Normalise a column header down to a bare drug name for matching
    against FULL_NAME_MIC_BREAKPOINTS.

    Handles real-world header noise seen in surveillance exports:
      'Meropenem/ Vaborbactam at 8'  -> 'meropenem vaborbactam'
      'Polymyxin B MIC (mcg/ml)'     -> 'polymyxin b'
      'Ceftazidime/ Avibactam'       -> 'ceftazidime avibactam'
      'Ampicillin/ Sulbactam'        -> 'ampicillin sulbactam'
    """
    s = col.strip()
    # Strip trailing "at <number>" (fixed inhibitor concentration, not a breakpoint)
    s = re.sub(r'\s+at\s+\d+(\.\d+)?\s*$', '', s, flags=re.IGNORECASE)
    # Strip trailing parenthetical units, e.g. "(mcg/ml)", "(mg/L)"
    s = re.sub(r'\s*\([^)]*\)\s*$', '', s)
    # Strip trailing/leading "MIC" or "_MIC" token
    s = re.sub(r'(^|\s|_)MIC(\s|_|$)', ' ', s, flags=re.IGNORECASE)
    # Replace slash-separated combo names with a space
    s = s.replace('/', ' ')
    # Collapse whitespace and lowercase
    s = re.sub(r'\s+', ' ', s).strip().lower()
    return s


def _detect_mic_columns_by_fullname(df: pd.DataFrame) -> dict:
    """
    Scan columns for bare full drug names (no abbreviation, no _MIC suffix
    required) by normalising headers and matching against
    FULL_NAME_MIC_BREAKPOINTS. More specific than abbreviation matching,
    so a single match is trusted enough to count toward schema detection.
    """
    found = {}
    for col in df.columns:
        key = _normalise_drug_colname(col)
        if key in FULL_NAME_MIC_BREAKPOINTS:
            found[col] = FULL_NAME_MIC_BREAKPOINTS[key]
    return found



def _detect_mic_columns(df: pd.DataFrame) -> dict[str, tuple[str, float]]:
    """
    Scan df columns for MIC-format headers using TWO complementary methods:

    1. Abbreviation + suffix matching (e.g. 'CAZ_MIC', 'MEM_MIC') —
       requires a known abbreviation from MIC_COL_MAP.
    2. Full drug-name matching (e.g. 'Cefiderocol', 'Ceftazidime/ Avibactam',
       'Polymyxin B MIC (mcg/ml)') — requires a normalised match against
       FULL_NAME_MIC_BREAKPOINTS. This covers files where columns are named
       directly after the drug with no abbreviation or suffix convention.

    Returns {col_name: (canonical_drug, breakpoint)} for every match found,
    merging both methods (full-name matches take priority on overlap since
    they are based on a complete, unambiguous drug name).
    """
    found = {}
    # Method 1: abbreviation + suffix
    for col in df.columns:
        cu = col.strip().upper()
        abbrev = cu.replace('_MIC', '').replace('MIC', '').strip('_').strip()
        if abbrev in MIC_COL_MAP and ('MIC' in cu or abbrev == cu):
            found[col] = MIC_COL_MAP[abbrev]
        elif cu in MIC_COL_MAP:
            found[col] = MIC_COL_MAP[cu]

    # Method 2: bare full drug name (overrides method 1 on overlap)
    fullname_matches = _detect_mic_columns_by_fullname(df)
    found.update(fullname_matches)

    return found


def _parse_mic_value(val) -> float:
    """
    Parse a MIC cell which may be:
      - plain float/int: 0.25, 4, 32
      - string with operator: <=0.25, >32, >=8, =2
      - string with dash: 0.25-0.5 (take upper bound)
      - NaN / empty
    Returns float or np.nan.
    """
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return np.nan
    s = str(val).strip().replace(' ', '')
    if s.upper() in ('', '-', 'NA', 'N/A', 'NAN', '.', 'NULL', 'NONE'):
        return np.nan
    # Remove operators, take the numeric part
    # For <=X or <X → use X (conservative: border cases scored at the value)
    # For >=X or >X → use X
    s_clean = re.sub(r'^[><=]+', '', s)
    # Range: take upper bound
    if '-' in s_clean:
        parts = s_clean.split('-')
        s_clean = parts[-1]
    try:
        return float(s_clean)
    except ValueError:
        return np.nan


def _parse_mic(df: pd.DataFrame, source_name: str) -> pd.DataFrame:
    """
    Parse a wide-format MIC file where each row is one isolate and
    each drug has its own MIC column (e.g. CAZ_MIC, MEM_MIC, CIP_MIC).

    Steps
    -----
    1. Identify country / organism / year columns.
    2. Detect all MIC columns and their EUCAST breakpoints.
    3. For each MIC column: classify each isolate as R (MIC > breakpoint) or S.
    4. Melt to long format: isolate x drug → R/S flag.
    5. Aggregate to country x pathogen x drug x year -> resistance_pct.
    """
    # ── Mandatory identity columns ────────────────────────────────────────
    country_col  = _find_col(df, ['Country', 'CountryName', 'Nation', 'Site'])
    pathogen_col = _find_col(df, ['Organism', 'Species', 'Pathogen',
                                   'Bacteria', 'Microorganism', 'Organism_Name'])
    year_col     = _find_col(df, ['Year', 'SurveyYear', 'CollectionYear',
                                   'IsolateYear', 'StudyYear', 'Date'])

    missing = [name for name, col in [
        ('Country', country_col),
        ('Organism/Pathogen', pathogen_col),
        ('Year', year_col),
    ] if col is None]
    if missing:
        raise ValueError(
            f"[MIC parser] Cannot find mandatory columns {missing} "
            f"in '{source_name}'.\nColumns present: {list(df.columns)}"
        )

    # ── MIC columns ───────────────────────────────────────────────────────
    mic_map = _detect_mic_columns(df)
    if not mic_map:
        raise ValueError(
            f"[MIC parser] No MIC columns detected in '{source_name}'.\n"
            f"Expected columns like CAZ_MIC, MEM_MIC, CIP_MIC.\n"
            f"Columns present: {list(df.columns)}"
        )

    print(f"    MIC cols : {len(mic_map)} detected → "
          f"{list(mic_map.keys())[:8]}{'...' if len(mic_map)>8 else ''}")

    # ── Build isolate-level binary resistance flags ───────────────────────
    rows = []
    id_cols = [country_col, pathogen_col, year_col]
    sub = df[id_cols + list(mic_map.keys())].copy()

    for mic_col, (drug_canonical, breakpoint) in mic_map.items():
        col_data = sub[[country_col, pathogen_col, year_col, mic_col]].copy()
        col_data.columns = ['country', 'pathogen', 'year', 'mic_raw']
        col_data['mic_val'] = col_data['mic_raw'].apply(_parse_mic_value)
        col_data = col_data.dropna(subset=['mic_val'])
        col_data['drug'] = drug_canonical
        # Classify: MIC > breakpoint → Resistant (1), else Susceptible (0)
        col_data['resistant'] = (col_data['mic_val'] > breakpoint).astype(int)
        rows.append(col_data[['country', 'pathogen', 'drug', 'year', 'resistant']])

    if not rows:
        raise ValueError(f"[MIC parser] All MIC columns were empty in '{source_name}'.")

    long_df = pd.concat(rows, ignore_index=True)

    # ── Aggregate to country x pathogen x drug x year ────────────────────
    agg = (long_df
           .groupby(['country', 'pathogen', 'drug', 'year'])
           .agg(resistance_pct=('resistant', lambda x: x.mean() * 100),
                n_isolates=('resistant', 'count'))
           .reset_index())

    return agg



# ══════════════════════════════════════════════════════════════════════════════
# SPIDAAR format  (patient-level point-prevalence study, 3GC-R endpoint)
# ══════════════════════════════════════════════════════════════════════════════
#
# Evidence trail (from the SPIDAAR codebook workbook, all 7 sheets reviewed):
#   - 'amrp' codebook label: "Drug susceptibility result to administered
#      AB class, patient level" with codes -1/0/1/2.
#   - The 'definitions' sheet's ONLY antimicrobial-resistance phenotype
#      definition in the entire workbook is "3GC-R: Resistance to
#      third-generation cephalosporins", with documented edge-case rules
#      (MRSA-oxacillin-resistant -> classified 3GC-R; intrinsic resistance
#      assumed for Enterococcus spp. and Enterobacter spp.). These rules are
#      specific to 3GC biology, not generic across antibiotic classes --
#      strong evidence that 'amrp' operationalises 3GC-R for this study.
#   - 'isol' lists one or more organism names per patient (comma-separated
#      for polymicrobial cases); 'amrp' is the PATIENT-level resistance
#      verdict (not isolate-specific), so it is applied to every listed
#      organism for that patient as an approximation.
#   - No Year/Date column exists anywhere in the 60-column data sheet, and
#      no study enrolment period is stated in any of the other 6 sheets
#      (the only 20XX-pattern matches found are unrelated journal citation
#      years for HAI diagnostic guideline references). The dataset is a
#      cross-sectional point-prevalence study with no internal time axis.
#      A year must therefore be supplied externally (see SPIDAAR_DEFAULT_YEAR
#      / the --start-year mechanism) and is applied uniformly to all rows.

SPIDAAR_AMRP_RESISTANT = 2
SPIDAAR_AMRP_SUSCEPTIBLE = 0
# amrp == -1 (no RX result) and amrp == 1 (mixed/untested) are excluded as
# ambiguous -- consistent with how the codebook describes them.


def _is_spidaar_format(df: pd.DataFrame) -> bool:
    """
    Detect the SPIDAAR patient-level schema by its distinctive column
    signature: pid, ctry, micp, isol, amrp all present together. This
    combination does not occur in GLASS/ATLAS/EARS-Net/MIC exports.
    """
    cols = {c.strip().lower() for c in df.columns}
    required = {'pid', 'ctry', 'micp', 'isol', 'amrp'}
    return required.issubset(cols)


def _parse_spidaar(df: pd.DataFrame, source_name: str,
                   default_year: int = None) -> pd.DataFrame:
    """
    Parse a SPIDAAR patient-level point-prevalence file into the standard
    country x pathogen x drug x year -> resistance_pct format.

    Resistance endpoint: 3GC-R (third-generation cephalosporin resistance),
    derived from the 'amrp' patient-level variable -- see the module-level
    comment block above for the supporting evidence.

    Parameters
    ----------
    default_year : Required. SPIDAAR has no internal year/date column, so
                   every row is assigned this single user-supplied year.
                   Raises ValueError if not provided.
    """
    if default_year is None:
        raise ValueError(
            "[SPIDAAR parser] This file has no Year/Date column anywhere "
            "(confirmed across all sheets of the source codebook), and no "
            "study enrolment period is documented. A year must be supplied "
            "explicitly via --start-year (or the interactive prompt) so "
            "every row can be assigned to a single calendar year."
        )

    col_map = {c.strip().lower(): c for c in df.columns}
    country_col = col_map.get('ctry')
    isol_col    = col_map.get('isol')
    amrp_col    = col_map.get('amrp')

    sub = df[[country_col, isol_col, amrp_col]].copy()
    sub.columns = ['country', 'isol', 'amrp']

    # Keep only unambiguous resistant/susceptible verdicts
    sub = sub[sub['amrp'].isin([SPIDAAR_AMRP_RESISTANT, SPIDAAR_AMRP_SUSCEPTIBLE])]
    sub = sub.dropna(subset=['isol', 'country'])

    if sub.empty:
        raise ValueError(
            f"[SPIDAAR parser] No usable rows in '{source_name}' -- all "
            f"patients had amrp = -1 (no RX result) or 1 (mixed/untested), "
            f"or no isolate/country recorded."
        )

    # Expand polymicrobial patients: one row per listed organism, applying
    # the patient-level amrp verdict to each (documented approximation).
    rows = []
    for _, row in sub.iterrows():
        organisms = [o.strip() for o in str(row['isol']).split(',') if o.strip()]
        is_resistant = (row['amrp'] == SPIDAAR_AMRP_RESISTANT)
        for org in organisms:
            rows.append({
                'country':        str(row['country']).strip(),
                'pathogen_raw':   org,
                'drug':           '3GC',
                'year':           default_year,
                'resistance_pct': 100.0 if is_resistant else 0.0,
            })

    out = pd.DataFrame(rows)
    out['pathogen'] = out['pathogen_raw'].apply(_normalise_pathogen)

    agg = (out.groupby(['country', 'pathogen', 'drug', 'year'])
              .agg(resistance_pct=('resistance_pct', 'mean'),
                   n_isolates=('resistance_pct', 'count'))
              .reset_index())
    return agg


# ══════════════════════════════════════════════════════════════════════════════
# Schema detection helpers
# ══════════════════════════════════════════════════════════════════════════════

def _normalise_colname(col: str) -> str:
    """Lowercase, strip, replace spaces/dashes with underscores."""
    return re.sub(r'[\s\-]+', '_', col.strip().lower())


def _detect_schema(df: pd.DataFrame, filepath: str) -> str:
    """
    Inspect column names and return a schema family tag:
    'glass' | 'atlas' | 'earsnet' | 'mic' | 'generic'
    """
    cols = {_normalise_colname(c) for c in df.columns}

    glass_signals   = {'resistance_percentage', 'resistance_pct', 'resistance_rate',
                       'percentage_resistant', 'pct_resistant'}
    atlas_signals   = {'interpretation', 'sir', 'category', 'survey_year', 'species'}
    earsnet_signals = {'regionname', 'countryname', 'bacteria', 'microorganism',
                       'antibioticgroup', 'percentage', 'time'}

    fname = Path(filepath).name.lower()

    if _is_spidaar_format(df):
        return 'spidaar'

    # MIC detection: check for MIC-format columns before atlas check
    # (MIC files often have 'Organism'/'Species' cols which would trigger
    # atlas falsely if checked first).
    #
    # Two-tier confidence threshold:
    #   - Full drug-name matches (e.g. 'Cefiderocol', 'Polymyxin B MIC')
    #     are unambiguous identifications of a specific antibiotic, so a
    #     SINGLE match is sufficient (covers single-drug surveillance files
    #     such as a colistin-only or polymyxin-only dataset).
    #   - Abbreviation-only matches (e.g. a bare column 'GM') are less
    #     specific and could coincide with unrelated short codes, so at
    #     least 2 are required to avoid false-positive schema detection.
    fullname_matches = _detect_mic_columns_by_fullname(df)
    mic_cols_found   = _detect_mic_columns(df)
    has_mic_cols     = (len(fullname_matches) >= 1) or (len(mic_cols_found) >= 2)
    sir_pairs_found  = _detect_sir_columns(df)
    has_sir_pairs    = len(sir_pairs_found) >= 2

    if any(s in cols for s in glass_signals) or 'glass' in fname:
        return 'glass'
    if has_mic_cols:
        return 'mic'
    if has_sir_pairs:
        return 'atlas_sir'
    if any(s in cols for s in atlas_signals) or 'atlas' in fname:
        return 'atlas'
    if any(s in cols for s in earsnet_signals) or 'ears' in fname or 'ecdc' in fname:
        return 'earsnet'
    return 'generic'


def _find_col(df: pd.DataFrame, candidates: list[str]) -> Optional[str]:
    """Return the first df column that fuzzy-matches any candidate string."""
    norm_map = {_normalise_colname(c): c for c in df.columns}
    for cand in candidates:
        c = _normalise_colname(cand)
        if c in norm_map:
            return norm_map[c]
        # partial match
        for k, v in norm_map.items():
            if c in k or k in c:
                return v
    return None


# ══════════════════════════════════════════════════════════════════════════════
# Pathogen & drug normalisation
# ══════════════════════════════════════════════════════════════════════════════

def _normalise_pathogen(raw: str) -> str:
    s = str(raw).strip()
    for pattern, canonical in PATHOGEN_MAP.items():
        if re.search(pattern, s, re.IGNORECASE):
            return canonical
    # Return cleaned version if no match
    return re.sub(r'[^A-Za-z0-9_]', '_', s).strip('_')


def _normalise_drug(raw: str) -> str:
    s = str(raw).strip()
    for pattern, canonical in DRUG_MAP.items():
        if re.search(pattern, s, re.IGNORECASE):
            return canonical
    return re.sub(r'[^A-Za-z0-9_]', '_', s).strip('_')


def _normalise_country(raw: str) -> str:
    s = str(raw).strip()
    lower = s.lower()
    if lower in COUNTRY_ALIASES:
        return COUNTRY_ALIASES[lower]
    # Title-case as fallback
    return s.title()


# ══════════════════════════════════════════════════════════════════════════════
# Per-schema parsers  →  each returns a tidy DataFrame with standard columns
# ══════════════════════════════════════════════════════════════════════════════

def _parse_glass(df: pd.DataFrame, source_name: str) -> pd.DataFrame:
    """
    Parse WHO GLASS country-aggregated export.

    Expected raw columns (flexible naming):
        Country | Pathogen/Organism | Antibiotic/Drug/Class |
        Year | Resistance% / Resistance_Rate / Percentage_Resistant |
        (optional) N_Isolates / Total_Tested
    """
    country_col = _find_col(df, ['Country', 'CountryName', 'Nation'])
    pathogen_col= _find_col(df, ['Pathogen', 'Organism', 'Bacteria',
                                  'Microorganism', 'Species'])
    drug_col    = _find_col(df, ['Antibiotic', 'Drug', 'AntibioticClass',
                                  'AntimicrobialClass', 'Class', 'Agent'])
    year_col    = _find_col(df, ['Year', 'ReportYear', 'SurveillanceYear'])
    res_col     = _find_col(df, ['Resistance_Percentage', 'Resistance_Pct',
                                  'Percentage_Resistant', 'Resistance_Rate',
                                  'PctResistant', 'Value', 'Percentage',
                                  'ResistancePct'])
    n_col       = _find_col(df, ['N_Isolates', 'Total_Tested', 'N', 'Count',
                                  'Isolates', 'TotalN'])

    missing = [name for name, col in [
        ('country', country_col), ('pathogen', pathogen_col),
        ('drug', drug_col), ('year', year_col), ('resistance', res_col),
    ] if col is None]
    if missing:
        raise ValueError(f"[GLASS parser] Cannot find columns: {missing} "
                         f"in file '{source_name}'. Columns found: {list(df.columns)}")

    rows = df[[country_col, pathogen_col, drug_col, year_col, res_col]].copy()
    rows.columns = ['country', 'pathogen', 'drug', 'year', 'resistance_pct']
    if n_col:
        rows['n_isolates'] = pd.to_numeric(df[n_col], errors='coerce').fillna(0).astype(int)
    else:
        rows['n_isolates'] = 0

    return rows


def _parse_atlas(df: pd.DataFrame, source_name: str) -> pd.DataFrame:
    """
    Parse isolate-level datasets (ATLAS / SMART / KEYSTONE style).

    Each row = one isolate with SIR interpretation.
    Aggregates to country × pathogen × drug × year → resistance_pct.
    """
    country_col = _find_col(df, ['Country', 'CountryName', 'Nation', 'Region'])
    pathogen_col= _find_col(df, ['Organism', 'Species', 'Pathogen',
                                  'Bacteria', 'Microorganism'])
    drug_col    = _find_col(df, ['Antibiotic', 'Agent', 'Drug', 'AntimicrobialAgent',
                                  'AntibioticClass', 'Class'])
    year_col    = _find_col(df, ['Year', 'SurveyYear', 'CollectionYear',
                                  'IsolateYear', 'StudyYear'])
    sir_col     = _find_col(df, ['Interpretation', 'SIR', 'Category',
                                  'Susceptibility', 'Result'])

    missing = [name for name, col in [
        ('country', country_col), ('pathogen', pathogen_col),
        ('drug', drug_col), ('year', year_col), ('SIR', sir_col),
    ] if col is None]
    if missing:
        raise ValueError(f"[ATLAS parser] Cannot find columns: {missing} "
                         f"in file '{source_name}'. Columns found: {list(df.columns)}")

    sub = df[[country_col, pathogen_col, drug_col, year_col, sir_col]].copy()
    sub.columns = ['country', 'pathogen', 'drug', 'year', 'sir']

    # Binary non-susceptible flag
    sub['non_susceptible'] = sub['sir'].isin(SIR_RESISTANT).astype(int)

    # Aggregate to country × pathogen × drug × year
    agg = (sub.groupby(['country', 'pathogen', 'drug', 'year'])
              .agg(resistance_pct=('non_susceptible', lambda x: x.mean() * 100),
                   n_isolates=('non_susceptible', 'count'))
              .reset_index())
    return agg


def _parse_earsnet(df: pd.DataFrame, source_name: str) -> pd.DataFrame:
    """
    Parse ECDC EARS-Net downloads.

    Typical columns: RegionName / CountryName, Bacteria, AntibioticGroup,
                     Percentage, Time (year)
    """
    country_col = _find_col(df, ['RegionName', 'CountryName', 'Country'])
    pathogen_col= _find_col(df, ['Bacteria', 'Microorganism', 'Pathogen'])
    drug_col    = _find_col(df, ['AntibioticGroup', 'Antibiotic', 'Class'])
    year_col    = _find_col(df, ['Time', 'Year', 'ReportYear'])
    res_col     = _find_col(df, ['Percentage', 'Value', 'ResistancePercentage',
                                  'Resistance_Pct'])
    n_col       = _find_col(df, ['NumValue', 'N', 'Isolates', 'TotalTested'])

    missing = [name for name, col in [
        ('country', country_col), ('pathogen', pathogen_col),
        ('drug', drug_col), ('year', year_col), ('resistance', res_col),
    ] if col is None]
    if missing:
        raise ValueError(f"[EARS-Net parser] Cannot find columns: {missing} "
                         f"in file '{source_name}'. Columns found: {list(df.columns)}")

    rows = df[[country_col, pathogen_col, drug_col, year_col, res_col]].copy()
    rows.columns = ['country', 'pathogen', 'drug', 'year', 'resistance_pct']
    if n_col:
        rows['n_isolates'] = pd.to_numeric(df[n_col], errors='coerce').fillna(0).astype(int)
    else:
        rows['n_isolates'] = 0

    return rows


def _parse_generic(df: pd.DataFrame, source_name: str) -> pd.DataFrame:
    """
    Last-resort parser: fuzzy-find the four mandatory columns.
    Works for any file that has the right data but non-standard headers.
    """
    country_col = _find_col(df, ['Country', 'Nation', 'CountryName', 'Region',
                                  'CountryRegion', 'Site'])
    pathogen_col= _find_col(df, ['Pathogen', 'Organism', 'Species', 'Bacteria',
                                  'Microorganism', 'Bug'])
    drug_col    = _find_col(df, ['Antibiotic', 'Drug', 'Agent', 'AntibioticClass',
                                  'Class', 'AntimicrobialClass', 'Antimicrobial'])
    year_col    = _find_col(df, ['Year', 'Date', 'SurveyYear', 'ReportYear',
                                  'CollectionYear', 'Time'])
    res_col     = _find_col(df, ['Resistance_Pct', 'Resistance_Percentage',
                                  'Percentage_Resistant', 'PctResistant',
                                  'ResistanceRate', 'Value', 'Percentage',
                                  'ResistancePct', 'Resistant_Pct'])
    sir_col     = _find_col(df, ['Interpretation', 'SIR', 'Category',
                                  'Susceptibility']) if res_col is None else None
    n_col       = _find_col(df, ['N', 'N_Isolates', 'Count', 'TotalTested',
                                  'Total', 'Isolates'])

    if country_col is None or pathogen_col is None or drug_col is None or year_col is None:
        raise ValueError(
            f"[Generic parser] Could not identify mandatory columns in '{source_name}'.\n"
            f"  Found columns: {list(df.columns)}\n"
            f"  Needed: country, pathogen, drug, year, + (resistance_pct or SIR)"
        )

    if sir_col is not None:
        # Isolate-level — delegate to ATLAS-style aggregation
        df2 = df.rename(columns={
            country_col: 'Country', pathogen_col: 'Organism',
            drug_col: 'Antibiotic', year_col: 'Year', sir_col: 'Interpretation'
        })
        return _parse_atlas(df2, source_name)

    if res_col is None:
        raise ValueError(
            f"[Generic parser] No resistance percentage or SIR column found in '{source_name}'."
        )

    rows = df[[country_col, pathogen_col, drug_col, year_col, res_col]].copy()
    rows.columns = ['country', 'pathogen', 'drug', 'year', 'resistance_pct']
    if n_col:
        rows['n_isolates'] = pd.to_numeric(df[n_col], errors='coerce').fillna(0).astype(int)
    else:
        rows['n_isolates'] = 0
    return rows


# ══════════════════════════════════════════════════════════════════════════════
# Single-file loader
# ══════════════════════════════════════════════════════════════════════════════

def _read_file(filepath: str) -> pd.DataFrame:
    """Read CSV or Excel into a DataFrame."""
    path = Path(filepath)
    ext  = path.suffix.lower()
    if ext == '.csv':
        # Try common encodings
        for enc in ('utf-8', 'latin-1', 'cp1252'):
            try:
                return pd.read_csv(filepath, encoding=enc, low_memory=False)
            except UnicodeDecodeError:
                continue
        raise ValueError(f"Cannot decode '{filepath}' with utf-8 / latin-1 / cp1252")
    elif ext in ('.xlsx', '.xls'):
        return pd.read_excel(filepath, engine='openpyxl' if ext == '.xlsx' else 'xlrd')
    else:
        raise ValueError(f"Unsupported file type '{ext}' for '{filepath}'. "
                         "Use .csv, .xlsx, or .xls")


def _clean_tidy(df: pd.DataFrame, source_name: str) -> pd.DataFrame:
    """
    Post-parse cleaning applied to every source:
      • Normalise country / pathogen / drug names
      • Parse year to int
      • Clip resistance_pct to [0, 100]
      • Drop rows with nulls in key columns
      • Add source_file column
    """
    df = df.copy()

    # Normalise text fields
    df['country']  = df['country'].astype(str).str.strip().apply(_normalise_country)
    df['pathogen'] = df['pathogen'].astype(str).str.strip().apply(_normalise_pathogen)
    df['drug']     = df['drug'].astype(str).str.strip().apply(_normalise_drug)

    # Year → int
    df['year'] = pd.to_numeric(df['year'], errors='coerce')
    df = df.dropna(subset=['year'])
    df['year'] = df['year'].astype(int)

    # Resistance % → float, clip
    df['resistance_pct'] = pd.to_numeric(df['resistance_pct'], errors='coerce')
    df['resistance_pct'] = df['resistance_pct'].clip(0, 100)

    # Ensure n_isolates column exists
    if 'n_isolates' not in df.columns:
        df['n_isolates'] = 0
    df['n_isolates'] = pd.to_numeric(df['n_isolates'], errors='coerce').fillna(0).astype(int)

    # Build combo key
    df['combo'] = df['pathogen'] + '_' + df['drug']

    # Source provenance
    df['source_file'] = Path(source_name).name

    # Drop rows missing key values
    df = df.dropna(subset=['country', 'pathogen', 'drug', 'year', 'resistance_pct'])

    return df[['country', 'pathogen', 'drug', 'combo', 'year',
               'resistance_pct', 'n_isolates', 'source_file']]


def load_single_file(filepath: str, force_schema: Optional[str] = None,
                     default_year: Optional[int] = None) -> pd.DataFrame:
    """
    Load, detect schema, parse, and clean one AMR surveillance file.

    Parameters
    ----------
    filepath     : Path to CSV or Excel file.
    force_schema : Override auto-detection: 'glass' | 'atlas' | 'atlas_sir' |
                   'earsnet' | 'mic' | 'spidaar' | 'generic'
    default_year : Required only for schemas with no internal year column
                   (currently: 'spidaar'). Ignored by all other schemas.

    Returns
    -------
    Tidy DataFrame with standard columns.
    """
    print(f"    Reading  : {Path(filepath).name}")
    raw = _read_file(filepath)
    print(f"    Rows×Cols: {raw.shape[0]:,} × {raw.shape[1]}  "
          f"| Cols: {list(raw.columns[:6])}{'…' if len(raw.columns)>6 else ''}")

    schema = force_schema or _detect_schema(raw, filepath)
    print(f"    Schema   : {schema}")

    parsers = {
        'glass':     _parse_glass,
        'atlas':     _parse_atlas,
        'atlas_sir': _parse_atlas_sir,
        'earsnet':   _parse_earsnet,
        'mic':       _parse_mic,
        'spidaar':   _parse_spidaar,
        'generic':   _parse_generic,
    }
    if schema not in parsers:
        raise ValueError(f"Unknown schema '{schema}'. Choose from: {list(parsers)}")

    if schema == 'spidaar':
        parsed = _parse_spidaar(raw, filepath, default_year=default_year)
    else:
        parsed = parsers[schema](raw, filepath)
    cleaned = _clean_tidy(parsed, filepath)
    print(f"    Parsed   : {len(cleaned):,} tidy rows  "
          f"| Countries: {cleaned.country.nunique()}  "
          f"| Combos: {cleaned.combo.nunique()}  "
          f"| Years: {cleaned.year.min()}–{cleaned.year.max()}")
    return cleaned


# ══════════════════════════════════════════════════════════════════════════════
# Multi-file loader
# ══════════════════════════════════════════════════════════════════════════════

def load_amr_files(
    file_paths: list[str],
    conflict_strategy: str = 'mean',
    force_schemas: Optional[dict[str, str]] = None,
    filter_combos: Optional[list[str]] = None,
    filter_countries: Optional[list[str]] = None,
    year_range: Optional[tuple[int, int]] = None,
    default_year: Optional[int] = None,
) -> pd.DataFrame:
    """
    Load and harmonise multiple AMR surveillance files into one unified panel.

    Parameters
    ----------
    file_paths        : Ordered list of file paths. For 'first'/'last' conflict
                        strategy, ordering matters (first = highest priority).
    conflict_strategy : How to resolve overlapping (country, combo, year) entries:
                        'mean'   — (isolate-count-)weighted mean  [default]
                        'median' — median of all reported values
                        'max'    — take highest reported value
                        'first'  — keep value from the first file in the list
                        'last'   — keep value from the last file in the list
    force_schemas     : Optional dict mapping filename → schema override,
                        e.g. {'my_data.csv': 'atlas'}.
    filter_combos     : If provided, keep only these combo strings
                        e.g. ['E_coli_3GC', 'K_pneumoniae_CARB'].
                        Pass None to keep all combos found.
    filter_countries  : If provided, keep only these country names.
                        Pass None to keep all countries found.
    year_range        : Optional (min_year, max_year) inclusive filter.
    default_year      : Required only if any file uses a yearless schema
                        (currently: SPIDAAR patient-level point-prevalence
                        files). Applied uniformly to every row of that file.

    Returns
    -------
    Harmonised panel with columns:
        country, pathogen, drug, combo, year,
        resistance_pct, n_isolates, source_files
    """
    if not file_paths:
        raise ValueError("file_paths is empty. Provide at least one AMR file.")

    force_schemas = force_schemas or {}
    all_frames: list[pd.DataFrame] = []

    print(f"\n[multi_file_loader] Loading {len(file_paths)} file(s) …")
    for i, fp in enumerate(file_paths, 1):
        print(f"\n  File {i}/{len(file_paths)}")
        schema_override = force_schemas.get(Path(fp).name) or force_schemas.get(fp)
        try:
            df = load_single_file(fp, force_schema=schema_override,
                                  default_year=default_year)
            df['_file_order'] = i
            all_frames.append(df)
        except Exception as e:
            print(f"    !  Skipping '{fp}':")
            for line in str(e).split('\n'):
                print(f"       {line}")

    if not all_frames:
        raise RuntimeError("No files could be loaded successfully.")

    combined = pd.concat(all_frames, ignore_index=True)
    print(f"\n[multi_file_loader] Combined raw rows: {len(combined):,} "
          f"from {combined.source_file.nunique()} file(s)")

    # ── Optional filters ──────────────────────────────────────────────────────
    if filter_combos:
        combined = combined[combined.combo.isin(filter_combos)]
        print(f"  Filtered to combos  : {filter_combos}")
    if filter_countries:
        combined = combined[combined.country.isin(filter_countries)]
        print(f"  Filtered to countries: {len(filter_countries)} countries")
    if year_range:
        combined = combined[
            (combined.year >= year_range[0]) & (combined.year <= year_range[1])
        ]
        print(f"  Filtered to years   : {year_range[0]}–{year_range[1]}")

    # ── Conflict resolution ───────────────────────────────────────────────────
    print(f"\n[multi_file_loader] Resolving conflicts with strategy='{conflict_strategy}' …")
    resolved = _resolve_conflicts(combined, strategy=conflict_strategy)

    print(f"[multi_file_loader] Final panel: {len(resolved):,} rows  "
          f"| {resolved.country.nunique()} countries  "
          f"| {resolved.combo.nunique()} combos  "
          f"| {resolved.year.min()}–{resolved.year.max()}")

    return resolved


def _resolve_conflicts(df: pd.DataFrame, strategy: str) -> pd.DataFrame:
    """
    Collapse multiple entries for the same (country, combo, year) into one.

    Adds 'source_files' column listing all contributing file names.
    """
    key = ['country', 'pathogen', 'drug', 'combo', 'year']

    # Always record which files contributed
    sources = (df.groupby(key)['source_file']
               .apply(lambda x: ' | '.join(sorted(x.unique())))
               .reset_index()
               .rename(columns={'source_file': 'source_files'}))

    if strategy == 'first':
        df_s = df.sort_values('_file_order').groupby(key, as_index=False).first()
    elif strategy == 'last':
        df_s = df.sort_values('_file_order').groupby(key, as_index=False).last()
    elif strategy == 'max':
        df_s = df.groupby(key, as_index=False)['resistance_pct'].max()
        df_s = df_s.merge(
            df.groupby(key, as_index=False)['n_isolates'].sum(), on=key
        )
    elif strategy == 'median':
        df_s = df.groupby(key, as_index=False)['resistance_pct'].median()
        df_s = df_s.merge(
            df.groupby(key, as_index=False)['n_isolates'].sum(), on=key
        )
    else:  # 'mean' — weighted by n_isolates where available
        def weighted_mean(g):
            n = g['n_isolates'].values.astype(float)
            r = g['resistance_pct'].values
            if n.sum() > 0:
                return np.average(r, weights=np.where(n > 0, n, 1))
            return r.mean()

        df_s = (df.groupby(key, as_index=False)
                  .apply(lambda g: pd.Series({
                      'resistance_pct': weighted_mean(g),
                      'n_isolates': int(g['n_isolates'].sum()),
                  }))
                  .reset_index())
        # After apply, key columns may be in index
        if all(k in df_s.index.names for k in key):
            df_s = df_s.reset_index()

    # Ensure key columns are present after group operations
    if 'combo' not in df_s.columns:
        df_s['combo'] = df_s['pathogen'] + '_' + df_s['drug']

    df_s = df_s.merge(sources, on=key, how='left')
    df_s = df_s.drop(columns=['_file_order'], errors='ignore')
    return df_s.reset_index(drop=True)


# ══════════════════════════════════════════════════════════════════════════════
# Ingestion report
# ══════════════════════════════════════════════════════════════════════════════

def ingestion_report(df: pd.DataFrame) -> str:
    """
    Return a human-readable summary of the harmonised panel for logging.
    """
    lines = [
        "═" * 60,
        "  AMR Multi-File Ingestion Report",
        "═" * 60,
        f"  Total rows      : {len(df):,}",
        f"  Countries       : {df.country.nunique()}  → {sorted(df.country.unique())}",
        f"  Pathogen-combos : {df.combo.nunique()}  → {sorted(df.combo.unique())}",
        f"  Year range      : {df.year.min()} – {df.year.max()}",
        f"  Mean resistance : {df.resistance_pct.mean():.1f}%  "
        f"(min {df.resistance_pct.min():.1f}%, max {df.resistance_pct.max():.1f}%)",
        "",
        "  Rows per source file:",
    ]
    if 'source_files' in df.columns:
        # Explode the pipe-separated source_files to count
        src_counts = (df['source_files']
                      .str.split(' | ')
                      .explode()
                      .value_counts())
        for src, cnt in src_counts.items():
            lines.append(f"    {src:<50s} {cnt:>6,} rows")
    lines.append("═" * 60)
    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════════
# Data quality report
# ══════════════════════════════════════════════════════════════════════════════

def data_quality_report(df: pd.DataFrame) -> str:
    """
    Print a concise data-quality summary after ingestion so the user
    can spot issues (e.g. all-zero resistance, sparse combos) before
    running the full pipeline.
    """
    lines = [
        "",
        "=" * 62,
        "  DATA QUALITY REPORT",
        "=" * 62,
        f"  Rows            : {len(df):,}",
        f"  Countries       : {df.country.nunique()}",
        f"  Pathogen-combos : {df.combo.nunique()}",
        f"  Year range      : {df.year.min()} - {df.year.max()}",
        f"  Mean resistance : {df.resistance_pct.mean():.1f}%",
        f"  Zero-resistance : {(df.resistance_pct == 0).sum():,} rows "
          f"({(df.resistance_pct == 0).mean()*100:.1f}%)",
        "",
        "  Top 10 combos by isolate count:",
        f"  {'Combo':<35} {'N':>6}  {'Mean R%':>8}  {'Countries':>10}",
        f"  {'-'*62}",
    ]
    top = (df.groupby('combo')
             .agg(n=('resistance_pct','count'),
                  mean_r=('resistance_pct','mean'),
                  countries=('country','nunique'))
             .sort_values('n', ascending=False)
             .head(10))
    for combo, row in top.iterrows():
        lines.append(f"  {combo:<35} {row.n:>6,}  {row.mean_r:>7.1f}%  {row.countries:>10}")

    zero_combos = (df.groupby('combo')['resistance_pct'].mean() == 0).sum()
    if zero_combos > 0:
        lines += [
            "",
            f"  WARNING: {zero_combos} combos have 0% mean resistance.",
            "  This usually means all isolates are susceptible (below EUCAST",
            "  breakpoints) OR the MIC breakpoints need adjusting for this",
            "  organism/antibiotic combination.",
            "  Forecasts for these combos will produce flat 0% projections.",
        ]
    lines.append("=" * 62)
    return "\n".join(lines)
