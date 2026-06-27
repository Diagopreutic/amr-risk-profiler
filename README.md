# AMR Risk Profiler
Machine learning pipeline for AMR resistance profiling, risk tier classification, and modifiable driver identification across multi-format global surveillance datasets.

## Installation
git clone https://github.com/your-username/amr-risk-profiler.git
cd amr-risk-profiler
pip install -r requirements.txt

## Quick start
### Install dependencies
pip install -r requirements.txt

### Pre-fetch World Bank cache
python fetch_worldbank_cache.py

### Run on your surveillance file
python main.py --amr your_data.csv

### Specify a historical start year
python main.py --amr your_data.csv --start-year 2010
