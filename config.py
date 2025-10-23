from pathlib import Path

# Project root
PROJECT_ROOT = Path(__file__).resolve().parent

# Data and outputs
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "result"
ENSEMBLE_OUTPUT_DIR = OUTPUT_DIR / "ensemble"
BENCHMARKS_OUTPUT_DIR = OUTPUT_DIR / "benchmarks"

# Input files
AAPL_FILE = DATA_DIR / "stock_105.AAPL_2025-09.csv"
VGT_FILE = DATA_DIR / "etf_VGT_2025-09.csv"
TWEETS_FILE = DATA_DIR / "tweets_105.AAPL_2025-09.csv"

# Intermediate and final artifacts (saved under result/ensemble)
AAPL_SCORES = ENSEMBLE_OUTPUT_DIR / "aapl_scores.csv"
VGT_SCORES = ENSEMBLE_OUTPUT_DIR / "vgt_scores.csv"
SENTIMENT_SCORES = ENSEMBLE_OUTPUT_DIR / "sentiment_scores.csv"
COMBINED_SCORES = ENSEMBLE_OUTPUT_DIR / "combined_scores.csv"
ENSEMBLE_RESULTS = ENSEMBLE_OUTPUT_DIR / "ensemble_results.pkl"
ENSEMBLE_SUMMARY = ENSEMBLE_OUTPUT_DIR / "ensemble_summary.txt"

# Ensure output directories exist at import time
for _dir in (OUTPUT_DIR, ENSEMBLE_OUTPUT_DIR, BENCHMARKS_OUTPUT_DIR):
    _dir.mkdir(parents=True, exist_ok=True)
