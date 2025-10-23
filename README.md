## FUSION-3D: Technical + Twitter Sentiment Ensemble for AAPL Direction (1d/3d/5d)

This repository implements a short-horizon equity modeling pipeline that fuses:

- AAPL technical signals (daily OHLCV + indicators)
- Sector context via VGT ETF
- Twitter sentiment (FinBERT-based, with fallback)

Predictions target the next 1/3/5 trading-day direction. The system includes independent scorers for AAPL, VGT, and sentiment, plus an ensemble weight optimizer and comprehensive visualizations.

### Highlights

- Leakage-safe, walk-forward style rolling analysis for per-day predictions
- Kronos-inspired modeling with a local offline implementation and an optional Hugging Face path
- Grid search over AAPL/VGT/Sentiment weights for the ensemble
- One-click main script to orchestrate scoring, ensembling, and visualization
- Fusion weights tuned on the first 80% of the timeline with 5-fold TimeSeriesSplit; the final 20% is a strict chronological holdout used only once for testing

## Project Structure

```
/Users/yuanyifu/Documents/NUS/Courses/BAP Practice/Formal Work/
├─ data/
│  ├─ stock_105.AAPL_2025-09.csv
│  ├─ etf_VGT_2025-09.csv
│  └─ tweets_105.AAPL_2025-09.csv
├─ docs/
│  ├─ kronos.md
│  ├─ proposal.txt
│  └─ stock_105_AAPL_columns_explanation.md
├─ analysis/
│  ├─ __init__.py
│  ├─ kronos_model.py
│  ├─ kronos_aapl_analysis.py
│  └─ kronos_vgt_analysis.py
├─ sentiment/
│  ├─ __init__.py
│  └─ sentiment_analysis.py
├─ ensemble/
│  ├─ __init__.py
│  ├─ score_ensemble.py
│  ├─ visualization.py
│  └─ main_ensemble_analysis.py
├─ benchmarks/
│  ├─ __init__.py
│  ├─ baseline.md
│  ├─ benchmark_models.py
│  ├─ model_factory.py
│  ├─ comprehensive_evaluator.py
│  └─ test_benchmark_system.py
├─ config.py                              # Centralized paths
├─ test_dependencies.py                  # Pre-flight dependency checker
├─ result/
└─ kronos_prediction_results.csv
```

See `docs/stock_105_AAPL_columns_explanation.md` for field semantics of the AAPL dataset.

## Data

- AAPL: `data/stock_105.AAPL_2025-09.csv` (2015-09–2025-09, 32 columns including indicators)
- VGT: `data/etf_VGT_2025-09.csv`
- Tweets: `data/tweets_105.AAPL_2025-09.csv` (FinBERT targets: text + engagement fields)

Notes

- All dates are daily and aligned close-to-close.
- Indicators are computed with TA-Lib (RSI, MACD, BBands, ATR, etc.).

## Methods Overview

- AAPL/VGT scorers (`kronos_aapl_analysis.py`, `kronos_vgt_analysis.py`)
  - Compute technical indicators via TA-Lib
  - Rolling 60-day window; generate per-day predictions for 1d/3d/5d via unified features
  - Outputs standardized scores in [-1, 1] and actual labels for evaluation
- Sentiment scorer (`sentiment_analysis.py`)
  - FinBERT-based sentiment extraction; robust keyword fallback when FinBERT unavailable
  - Aggregates to daily sentiment features and produces 1d/3d/5d sentiment scores
- Ensemble (`ensemble/score_ensemble.py`)
  - Loads/creates `aapl_scores.csv`, `vgt_scores.csv`, `sentiment_scores.csv`
  - Inner join on date; tunes fusion weights via 5-fold TimeSeriesSplit on the first 80% of the series (chronological CV, no leakage into the test window)
  - Evaluates once on the last 20% holdout; persists `combined_scores.csv`, `ensemble_results.pkl`, a text summary, and a full grid-search dump
- Visualization (`ensemble/visualization.py`)
  - Produces multiple figures under `result/` including comparisons and grid-search analyses
  - Non-interactive backend; plots are saved to disk (no pop-up windows)

## Environment and Dependencies

- Python 3.10+
- Recommended: use a virtual environment (conda or venv)

Core packages

- numpy, pandas, scikit-learn, joblib
- matplotlib, seaborn
- TA-Lib
- xgboost
- torch, transformers (for Kronos/FinBERT paths; robust fallbacks if offline)

TA-Lib installation

- Conda (recommended):
  ```bash
  conda install -c conda-forge ta-lib
  ```
- Homebrew + pip (macOS):
  ```bash
  brew install ta-lib
  pip install TA-Lib
  ```

PyTorch (CPU example)

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

Transformers/FinBERT

```bash
pip install transformers
```

XGBoost

```bash
pip install xgboost
```

If internet is restricted, the code falls back to a keyword-based sentiment scorer and a local Kronos-style predictor.

## Quickstart

1) Activate your environment

```bash
# activate your environment (conda or venv), e.g.:
# python -m venv .venv && source .venv/bin/activate
# conda activate <your-env>
```

2) Verify dependencies (pre-flight)

```bash
python test_dependencies.py --verbose
# If you don't have the data files yet, you can skip file checks:
# python test_dependencies.py --verbose --skip-data
```
3) Prepare sentiment scores (optional)

- Paths are centralized via `config.py`. If you want to precompute daily sentiment scores:

```bash
python - << 'PY'
from sentiment.sentiment_analysis import SentimentScoreGenerator
from config import SENTIMENT_SCORES
sg = SentimentScoreGenerator()
df = sg.generate_scores()
if df is not None:
    df.to_csv(SENTIMENT_SCORES, index=False)
    print('✓ sentiment_scores.csv saved')
PY
```

4) Run the ensemble pipeline (single-line)

```bash
python -m ensemble.main_ensemble_analysis
```

5) Run the benchmark suite (single-line)

```bash
python -m benchmarks.comprehensive_evaluator
```

6) Outputs

- In `result/ensemble/`:
  - Data: `aapl_scores.csv`, `vgt_scores.csv`, `sentiment_scores.csv`, `combined_scores.csv`
  - Artifacts: `ensemble_results.pkl`, `ensemble_summary.txt`
  - Numeric dumps: `grid_search_results.txt` (full CV grid per horizon), `summary_statistics.txt`
  - Plots: `model_comparison.png`, `ensemble_performance.png`, `time_series_analysis.png`, `grid_search_analysis.png`, `final_results_summary.png`
- In `result/benchmarks/`:
  - Artifacts: `comprehensive_evaluation_results.pkl`
  - Numeric dumps: `evaluation_summary.txt` (ranked summary), `evaluation_results_full.txt` (per-model metrics per horizon)
  - Plots: `comprehensive_model_comparison.png`, `category_performance_analysis.png`, `horizon_comparison_analysis.png`, `performance_distribution.png`

## Benchmarks: included models and how to configure

- By default, benchmarks evaluate the following models via `benchmarks/model_factory.py`:
  - ML: `logistic_regression`, `random_forest_100`, `xgboost_100`
  - DL: `lstm_64_2`
  - Existing: `kronos_aapl`, `kronos_vgt`, `sentiment_analysis`
- Technical rule-based baselines are currently disabled.

Control what runs:

- To add/remove specific models: edit `_register_models()` in `benchmarks/model_factory.py` (comment/uncomment entries).
- To restrict categories evaluated (e.g., only ML and DL): in `benchmarks/comprehensive_evaluator.py`, change the default categories in `ComprehensiveModelEvaluator.evaluate_all_models(...)` from `['ml', 'dl', 'existing']` to your desired list, or pass a categories list where it is called in `main()`.
  - Examples: `['ml']`, `['ml', 'dl']`, or `['existing']`.

Quick run variants (no pop-ups; files saved only):

- Only baselines (ML + DL):
  ```bash
  python -c "from benchmarks.comprehensive_evaluator import ComprehensiveModelEvaluator as E; e=E(); e.evaluate_all_models(categories=['ml','dl']); e.create_performance_visualization(); e.save_results()"
  ```
- Only existing components (Kronos AAPL/VGT + Sentiment):
  ```bash
  python -c "from benchmarks.comprehensive_evaluator import ComprehensiveModelEvaluator as E; e=E(); e.evaluate_all_models(categories=['existing']); e.create_performance_visualization(); e.save_results()"
  ```

## Usage Notes and Customization

- Real Kronos vs local predictor
  - The default scorers use a local Kronos-style predictor. A real HF model path exists in `kronos_model.py` and can be toggled by setting `offline_mode=False` and enabling `use_real_kronos=True` in the rolling analysis classes (requires small code change in AAPL/VGT modules).
- Sentiment analysis
  - If FinBERT downloads are blocked, the fallback keyword-based analyzer is used automatically. For best results, keep internet access to fetch `ProsusAI/finbert` weights once.
- Paths
  - Paths are centralized in `config.py` (no hard-coded absolute paths). If you relocate data or want a different output folder, modify `config.py` only.

## Reproducibility and Evaluation

- Rolling predictions use a 60-day lookback and produce per-day labels and predictions for 1d/3d/5d.
- Ensemble tunes fusion weights with 5-fold TimeSeriesSplit on the training partition (first 80%) only; the last 20% is untouched during tuning and used once as a chronological holdout for final evaluation.
- Key metrics: Accuracy, Precision, Recall, F1; plus returns-oriented indicators in the scorers.

## References

- Kronos: A Foundation Model for the Language of Financial Markets — see `docs/kronos.md` and the Hugging Face models (e.g., `NeoQuasar/Kronos-mini`).
- FinBERT: Araci, D. (2019). Financial sentiment analysis with pre-trained language models. `ProsusAI/finbert` on Hugging Face.

## Acknowledgements

- This project leverages ideas and tooling from the Kronos foundation model and FinBERT community resources.
