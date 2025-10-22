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


## Project Structure

```
/Users/yuanyifu/Documents/NUS/Courses/BAP Practice/Formal Work/
├─ data/
│  ├─ stock_105.AAPL_2025-09.csv          # AAPL daily OHLCV (+ engineered columns)
│  ├─ etf_VGT_2025-09.csv                 # VGT ETF daily OHLCV
│  └─ tweets_105.AAPL_2025-09.csv         # AAPL tweets with engagement fields
├─ docs/
│  ├─ kronos.md                           # Kronos FM overview and usage (reference)
│  ├─ proposal.txt                        # Project rationale and plan
│  └─ stock_105_AAPL_columns_explanation.md
├─ kronos_model.py                        # Local Kronos-style tokenizer/model/predictor
├─ kronos_aapl_analysis.py                # AAPL scorer (tech + Kronos-inspired rolling)
├─ kronos_vgt_analysis.py                 # VGT scorer (tech + Kronos-inspired rolling)
├─ sentiment_analysis.py                  # FinBERT sentiment scorer (+ robust fallback)
├─ score_ensemble.py                      # Weight search + combine AAPL/VGT/Sentiment
├─ visualization.py                       # Visual reports (saved under result_new/)
├─ main_ensemble_analysis.py              # Orchestration entrypoint
├─ result_new/                            # Generated plots and summaries
└─ kronos_prediction_results.csv          # Example output (optional)
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
- Ensemble (`score_ensemble.py`)
  - Loads/creates `aapl_scores.csv`, `vgt_scores.csv`, `sentiment_scores.csv`
  - Inner join on date; time-series CV; searches weight combos for best accuracy
  - Persists `combined_scores.csv`, `ensemble_results.pkl`, and text summary
- Visualization (`visualization.py`)
  - Produces multiple figures under `result_new/` including comparisons and grid-search analyses


## Environment and Dependencies
- Python 3.10+
- Recommended: conda env `general` (as per your local setup)

Core packages
- numpy, pandas, scikit-learn, joblib
- matplotlib, seaborn
- TA-Lib
- torch, transformers (for Kronos/FinBERT paths; robust fallbacks if offline)

TA-Lib installation
- Conda (recommended):
  ```bash
  conda activate general
  conda install -c conda-forge ta-lib
  ```
- Homebrew + pip (macOS):
  ```bash
  brew install ta-lib
  pip install TA-Lib
  ```

PyTorch (CPU example)
```bash
conda activate general
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

Transformers/FinBERT
```bash
conda activate general
pip install transformers
```

If internet is restricted, the code falls back to a keyword-based sentiment scorer and a local Kronos-style predictor.


## Quickstart
1) Activate your environment
```bash
conda activate general
```

2) Ensure TA-Lib and core packages are installed (see above)

3) Prepare sentiment scores (recommended due to path defaults in `sentiment_analysis.py`)
- The sentiment module defaults to a user-specific path. Generate scores using the project data path and save to the project root as `sentiment_scores.csv` so the ensemble can load it directly:
```bash
python - << 'PY'
from sentiment_analysis import SentimentScoreGenerator
import pandas as pd
p = '/Users/yuanyifu/Documents/NUS/Courses/BAP Practice/Formal Work/data/tweets_105.AAPL_2025-09.csv'
sg = SentimentScoreGenerator(data_path=p)
df = sg.generate_scores()
if df is not None:
    df.to_csv('/Users/yuanyifu/Documents/NUS/Courses/BAP Practice/Formal Work/sentiment_scores.csv', index=False)
    print('✓ sentiment_scores.csv saved')
else:
    print('✗ sentiment score generation failed')
PY
```

4) Run the complete pipeline
```bash
python /Users/yuanyifu/Documents/NUS/Courses/BAP Practice/Formal Work/main_ensemble_analysis.py
```

5) Outputs
- Root directory
  - `aapl_scores.csv`, `vgt_scores.csv`, `sentiment_scores.csv`
  - `combined_scores.csv`, `ensemble_results.pkl`, `ensemble_summary.txt`
- Plots in `result_new/`:
  - `model_comparison.png`, `ensemble_performance.png`, `time_series_analysis.png`, `grid_search_analysis.png`, `final_results_summary.png`, `summary_statistics.txt`


## Usage Notes and Customization
- Real Kronos vs local predictor
  - The default scorers use a local Kronos-style predictor. A real HF model path exists in `kronos_model.py` and can be toggled by setting `offline_mode=False` and enabling `use_real_kronos=True` in the rolling analysis classes (requires small code change in AAPL/VGT modules).
- Sentiment analysis
  - If FinBERT downloads are blocked, the fallback keyword-based analyzer is used automatically. For best results, keep internet access to fetch `ProsusAI/finbert` weights once.
- Paths
  - The sentiment module contains hard-coded defaults for another user. The Quickstart step above avoids edits by pre-creating `sentiment_scores.csv` in the project root. Alternatively, update `SentimentScoreGenerator(data_path=...)` in `score_ensemble.py` or fix the default path in `sentiment_analysis.py`.


## Reproducibility and Evaluation
- Rolling predictions use a 60-day lookback and produce per-day labels and predictions for 1d/3d/5d.
- Ensemble uses time-series CV to pick weights and evaluates on a holdout split.
- Key metrics: Accuracy, Precision, Recall, F1; plus returns-oriented indicators in the scorers.


## References
- Kronos: A Foundation Model for the Language of Financial Markets — see `docs/kronos.md` and the Hugging Face models (e.g., `NeoQuasar/Kronos-mini`).
- FinBERT: Araci, D. (2019). Financial sentiment analysis with pre-trained language models. `ProsusAI/finbert` on Hugging Face.


## Acknowledgements
- This project leverages ideas and tooling from the Kronos foundation model and FinBERT community resources.
