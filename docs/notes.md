
- For day 61: the previous 60 days are used only as the lookback window to compute features and make a prediction. There is no 80/20 split inside that 60-day window.
- Where 80/20 happens: after you have predictions for all days, the ensemble stage splits the whole time series chronologically into 80% train and 20% test. Inside the 80% train portion it uses TimeSeriesSplit (5 folds) to pick the best AAPL/VGT/Sentiment weights; then it evaluates once on the last 20% holdout.

So: rolling 60-day window = feature context per day; 80/20 + TimeSeriesSplit = ensemble training/evaluation over the full timeline.


Almost. Two clarifications:

- Component level (AAPL, VGT, Sentiment): yes—each day t uses the previous 60 days only to build features and output that day’s prediction. No 80/20 split inside that 60-day window.

```241:260:/Users/yuanyifu/Documents/NUS/Courses/BAP
def run_rolling_analysis(self):
    ...
    start_idx = self.lookback_window
    end_idx = len(self.data) - 5
    for i in range(start_idx, end_idx):
        features = self.predictor.prepare_features(self.data, i - self.lookback_window, i)
        ...
```

- Ensemble level: weights are selected by 5-fold TimeSeriesSplit over the full timeline, then evaluated once on a chronological 80/20 holdout.(相当于用了整个time series data来做5-fold CV，得到ensemble model的最佳fusion weights，又在整个time series data的20%上做testing，有问题)

```263:266:/Users/yuanyifu/Documents/NUS/Courses/BAP
# Use TimeSeriesSplit for time series data
tscv = TimeSeriesSplit(n_splits=5)
```

```350:357:/Users/yuanyifu/Documents/NUS/Courses/BAP
# Split data (use last 20% for testing)
split_idx = int(len(X) * 0.8)
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]
# Get best weights (from CV on full series)
best_weights = self.grid_search_results[horizon]['best_weights']
```


* What TimeSeriesSplit does: keeps chronology within each fold — train uses earlier dates, validation uses later dates. No shuffling. That’s it.
* What it doesn’t guarantee: if you run it on the entire series, the future (your intended test window) still influences weight selection via CV. That’s leakage for final evaluation.

Do this instead:* First hold out the final window (e.g., last 20%) as test.

* Tune fusion weights only on the first 80% using TimeSeriesSplit.
* Optionally add a purge/embargo around split boundaries (e.g., 3–5 days) to avoid label overlap leakage.
