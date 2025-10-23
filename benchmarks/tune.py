#!/usr/bin/env python3
"""CLI for per-model hyperparameter tuning (ML/DL).

Examples:
  python -m benchmarks.tune --model random_forest --horizon 1d --cv-folds 5
  python -m benchmarks.tune --model xgboost --horizon 3d --grid grids/xgb_3d.json
"""

import os
import json
import argparse
from typing import Dict, Any

from benchmarks.comprehensive_evaluator import ComprehensiveModelEvaluator
from benchmarks.model_factory import ModelFactory


def main():
    parser = argparse.ArgumentParser(description="Per-model hyperparameter tuning")
    parser.add_argument("--model", required=True, help="Model key (e.g., random_forest, xgboost, logistic_regression, lstm)")
    parser.add_argument("--horizon", default="1d", choices=["1d","3d","5d"], help="Prediction horizon")
    parser.add_argument("--cv-folds", type=int, default=5, help="TimeSeriesSplit folds")
    parser.add_argument("--grid", default=None, help="Path to JSON param grid (optional)")
    args = parser.parse_args()

    # Load grid with robust path resolution
    grid: Dict[str, Any] = {}
    if args.grid:
        grid_path = args.grid
        if not os.path.isabs(grid_path) and not os.path.exists(grid_path):
            # Try relative to this benchmarks module directory
            script_dir = os.path.dirname(os.path.abspath(__file__))
            candidate = os.path.join(script_dir, grid_path)
            if os.path.exists(candidate):
                grid_path = candidate
        if not os.path.exists(grid_path):
            raise FileNotFoundError(f"Grid file not found: {args.grid} (resolved path: {grid_path})")
        with open(grid_path, 'r') as f:
            grid = json.load(f)

    evaluator = ComprehensiveModelEvaluator()
    evaluator.tune_model(args.model, horizon=args.horizon, param_grid=grid or None, cv_folds=args.cv_folds)


if __name__ == "__main__":
    main()


