from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Optional
import json

import numpy as np
import polars as pl
from tqdm import tqdm

from src.backtest.engine import MultiBandDCABacktester
from src.data.store import ResultStore


@dataclass
class OptimizationResult:
    best_params: dict
    best_score: float
    all_results: pl.DataFrame
    metric: str


class ParametricOptimizer:
    def __init__(
        self,
        df: pl.DataFrame,
        metric: str = "sharpe_ratio",
        store: ResultStore | None = None,
    ):
        self.df = df
        self.metric = metric
        self.store = store

    def _run_single(self, params: dict) -> dict:
        bt = MultiBandDCABacktester(**params)
        result = bt.run(self.df)
        metrics = result.to_dict()
        metrics.update(params)
        return metrics

    def grid_search(
        self,
        param_grid: dict,
        fixed_params: Optional[dict] = None,
    ) -> OptimizationResult:
        fixed_params = fixed_params or {}

        param_names = list(param_grid.keys())
        param_values = [param_grid[k] for k in param_names]
        combinations = list(product(*param_values))

        results = []
        for combo in tqdm(combinations, desc="Grid Search"):
            params = dict(zip(param_names, combo))
            params.update(fixed_params)
            try:
                metrics = self._run_single(params)
                results.append(metrics)
            except Exception as e:
                print(f"Error with params {params}: {e}")
                continue

        results_df = pl.DataFrame(results)

        if results_df.is_empty():
            raise ValueError("No valid results from optimization")

        best_idx = results_df[self.metric].arg_max()
        best_row = results_df.row(best_idx, named=True)
        best_params = {k: best_row[k] for k in param_names}
        best_params.update(fixed_params)

        return OptimizationResult(
            best_params=best_params,
            best_score=best_row[self.metric],
            all_results=results_df,
            metric=self.metric,
        )

    def random_search(
        self,
        param_ranges: dict,
        n_trials: int = 100,
        fixed_params: Optional[dict] = None,
    ) -> OptimizationResult:
        fixed_params = fixed_params or {}

        results = []
        for _ in tqdm(range(n_trials), desc="Random Search"):
            params = {}
            for name, (low, high) in param_ranges.items():
                if isinstance(low, int) and isinstance(high, int):
                    params[name] = np.random.randint(low, high + 1)
                else:
                    params[name] = np.random.uniform(low, high)

            params.update(fixed_params)
            try:
                metrics = self._run_single(params)
                results.append(metrics)
            except Exception:
                continue

        results_df = pl.DataFrame(results)

        if results_df.is_empty():
            raise ValueError("No valid results from optimization")

        best_idx = results_df[self.metric].arg_max()
        best_row = results_df.row(best_idx, named=True)
        best_params = {k: best_row[k] for k in param_ranges.keys()}
        best_params.update(fixed_params)

        return OptimizationResult(
            best_params=best_params,
            best_score=best_row[self.metric],
            all_results=results_df,
            metric=self.metric,
        )

    def save_results(self, result: OptimizationResult, path: str | Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        result.all_results.write_csv(path / "all_results.csv")

        with open(path / "best_params.json", "w") as f:
            json.dump(
                {
                    "best_params": result.best_params,
                    "best_score": result.best_score,
                    "metric": result.metric,
                },
                f,
                indent=2,
                default=str,
            )


def run_optimization(
    df: pl.DataFrame,
    param_ranges: Optional[dict] = None,
    param_grid: Optional[dict] = None,
    fixed_params: Optional[dict] = None,
    metric: str = "sharpe_ratio",
    n_trials: int = 100,
    method: str = "random",
    store: ResultStore | None = None,
) -> OptimizationResult:
    optimizer = ParametricOptimizer(df, metric=metric, store=store)

    if method == "grid" and param_grid:
        return optimizer.grid_search(param_grid, fixed_params)
    elif method == "random" and param_ranges:
        return optimizer.random_search(param_ranges, n_trials, fixed_params)
    else:
        raise ValueError(f"Invalid method '{method}' or missing parameters")
