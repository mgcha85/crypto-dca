from dataclasses import dataclass
from pathlib import Path
from datetime import datetime
import json

import polars as pl

from src.data.loader import CryptoDataLoader
from src.data.store import ResultStore
from src.features.indicators import FeatureEngineer
from src.backtest.engine import MultiBandDCABacktester
from src.optimizer.parametric import run_optimization, OptimizationResult
from src.ml.pipeline import MLPipeline, MLResult
from src.dl.pipeline import DLPipeline, DLResult


@dataclass
class ExperimentConfig:
    symbol: str
    timeframe: str
    start_date: str
    end_date: str
    param_grid: dict
    fixed_params: dict
    optimization_metric: str = "sharpe_ratio"
    run_ml: bool = True
    run_dl: bool = True


@dataclass
class ExperimentResult:
    config: ExperimentConfig
    optimization: OptimizationResult | None
    best_backtest: dict
    ml_results: dict[str, MLResult] | None
    dl_result: DLResult | None
    trades_df: pl.DataFrame


class ExperimentRunner:
    def __init__(
        self,
        data_path: str | Path,
        results_db: str | Path = "results/results.db",
    ):
        self.loader = CryptoDataLoader(data_path)
        self.store = ResultStore(results_db)
        self.fe = FeatureEngineer()

    def run_single_timeframe(
        self,
        config: ExperimentConfig,
    ) -> ExperimentResult:
        print(f"\n{'='*60}")
        print(f"Running experiment: {config.symbol} {config.timeframe}")
        print(f"Period: {config.start_date} to {config.end_date}")
        print(f"{'='*60}")

        df = self.loader.load(
            config.symbol,
            config.timeframe,
            config.start_date,
            config.end_date,
        )
        print(f"Loaded {len(df)} candles")

        print("\n--- Optimization ---")
        opt_result = run_optimization(
            df,
            param_grid=config.param_grid,
            fixed_params=config.fixed_params,
            metric=config.optimization_metric,
            method="grid",
        )
        print(f"Best params: {opt_result.best_params}")
        print(f"Best {config.optimization_metric}: {opt_result.best_score:.4f}")

        print("\n--- Best Backtest ---")
        best_params = {**config.fixed_params}
        for k, v in opt_result.best_params.items():
            if k in config.param_grid:
                best_params[k] = v

        bt = MultiBandDCABacktester(**best_params)
        bt_result = bt.run(df)
        trades_df = bt.get_trades_df(bt_result)

        print(f"Total Return: {bt_result.total_return_pct:.2f}%")
        print(f"Win Rate: {bt_result.win_rate:.2f}%")
        print(f"Sharpe: {bt_result.sharpe_ratio:.4f}")
        print(f"Max DD: {bt_result.max_drawdown_pct:.2f}%")
        print(f"Trades: {bt_result.num_trades}")

        ml_results = None
        if config.run_ml and len(trades_df) >= 30:
            print("\n--- ML Training ---")
            ml = MLPipeline(hold_days_threshold=5.0, undersample=True)
            try:
                ml_results = ml.run_pipeline(
                    df, trades_df, self.fe,
                    model_types=["random_forest", "logistic_regression"],
                )
            except Exception as e:
                print(f"ML failed: {e}")

        dl_result = None
        if config.run_dl and len(trades_df) >= 30:
            print("\n--- DL Training ---")
            dl = DLPipeline(
                sequence_length=100,
                hold_days_threshold=5.0,
                hidden_size=64,
                num_layers=2,
                batch_size=16,
                epochs=30,
                early_stopping_patience=5,
            )
            try:
                dl_result = dl.run_pipeline(df, trades_df, self.fe, model_type="lstm")
            except Exception as e:
                print(f"DL failed: {e}")

        exp_id = self.store.create_experiment(
            symbol=config.symbol,
            timeframe=config.timeframe,
            experiment_type="full_pipeline",
            start_date=config.start_date,
            end_date=config.end_date,
            description=f"Optimization + ML/DL for {config.symbol} {config.timeframe}",
        )

        params_id = self.store.save_backtest_params(exp_id, best_params)
        self.store.save_backtest_result(exp_id, params_id, bt_result.to_dict())
        self.store.save_trades(exp_id, params_id, trades_df)

        if opt_result:
            self.store.save_optimization_result(
                exp_id, params_id, config.optimization_metric,
                opt_result.best_score, len(opt_result.all_results), "grid",
            )

        if ml_results:
            for model_name, result in ml_results.items():
                self.store.save_ml_result(
                    exp_id, model_name,
                    {
                        "accuracy": result.accuracy,
                        "precision": result.precision,
                        "recall": result.recall,
                        "f1": result.f1,
                        "auc_roc": result.auc_roc,
                    },
                    json.dumps(result.feature_importance) if result.feature_importance else None,
                )

        if dl_result:
            self.store.save_dl_result(
                exp_id, dl_result.model_name,
                {"sequence_length": 100, "hidden_size": 64, "num_layers": 2},
                {
                    "epochs_trained": dl_result.epochs_trained,
                    "best_val_loss": dl_result.best_val_loss,
                    "test_accuracy": dl_result.test_accuracy,
                    "test_f1": dl_result.test_f1,
                },
            )

        return ExperimentResult(
            config=config,
            optimization=opt_result,
            best_backtest=bt_result.to_dict(),
            ml_results=ml_results,
            dl_result=dl_result,
            trades_df=trades_df,
        )

    def run_multi_timeframe(
        self,
        symbol: str,
        timeframes: list[str],
        start_date: str,
        end_date: str,
        param_grid: dict,
        fixed_params: dict,
    ) -> dict[str, ExperimentResult]:
        results = {}

        for tf in timeframes:
            config = ExperimentConfig(
                symbol=symbol,
                timeframe=tf,
                start_date=start_date,
                end_date=end_date,
                param_grid=param_grid,
                fixed_params=fixed_params,
            )

            try:
                results[tf] = self.run_single_timeframe(config)
            except Exception as e:
                print(f"Failed for {tf}: {e}")

        return results


def run_full_experiment(
    symbol: str = "BTCUSDT",
    timeframes: list[str] | None = None,
    start_date: str = "2024-01-01",
    end_date: str = "2024-12-31",
    data_path: str = "/mnt/data/finance/cryptocurrency",
) -> dict[str, ExperimentResult]:
    timeframes = timeframes or ["5m", "15m", "1h"]

    param_grid = {
        "bb_entry_sigma": [-3.0, -2.5, -2.0],
        "bb_exit_sigma": [0.5, 0.7, 1.0],
        "n_splits": [2, 3, 4],
        "avg_down_pct": [5.0, 10.0, 15.0],
    }

    fixed_params = {
        "bb_period": 20,
        "max_hold_days": 60,
        "initial_capital": 10000.0,
        "position_size_pct": 10.0,
    }

    runner = ExperimentRunner(data_path)
    return runner.run_multi_timeframe(
        symbol, timeframes, start_date, end_date, param_grid, fixed_params,
    )


if __name__ == "__main__":
    results = run_full_experiment(
        symbol="BTCUSDT",
        timeframes=["5m", "15m", "1h"],
        start_date="2024-01-01",
        end_date="2024-06-30",
    )

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for tf, result in results.items():
        print(f"\n{tf}:")
        print(f"  Return: {result.best_backtest['total_return_pct']:.2f}%")
        print(f"  Sharpe: {result.best_backtest['sharpe_ratio']:.4f}")
        print(f"  Win Rate: {result.best_backtest['win_rate']:.2f}%")
