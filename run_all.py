#!/usr/bin/env python3
"""
Multi-coin backtesting with train/test split:
- Train: 2020-01-01 ~ 2025-12-31
- Test (Prediction): 2026-01-01 ~ 2026-04-11
"""
import gc
import sys
from pathlib import Path
from datetime import datetime

import polars as pl

from src.data.loader import CryptoDataLoader
from src.data.store import ResultStore
from src.features.indicators import FeatureEngineer
from src.backtest.engine import MultiBandDCABacktester
from src.optimizer.parametric import run_optimization
from src.ml.pipeline import MLPipeline
from src.dl.pipeline import DLPipeline
from src.visualization.report import ReportGenerator

DATA_PATH = "/mnt/data/finance/cryptocurrency"
RESULTS_DB = "results/results.db"

SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT"]
TIMEFRAMES = ["5m", "15m", "1h"]

TRAIN_START = "2020-01-01"
TRAIN_END = "2025-12-31"
TEST_START = "2026-01-01"
TEST_END = "2026-04-11"

PARAM_GRID = {
    "bb_entry_sigma": [-3.0, -2.5, -2.0],
    "bb_exit_sigma": [0.5, 0.7, 1.0],
    "n_splits": [2, 3, 4],
}

FIXED_PARAMS = {
    "bb_period": 20,
    "avg_down_pct": 5.0,
    "max_hold_days": 60,
    "initial_capital": 10000.0,
}


def run_backtest_only(
    loader: CryptoDataLoader,
    symbol: str,
    timeframe: str,
    start_date: str,
    end_date: str,
    params: dict,
) -> tuple[dict, pl.DataFrame] | None:
    try:
        df = loader.load(symbol, timeframe, start_date, end_date)
        if len(df) < 1000:
            print(f"  Skipping {symbol} {timeframe}: not enough data ({len(df)} candles)")
            return None
        
        bt = MultiBandDCABacktester(**params)
        result = bt.run(df)
        trades_df = bt.get_trades_df(result)
        
        return result.to_dict(), trades_df
    except Exception as e:
        print(f"  Error in backtest: {e}")
        return None


def optimize_params(
    loader: CryptoDataLoader,
    symbol: str,
    timeframe: str,
    start_date: str,
    end_date: str,
) -> dict:
    try:
        df = loader.load(symbol, timeframe, start_date, end_date)
        if len(df) < 5000:
            print(f"  Not enough train data, using defaults")
            return {**FIXED_PARAMS, "bb_entry_sigma": -2.5, "bb_exit_sigma": 0.7, "n_splits": 2}
        
        opt_result = run_optimization(
            df,
            param_grid=PARAM_GRID,
            fixed_params=FIXED_PARAMS,
            metric="sharpe_ratio",
            method="grid",
        )
        
        best_params = {**FIXED_PARAMS, **opt_result.best_params}
        return best_params
    except Exception as e:
        print(f"  Optimization error: {e}")
        return {**FIXED_PARAMS, "bb_entry_sigma": -2.5, "bb_exit_sigma": 0.7, "n_splits": 2}


def run_ml(
    loader: CryptoDataLoader,
    fe: FeatureEngineer,
    symbol: str,
    timeframe: str,
    train_start: str,
    train_end: str,
    params: dict,
) -> dict | None:
    try:
        df = loader.load(symbol, timeframe, train_start, train_end)
        bt = MultiBandDCABacktester(**params)
        result = bt.run(df)
        trades_df = bt.get_trades_df(result)
        
        if len(trades_df) < 50:
            print(f"  Not enough trades for ML: {len(trades_df)}")
            return None
        
        ml = MLPipeline(hold_days_threshold=5.0, undersample=True)
        results = ml.run_pipeline(
            df, trades_df, fe,
            model_types=["random_forest", "logistic_regression"],
        )
        
        return {name: r.__dict__ for name, r in results.items()}
    except Exception as e:
        print(f"  ML error: {e}")
        return None
    finally:
        gc.collect()


def run_dl(
    loader: CryptoDataLoader,
    fe: FeatureEngineer,
    symbol: str,
    timeframe: str,
    train_start: str,
    train_end: str,
    params: dict,
) -> dict | None:
    try:
        df = loader.load(symbol, timeframe, train_start, train_end)
        bt = MultiBandDCABacktester(**params)
        result = bt.run(df)
        trades_df = bt.get_trades_df(result)
        
        if len(trades_df) < 50:
            print(f"  Not enough trades for DL: {len(trades_df)}")
            return None
        
        dl = DLPipeline(
            sequence_length=100,
            hold_days_threshold=5.0,
            hidden_size=64,
            num_layers=2,
            batch_size=32,
            epochs=30,
            early_stopping_patience=5,
        )
        result = dl.run_pipeline(df, trades_df, fe, model_type="lstm")
        
        return result.__dict__
    except Exception as e:
        print(f"  DL error: {e}")
        return None
    finally:
        gc.collect()


def process_symbol(
    symbol: str,
    loader: CryptoDataLoader,
    store: ResultStore,
    fe: FeatureEngineer,
) -> dict:
    print(f"\n{'='*60}")
    print(f"Processing {symbol}")
    print(f"{'='*60}")
    
    results = {}
    
    for tf in TIMEFRAMES:
        print(f"\n--- {symbol} {tf} ---")
        
        print(f"  [1/4] Optimizing on train data (2020-2025)...")
        best_params = optimize_params(loader, symbol, tf, TRAIN_START, TRAIN_END)
        print(f"  Best params: entry_σ={best_params.get('bb_entry_sigma')}, "
              f"exit_σ={best_params.get('bb_exit_sigma')}, "
              f"n_splits={best_params.get('n_splits')}")
        
        print(f"  [2/4] Running backtest on test data (2026)...")
        bt_result = run_backtest_only(loader, symbol, tf, TEST_START, TEST_END, best_params)
        
        if bt_result is None:
            print(f"  Skipping {tf}")
            continue
        
        metrics, trades_df = bt_result
        print(f"  Return: {metrics['total_return_pct']:.2f}%, "
              f"Sharpe: {metrics['sharpe_ratio']:.2f}, "
              f"Win Rate: {metrics['win_rate']:.1f}%, "
              f"Trades: {metrics['num_trades']}")
        
        print(f"  [3/4] Training ML on train data...")
        ml_results = run_ml(loader, fe, symbol, tf, TRAIN_START, TRAIN_END, best_params)
        if ml_results:
            for name, r in ml_results.items():
                print(f"    {name}: Acc={r['accuracy']:.2f}, F1={r['f1']:.3f}")
        
        print(f"  [4/4] Training DL on train data...")
        dl_result = run_dl(loader, fe, symbol, tf, TRAIN_START, TRAIN_END, best_params)
        if dl_result:
            print(f"    LSTM: Acc={dl_result['test_accuracy']:.2f}, F1={dl_result['test_f1']:.3f}")
        
        exp_id = store.create_experiment(
            symbol=symbol,
            timeframe=tf,
            experiment_type="train_test_split",
            start_date=TEST_START,
            end_date=TEST_END,
            description=f"Train 2020-2025, Test 2026 | {symbol} {tf}",
        )
        
        params_id = store.save_backtest_params(exp_id, best_params)
        store.save_backtest_result(exp_id, params_id, metrics)
        store.save_trades(exp_id, params_id, trades_df)
        
        if ml_results:
            for name, r in ml_results.items():
                store.save_ml_result(exp_id, name, r, None)
        
        if dl_result:
            store.save_dl_result(
                exp_id, "lstm",
                {"sequence_length": 100, "hidden_size": 64, "num_layers": 2},
                dl_result,
            )
        
        results[tf] = {
            "params": best_params,
            "metrics": metrics,
            "trades_df": trades_df,
            "ml": ml_results,
            "dl": dl_result,
        }
        
        gc.collect()
    
    return results


def generate_reports(all_results: dict, loader: CryptoDataLoader):
    print(f"\n{'='*60}")
    print("Generating HTML Reports")
    print(f"{'='*60}")
    
    report_gen = ReportGenerator(output_dir="docs")
    
    for symbol, tf_results in all_results.items():
        for tf, result in tf_results.items():
            if result is None:
                continue
            
            try:
                df = loader.load(symbol, tf, TEST_START, TEST_END)
                
                report_gen.create_trade_chart(
                    df=df,
                    trades_df=result["trades_df"],
                    symbol=symbol,
                    timeframe=tf,
                    metrics=result["metrics"],
                )
                print(f"  Created chart: {symbol} {tf}")
            except Exception as e:
                print(f"  Failed chart {symbol} {tf}: {e}")
    
    report_gen.create_index_page(all_results)
    print("  Created index.html")


def main():
    print("Multi-Coin DCA Backtesting System")
    print(f"Train: {TRAIN_START} ~ {TRAIN_END}")
    print(f"Test:  {TEST_START} ~ {TEST_END}")
    print(f"Symbols: {SYMBOLS}")
    print(f"Timeframes: {TIMEFRAMES}")
    
    loader = CryptoDataLoader(DATA_PATH)
    store = ResultStore(RESULTS_DB)
    fe = FeatureEngineer()
    
    all_results = {}
    
    for symbol in SYMBOLS:
        try:
            all_results[symbol] = process_symbol(symbol, loader, store, fe)
        except Exception as e:
            print(f"Failed {symbol}: {e}")
            all_results[symbol] = {}
    
    generate_reports(all_results, loader)
    
    print(f"\n{'='*60}")
    print("FINAL SUMMARY (2026 Test Results)")
    print(f"{'='*60}")
    
    for symbol, tf_results in all_results.items():
        print(f"\n{symbol}:")
        for tf, result in tf_results.items():
            if result:
                m = result["metrics"]
                print(f"  {tf}: Return={m['total_return_pct']:.2f}%, "
                      f"Sharpe={m['sharpe_ratio']:.2f}, "
                      f"WinRate={m['win_rate']:.1f}%, "
                      f"Trades={m['num_trades']}")


if __name__ == "__main__":
    main()
