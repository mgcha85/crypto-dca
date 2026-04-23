#!/usr/bin/env python3
"""
Multi-coin backtesting WITH LSTM filter comparison.
Train: 2020-2025, Test: 2026
Compares results with and without LSTM entry filter.
"""
import gc
import torch
from pathlib import Path

import polars as pl

from src.data.loader import CryptoDataLoader
from src.features.indicators import FeatureEngineer
from src.backtest.engine import MultiBandDCABacktester, MLFilteredBacktester
from src.dl.pipeline import DLPipeline, LSTMModel
from src.visualization.report import ReportGenerator

DATA_PATH = "/mnt/data/finance/cryptocurrency"
SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT"]
TIMEFRAMES = ["5m", "15m", "1h"]

TRAIN_START = "2020-01-01"
TRAIN_END = "2025-12-31"
TEST_START = "2026-01-01"
TEST_END = "2026-04-11"

PARAMS = {
    "bb_entry_sigma": -2.0,
    "bb_exit_sigma": 1.0,
    "n_splits": 2,
    "avg_down_pct": 5.0,
    "max_hold_days": 60,
    "initial_capital": 10000.0,
}


def train_lstm(loader, fe, symbol, timeframe):
    train_df = loader.load(symbol, timeframe, TRAIN_START, TRAIN_END)
    if len(train_df) < 10000:
        return None, None
    
    bt = MultiBandDCABacktester(**PARAMS)
    result = bt.run(train_df)
    trades_df = bt.get_trades_df(result)
    
    if len(trades_df) < 50:
        return None, None
    
    dl = DLPipeline(
        sequence_length=100,
        hold_days_threshold=3.0,
        hidden_size=64,
        num_layers=2,
        batch_size=32,
        epochs=30,
        early_stopping_patience=10,
    )
    
    try:
        sequences, features, labels = dl.prepare_sequences(train_df, trades_df, fe)
        if len(labels) < 50:
            return None, None
        
        train_loader, val_loader, _ = dl.create_dataloaders(sequences, features, labels)
        
        model = LSTMModel(
            seq_features=sequences.shape[2],
            static_features=features.shape[1],
            hidden_size=64,
            num_layers=2,
        )
        
        class_weights = torch.FloatTensor([
            labels.sum() / len(labels),
            (len(labels) - labels.sum()) / len(labels)
        ])
        
        model, _, _, _ = dl.train_model(model, train_loader, val_loader, class_weights)
        
        return model, dl
    except Exception as e:
        print(f"  LSTM training failed: {e}")
        return None, None


def run_comparison(loader, fe, symbol, timeframe, model, dl):
    test_df = loader.load(symbol, timeframe, TEST_START, TEST_END)
    if len(test_df) < 1000:
        return None
    
    bt_no_filter = MultiBandDCABacktester(**PARAMS)
    result_no_filter = bt_no_filter.run(test_df)
    trades_no_filter = bt_no_filter.get_trades_df(result_no_filter)
    
    result_filtered = None
    trades_filtered = None
    
    if model is not None and dl is not None:
        test_featured = fe.add_basic_features(test_df)
        test_featured = fe.add_entry_exit_bands(test_featured)
        
        bt_filtered = MLFilteredBacktester(
            model=model,
            seq_scaler=dl.seq_scaler,
            feature_scaler=dl.feature_scaler,
            seq_cols=dl.seq_cols,
            feature_cols=dl.feature_cols,
            sequence_length=100,
            **PARAMS,
        )
        result_filtered = bt_filtered.run_with_filter(test_df, test_featured)
        trades_filtered = bt_filtered.get_trades_df(result_filtered)
    
    return {
        "df": test_df,
        "no_filter": {
            "result": result_no_filter,
            "trades": trades_no_filter,
            "metrics": result_no_filter.to_dict(),
        },
        "with_filter": {
            "result": result_filtered,
            "trades": trades_filtered,
            "metrics": result_filtered.to_dict() if result_filtered else None,
        } if result_filtered else None,
    }


def main():
    print("=" * 60)
    print("Multi-Coin LSTM Filter Comparison")
    print(f"Train: {TRAIN_START} ~ {TRAIN_END}")
    print(f"Test:  {TEST_START} ~ {TEST_END}")
    print("=" * 60)
    
    loader = CryptoDataLoader(DATA_PATH)
    fe = FeatureEngineer()
    
    all_results = {}
    
    for symbol in SYMBOLS:
        print(f"\n{'='*50}")
        print(f"Processing {symbol}")
        print("=" * 50)
        
        all_results[symbol] = {}
        
        for tf in TIMEFRAMES:
            print(f"\n--- {symbol} {tf} ---")
            
            print("  Training LSTM...")
            model, dl = train_lstm(loader, fe, symbol, tf)
            
            if model:
                print("  LSTM trained successfully")
            else:
                print("  LSTM training skipped/failed")
            
            print("  Running backtest comparison...")
            comparison = run_comparison(loader, fe, symbol, tf, model, dl)
            
            if comparison is None:
                print("  Skipped - not enough data")
                continue
            
            no_filter = comparison["no_filter"]["metrics"]
            print(f"  No filter:   Return={no_filter['total_return_pct']:.2f}%, "
                  f"Trades={no_filter['num_trades']}, WinRate={no_filter['win_rate']:.1f}%")
            
            if comparison["with_filter"]:
                with_filter = comparison["with_filter"]["metrics"]
                print(f"  With filter: Return={with_filter['total_return_pct']:.2f}%, "
                      f"Trades={with_filter['num_trades']}, WinRate={with_filter['win_rate']:.1f}%")
            
            all_results[symbol][tf] = comparison
            gc.collect()
    
    print("\n" + "=" * 60)
    print("FINAL SUMMARY")
    print("=" * 60)
    print(f"\n{'Symbol':<10} {'TF':<5} {'No Filter':<15} {'With LSTM':<15} {'Diff':<10}")
    print("-" * 55)
    
    for symbol, tf_results in all_results.items():
        for tf, data in tf_results.items():
            no_ret = data["no_filter"]["metrics"]["total_return_pct"]
            
            if data["with_filter"]:
                with_ret = data["with_filter"]["metrics"]["total_return_pct"]
                diff = with_ret - no_ret
                print(f"{symbol:<10} {tf:<5} {no_ret:>+.2f}%{'':>8} {with_ret:>+.2f}%{'':>8} {diff:>+.2f}%")
            else:
                print(f"{symbol:<10} {tf:<5} {no_ret:>+.2f}%{'':>8} {'N/A':<15}")
    
    return all_results


if __name__ == "__main__":
    results = main()
