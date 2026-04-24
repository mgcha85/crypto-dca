#!/usr/bin/env python3
"""
Monthly Walk-Forward Cross-Validation for BTCUSDT.
Train: all data before test month
Test: single month
Period: 2021-01 ~ 2026-04
"""
import gc
from datetime import datetime
from pathlib import Path

import polars as pl
import torch

from src.data.loader import CryptoDataLoader
from src.features.indicators import FeatureEngineer
from src.backtest.engine import MultiBandDCABacktester, MLFilteredBacktester
from src.dl.pipeline import DLPipeline, LSTMModel


DATA_PATH = "/mnt/data/finance/cryptocurrency"
SYMBOL = "BTCUSDT"
TIMEFRAME = "1h"

START_YEAR = 2021
START_MONTH = 1
END_YEAR = 2026
END_MONTH = 4

PARAMS = {
    "bb_entry_sigma": -2.0,
    "bb_exit_sigma": 1.0,
    "n_splits": 2,
    "avg_down_pct": 5.0,
    "max_hold_days": 60,
    "initial_capital": 10000.0,
}


def get_month_range(year: int, month: int) -> tuple[str, str]:
    """Get start and end date for a given month."""
    start = f"{year}-{month:02d}-01"
    if month == 12:
        end = f"{year + 1}-01-01"
    else:
        end = f"{year}-{month + 1:02d}-01"
    return start, end


def train_lstm_for_month(
    loader: CryptoDataLoader,
    fe: FeatureEngineer,
    test_year: int,
    test_month: int,
) -> tuple:
    """Train LSTM on all data before the test month."""
    train_start = "2020-01-01"
    if test_month == 1:
        train_end = f"{test_year - 1}-12-31"
    else:
        train_end = f"{test_year}-{test_month - 1:02d}-28"
    
    train_df = loader.load(SYMBOL, TIMEFRAME, train_start, train_end)
    if len(train_df) < 5000:
        print(f"    Not enough training data: {len(train_df)} rows")
        return None, None
    
    bt = MultiBandDCABacktester(**PARAMS)
    result = bt.run(train_df)
    trades_df = bt.get_trades_df(result)
    
    if len(trades_df) < 30:
        print(f"    Not enough trades for training: {len(trades_df)}")
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
        if len(labels) < 30:
            print(f"    Not enough samples: {len(labels)}")
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
        print(f"    LSTM training failed: {e}")
        return None, None


def run_month_backtest(
    loader: CryptoDataLoader,
    fe: FeatureEngineer,
    year: int,
    month: int,
    model,
    dl: DLPipeline,
) -> dict:
    """Run backtest for a single month."""
    test_start, test_end = get_month_range(year, month)
    
    test_df = loader.load(SYMBOL, TIMEFRAME, test_start, test_end)
    if len(test_df) < 100:
        return None
    
    first_price = test_df["close"][0]
    last_price = test_df["close"][-1]
    market_return = ((last_price - first_price) / first_price) * 100
    
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
        "year": year,
        "month": month,
        "market_return": market_return,
        "start_price": first_price,
        "end_price": last_price,
        "no_filter": {
            "return": result_no_filter.total_return_pct,
            "trades": result_no_filter.num_trades,
            "win_rate": result_no_filter.win_rate,
            "sharpe": result_no_filter.sharpe_ratio,
            "max_dd": result_no_filter.max_drawdown_pct,
        },
        "with_filter": {
            "return": result_filtered.total_return_pct if result_filtered else None,
            "trades": result_filtered.num_trades if result_filtered else None,
            "win_rate": result_filtered.win_rate if result_filtered else None,
            "sharpe": result_filtered.sharpe_ratio if result_filtered else None,
            "max_dd": result_filtered.max_drawdown_pct if result_filtered else None,
        } if result_filtered else None,
    }


def generate_html_report(results: list[dict]) -> str:
    no_filter_cumulative = 100
    with_filter_cumulative = 100
    
    rows = []
    for r in results:
        no_filter_cumulative *= (1 + r["no_filter"]["return"] / 100)
        if r["with_filter"]:
            with_filter_cumulative *= (1 + r["with_filter"]["return"] / 100)
        
        month_str = f"{r['year']}-{r['month']:02d}"
        no_ret = r["no_filter"]["return"]
        no_class = "positive" if no_ret >= 0 else "negative"
        if r["with_filter"]:
            with_ret = r["with_filter"]["return"]
            with_class = "positive" if with_ret >= 0 else "negative"
            diff = with_ret - no_ret
            diff_class = "positive" if diff >= 0 else "negative"
            
            row = f"""
        <tr>
            <td>{month_str}</td>
            <td class="{no_class}">{no_ret:+.2f}%</td>
            <td class="{with_class}">{with_ret:+.2f}%</td>
            <td class="{diff_class}">{diff:+.2f}%</td>
            <td>{r["no_filter"]["trades"]} / {r["with_filter"]["trades"]}</td>
            <td>{r["market_return"]:+.2f}%</td>
        </tr>"""
        else:
            row = f"""
        <tr>
            <td>{month_str}</td>
            <td class="{no_class}">{no_ret:+.2f}%</td>
            <td>N/A</td>
            <td>N/A</td>
            <td>{r["no_filter"]["trades"]} / -</td>
            <td>{r["market_return"]:+.2f}%</td>
        </tr>"""
        rows.append(row)
    
    table_rows = "\n".join(rows)
    valid_results = [r for r in results if r["with_filter"] is not None]
    
    total_no_filter = no_filter_cumulative - 100
    total_with_filter = with_filter_cumulative - 100
    lstm_better = sum(1 for r in valid_results if r["with_filter"]["return"] > r["no_filter"]["return"])
    avg_no_filter = sum(r["no_filter"]["return"] for r in results) / len(results)
    avg_with_filter = sum(r["with_filter"]["return"] for r in valid_results) / len(valid_results) if valid_results else 0
    market_cumulative = 100
    for r in results:
        market_cumulative *= (1 + r["market_return"] / 100)
    total_market = market_cumulative - 100
    
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>BTCUSDT Monthly Walk-Forward CV Results</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
            max-width: 1400px;
            margin: 0 auto;
            padding: 20px;
            background: #f5f5f5;
        }}
        h1, h2, h3 {{ color: #333; }}
        .card {{
            background: white;
            border-radius: 8px;
            padding: 20px;
            margin-bottom: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
        }}
        th, td {{
            padding: 10px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }}
        th {{ background: #f5f5f5; font-weight: 600; }}
        tr:hover {{ background: #f9f9f9; }}
        .positive {{ color: #4CAF50; font-weight: bold; }}
        .negative {{ color: #f44336; font-weight: bold; }}
        .summary-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
            gap: 15px;
            margin-bottom: 20px;
        }}
        .summary-item {{
            text-align: center;
            padding: 20px;
            background: #f8f9fa;
            border-radius: 8px;
        }}
        .summary-value {{
            font-size: 24px;
            font-weight: bold;
        }}
        .summary-label {{
            color: #666;
            font-size: 13px;
            margin-top: 5px;
        }}
        a {{ color: #2196F3; text-decoration: none; }}
        a:hover {{ text-decoration: underline; }}
    </style>
</head>
<body>
    <h1>BTCUSDT Monthly Walk-Forward CV Results</h1>
    <p>Period: 2021-01 ~ 2026-04 | Timeframe: 1h | Walk-Forward: Train on all prior data, test on each month</p>
    
    <div class="card">
        <h2>Summary</h2>
        <div class="summary-grid">
            <div class="summary-item">
                <div class="summary-value {'positive' if total_no_filter >= 0 else 'negative'}">{total_no_filter:+.1f}%</div>
                <div class="summary-label">No Filter<br>Cumulative Return</div>
            </div>
            <div class="summary-item">
                <div class="summary-value {'positive' if total_with_filter >= 0 else 'negative'}">{total_with_filter:+.1f}%</div>
                <div class="summary-label">With LSTM<br>Cumulative Return</div>
            </div>
            <div class="summary-item">
                <div class="summary-value {'positive' if total_market >= 0 else 'negative'}">{total_market:+.1f}%</div>
                <div class="summary-label">Buy & Hold<br>Cumulative Return</div>
            </div>
            <div class="summary-item">
                <div class="summary-value">{lstm_better}/{len(valid_results)}</div>
                <div class="summary-label">Months LSTM<br>Outperformed</div>
            </div>
            <div class="summary-item">
                <div class="summary-value">{avg_no_filter:+.2f}%</div>
                <div class="summary-label">Avg Monthly<br>No Filter</div>
            </div>
            <div class="summary-item">
                <div class="summary-value">{avg_with_filter:+.2f}%</div>
                <div class="summary-label">Avg Monthly<br>With LSTM</div>
            </div>
        </div>
    </div>

    <div class="card">
        <h2>Monthly Results</h2>
        <table>
            <tr>
                <th>Month</th>
                <th>No Filter</th>
                <th>With LSTM</th>
                <th>Difference</th>
                <th>Trades (No/LSTM)</th>
                <th>BTC Return</th>
            </tr>
{table_rows}
        </table>
    </div>

    <div class="card">
        <h2>Methodology</h2>
        <ul>
            <li><strong>Walk-Forward CV:</strong> For each month, train LSTM on ALL data from 2020-01-01 to the end of the previous month</li>
            <li><strong>No Look-Ahead:</strong> Model never sees future data during training</li>
            <li><strong>Strategy:</strong> BB -2.0σ entry, +1.0σ exit, 2-split DCA, 5% avg-down, 60-day max hold</li>
            <li><strong>LSTM Filter:</strong> Predict if hold time will be >3 days (bad entry) or ≤3 days (good entry)</li>
        </ul>
    </div>

    <p><a href="index.html">← Back to Main Results</a></p>
</body>
</html>"""
    
    return html


def main():
    print("=" * 60)
    print("BTCUSDT Monthly Walk-Forward Cross-Validation")
    print(f"Period: {START_YEAR}-{START_MONTH:02d} ~ {END_YEAR}-{END_MONTH:02d}")
    print("=" * 60)
    
    loader = CryptoDataLoader(DATA_PATH)
    fe = FeatureEngineer()
    
    results = []
    
    year = START_YEAR
    month = START_MONTH
    
    while (year < END_YEAR) or (year == END_YEAR and month <= END_MONTH):
        print(f"\n--- {year}-{month:02d} ---")
        print("  Training LSTM...")
        model, dl = train_lstm_for_month(loader, fe, year, month)
        
        if model:
            print("  LSTM trained successfully")
        else:
            print("  LSTM training skipped")
        
        print("  Running backtest...")
        result = run_month_backtest(loader, fe, year, month, model, dl)
        
        if result:
            results.append(result)
            no_ret = result["no_filter"]["return"]
            mkt_ret = result["market_return"]
            
            if result["with_filter"]:
                with_ret = result["with_filter"]["return"]
                print(f"  No filter: {no_ret:+.2f}%, With LSTM: {with_ret:+.2f}%, BTC: {mkt_ret:+.2f}%")
            else:
                print(f"  No filter: {no_ret:+.2f}%, BTC: {mkt_ret:+.2f}%")
        else:
            print("  Skipped - not enough data")
        
        del model, dl
        gc.collect()
        month += 1
        if month > 12:
            month = 1
            year += 1
    
    print("\n" + "=" * 60)
    print("Generating HTML report...")
    html = generate_html_report(results)
    
    output_path = Path("docs/btcusdt_monthly_cv.html")
    output_path.write_text(html)
    print(f"Saved to {output_path}")
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    valid = [r for r in results if r["with_filter"]]
    lstm_better = sum(1 for r in valid if r["with_filter"]["return"] > r["no_filter"]["return"])
    
    print(f"Total months: {len(results)}")
    print(f"LSTM trained months: {len(valid)}")
    print(f"LSTM outperformed: {lstm_better}/{len(valid)}")
    
    return results


if __name__ == "__main__":
    results = main()
