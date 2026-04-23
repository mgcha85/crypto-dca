#!/usr/bin/env python3
import argparse
from pathlib import Path

from src.runner import run_full_experiment, ExperimentRunner, ExperimentConfig
from src.visualization.report import generate_report_html, generate_index_html
from src.data.loader import CryptoDataLoader
from src.backtest.engine import MultiBandDCABacktester


def run_all_coins(
    data_path: str = "/mnt/data/finance/cryptocurrency",
    timeframes: list[str] | None = None,
    start_date: str = "2024-01-01",
    end_date: str = "2024-06-30",
    output_dir: str = "docs",
):
    timeframes = timeframes or ["5m", "15m", "1h"]
    loader = CryptoDataLoader(data_path)
    symbols = loader.get_available_symbols()

    print(f"Available symbols: {symbols}")
    print(f"Timeframes: {timeframes}")
    print(f"Period: {start_date} to {end_date}")

    all_results = {}

    for symbol in symbols:
        print(f"\n{'#' * 60}")
        print(f"Processing {symbol}")
        print(f"{'#' * 60}")

        try:
            results = run_full_experiment(
                symbol=symbol,
                timeframes=timeframes,
                start_date=start_date,
                end_date=end_date,
                data_path=data_path,
            )

            for tf, result in results.items():
                key = f"{symbol.lower()}_{tf}"
                all_results[key] = {
                    "backtest": result.best_backtest,
                    "symbol": symbol,
                    "timeframe": tf,
                }

                df = loader.load(symbol, tf, start_date, end_date)
                bt = MultiBandDCABacktester(**result.optimization.best_params)
                bt_result = bt.run(df)
                trades_df = bt.get_trades_df(bt_result)

                ml_results_dict = None
                if result.ml_results:
                    ml_results_dict = {
                        name: {
                            "accuracy": r.accuracy,
                            "f1": r.f1,
                            "auc_roc": r.auc_roc,
                        }
                        for name, r in result.ml_results.items()
                    }

                dl_result_dict = None
                if result.dl_result:
                    dl_result_dict = {
                        "test_accuracy": result.dl_result.test_accuracy,
                        "test_f1": result.dl_result.test_f1,
                        "epochs_trained": result.dl_result.epochs_trained,
                    }

                generate_report_html(
                    symbol=symbol,
                    timeframe=tf,
                    backtest_results=result.best_backtest,
                    trades_df=trades_df,
                    df=df,
                    ml_results=ml_results_dict,
                    dl_result=dl_result_dict,
                    output_dir=output_dir,
                )

        except Exception as e:
            print(f"Failed for {symbol}: {e}")
            continue

    generate_index_html(all_results, output_dir=output_dir)
    print(f"\n\nAll reports generated in {output_dir}/")

    return all_results


def main():
    parser = argparse.ArgumentParser(description="Multi-Band DCA Backtesting System")
    parser.add_argument("--data-path", default="/mnt/data/finance/cryptocurrency")
    parser.add_argument("--symbol", default=None, help="Single symbol to process (or all if not specified)")
    parser.add_argument("--timeframes", nargs="+", default=["5m", "15m", "1h"])
    parser.add_argument("--start-date", default="2024-01-01")
    parser.add_argument("--end-date", default="2024-06-30")
    parser.add_argument("--output-dir", default="docs")

    args = parser.parse_args()

    if args.symbol:
        results = run_full_experiment(
            symbol=args.symbol,
            timeframes=args.timeframes,
            start_date=args.start_date,
            end_date=args.end_date,
            data_path=args.data_path,
        )

        print("\n" + "=" * 60)
        print("SUMMARY")
        print("=" * 60)
        for tf, result in results.items():
            print(f"\n{tf}:")
            print(f"  Return: {result.best_backtest['total_return_pct']:.2f}%")
            print(f"  Sharpe: {result.best_backtest['sharpe_ratio']:.4f}")
            print(f"  Win Rate: {result.best_backtest['win_rate']:.2f}%")
    else:
        run_all_coins(
            data_path=args.data_path,
            timeframes=args.timeframes,
            start_date=args.start_date,
            end_date=args.end_date,
            output_dir=args.output_dir,
        )


if __name__ == "__main__":
    main()
