from pathlib import Path
from datetime import datetime
import json

import polars as pl
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.data.store import ResultStore
from src.data.loader import CryptoDataLoader
from src.backtest.engine import MultiBandDCABacktester
from src.features.indicators import FeatureEngineer


def create_trade_chart(
    df: pl.DataFrame,
    trades_df: pl.DataFrame,
    position_ids: list[int],
    title: str = "Trade Example",
    padding_candles: int = 50,
) -> go.Figure:
    selected_trades = trades_df.filter(pl.col("position_id").is_in(position_ids))

    if selected_trades.is_empty():
        raise ValueError("No trades found for given position_ids")

    first_entry = selected_trades["entry_time"].min()
    last_exit = selected_trades["exit_time"].max()

    df_filtered = df.filter(
        (pl.col("datetime") >= first_entry - pl.duration(minutes=padding_candles * 5))
        & (pl.col("datetime") <= last_exit + pl.duration(minutes=padding_candles * 5))
    )

    fe = FeatureEngineer()
    df_featured = fe.add_basic_features(df_filtered)
    df_featured = fe.add_entry_exit_bands(df_featured, entry_sigma=-2.5, exit_sigma=0.7)

    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.7, 0.3],
        subplot_titles=["Price & Bollinger Bands", "RSI"],
    )

    fig.add_trace(
        go.Candlestick(
            x=df_featured["datetime"].to_list(),
            open=df_featured["open"].to_list(),
            high=df_featured["high"].to_list(),
            low=df_featured["low"].to_list(),
            close=df_featured["close"].to_list(),
            name="OHLC",
        ),
        row=1, col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=df_featured["datetime"].to_list(),
            y=df_featured["entry_band"].to_list(),
            mode="lines",
            name="Entry Band (-2.5σ)",
            line=dict(color="green", dash="dash"),
        ),
        row=1, col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=df_featured["datetime"].to_list(),
            y=df_featured["exit_band"].to_list(),
            mode="lines",
            name="Exit Band (+0.7σ)",
            line=dict(color="red", dash="dash"),
        ),
        row=1, col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=df_featured["datetime"].to_list(),
            y=df_featured["bb_middle"].to_list(),
            mode="lines",
            name="BB Middle",
            line=dict(color="gray", width=1),
        ),
        row=1, col=1,
    )

    entry_times = selected_trades["entry_time"].to_list()
    entry_prices = selected_trades["entry_price"].to_list()
    entry_nums = selected_trades["entry_num"].to_list()

    fig.add_trace(
        go.Scatter(
            x=entry_times,
            y=entry_prices,
            mode="markers+text",
            name="Buy",
            marker=dict(symbol="triangle-up", size=15, color="green"),
            text=[f"B{n}" for n in entry_nums],
            textposition="bottom center",
        ),
        row=1, col=1,
    )

    exit_trades = selected_trades.unique(subset=["position_id"])
    exit_times = exit_trades["exit_time"].to_list()
    exit_prices = exit_trades["exit_price"].to_list()

    fig.add_trace(
        go.Scatter(
            x=exit_times,
            y=exit_prices,
            mode="markers+text",
            name="Sell",
            marker=dict(symbol="triangle-down", size=15, color="red"),
            text=["S" for _ in exit_times],
            textposition="top center",
        ),
        row=1, col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=df_featured["datetime"].to_list(),
            y=df_featured["rsi"].to_list(),
            mode="lines",
            name="RSI",
            line=dict(color="purple"),
        ),
        row=2, col=1,
    )

    fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)

    fig.update_layout(
        title=title,
        xaxis_rangeslider_visible=False,
        height=800,
        showlegend=True,
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
    )

    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="RSI", row=2, col=1)

    return fig


def create_equity_chart(
    equity_curve: pl.Series,
    title: str = "Equity Curve",
) -> go.Figure:
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            y=equity_curve.to_list(),
            mode="lines",
            name="Equity",
            line=dict(color="blue", width=2),
        )
    )

    fig.update_layout(
        title=title,
        xaxis_title="Candles",
        yaxis_title="Equity ($)",
        height=400,
    )

    return fig


def create_optimization_heatmap(
    results_df: pl.DataFrame,
    x_col: str,
    y_col: str,
    z_col: str = "sharpe_ratio",
    title: str = "Optimization Heatmap",
) -> go.Figure:
    pivot = results_df.pivot(
        on=x_col,
        index=y_col,
        values=z_col,
        aggregate_function="mean",
    )

    x_vals = [c for c in pivot.columns if c != y_col]
    y_vals = pivot[y_col].to_list()
    z_vals = pivot.select(x_vals).to_numpy()

    fig = go.Figure(data=go.Heatmap(
        x=x_vals,
        y=y_vals,
        z=z_vals,
        colorscale="RdYlGn",
        colorbar=dict(title=z_col),
    ))

    fig.update_layout(
        title=title,
        xaxis_title=x_col,
        yaxis_title=y_col,
        height=500,
    )

    return fig


def generate_report_html(
    symbol: str,
    timeframe: str,
    backtest_results: dict,
    trades_df: pl.DataFrame,
    df: pl.DataFrame,
    optimization_df: pl.DataFrame | None = None,
    ml_results: dict | None = None,
    dl_result: dict | None = None,
    output_dir: str | Path = "docs",
) -> str:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    position_ids = trades_df["position_id"].unique().to_list()[:2]
    trade_chart = create_trade_chart(df, trades_df, position_ids, f"{symbol} {timeframe} - Sample Trades")
    trade_chart_html = trade_chart.to_html(full_html=False, include_plotlyjs=False)

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{symbol} {timeframe} - Multi-Band DCA Results</title>
    <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
            max-width: 1200px;
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
        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
        }}
        .metric {{
            text-align: center;
            padding: 15px;
            background: #f8f9fa;
            border-radius: 8px;
        }}
        .metric-value {{
            font-size: 24px;
            font-weight: bold;
            color: #2196F3;
        }}
        .metric-label {{
            color: #666;
            font-size: 14px;
        }}
        .positive {{ color: #4CAF50; }}
        .negative {{ color: #f44336; }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 10px;
        }}
        th, td {{
            padding: 10px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }}
        th {{ background: #f5f5f5; }}
        .chart-container {{ margin: 20px 0; }}
    </style>
</head>
<body>
    <h1>{symbol} {timeframe} - Multi-Band DCA Strategy Results</h1>
    <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>

    <div class="card">
        <h2>Backtest Performance</h2>
        <div class="metrics-grid">
            <div class="metric">
                <div class="metric-value {'positive' if backtest_results['total_return_pct'] > 0 else 'negative'}">
                    {backtest_results['total_return_pct']:.2f}%
                </div>
                <div class="metric-label">Total Return</div>
            </div>
            <div class="metric">
                <div class="metric-value">{backtest_results['sharpe_ratio']:.2f}</div>
                <div class="metric-label">Sharpe Ratio</div>
            </div>
            <div class="metric">
                <div class="metric-value">{backtest_results['win_rate']:.1f}%</div>
                <div class="metric-label">Win Rate</div>
            </div>
            <div class="metric">
                <div class="metric-value negative">{backtest_results['max_drawdown_pct']:.2f}%</div>
                <div class="metric-label">Max Drawdown</div>
            </div>
            <div class="metric">
                <div class="metric-value">{backtest_results['num_trades']}</div>
                <div class="metric-label">Total Trades</div>
            </div>
            <div class="metric">
                <div class="metric-value">{backtest_results['avg_hold_days']:.1f}</div>
                <div class="metric-label">Avg Hold Days</div>
            </div>
        </div>
    </div>

    <div class="card">
        <h2>Sample Trades</h2>
        <div class="chart-container">
            {trade_chart_html}
        </div>
    </div>
"""

    if ml_results:
        html_content += """
    <div class="card">
        <h2>ML Model Results</h2>
        <table>
            <tr>
                <th>Model</th>
                <th>Accuracy</th>
                <th>F1 Score</th>
                <th>AUC-ROC</th>
            </tr>
"""
        for name, metrics in ml_results.items():
            html_content += f"""
            <tr>
                <td>{name}</td>
                <td>{metrics.get('accuracy', 0):.4f}</td>
                <td>{metrics.get('f1', 0):.4f}</td>
                <td>{metrics.get('auc_roc', 0):.4f}</td>
            </tr>
"""
        html_content += """
        </table>
    </div>
"""

    if dl_result:
        html_content += f"""
    <div class="card">
        <h2>Deep Learning Results</h2>
        <div class="metrics-grid">
            <div class="metric">
                <div class="metric-value">{dl_result.get('test_accuracy', 0):.4f}</div>
                <div class="metric-label">Test Accuracy</div>
            </div>
            <div class="metric">
                <div class="metric-value">{dl_result.get('test_f1', 0):.4f}</div>
                <div class="metric-label">Test F1</div>
            </div>
            <div class="metric">
                <div class="metric-value">{dl_result.get('epochs_trained', 0)}</div>
                <div class="metric-label">Epochs Trained</div>
            </div>
        </div>
    </div>
"""

    html_content += """
</body>
</html>
"""

    filename = f"{symbol.lower()}_{timeframe}.html"
    output_path = output_dir / filename
    output_path.write_text(html_content)

    return str(output_path)


def generate_index_html(
    results: dict,
    output_dir: str | Path = "docs",
) -> str:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = ""
    for key, data in results.items():
        symbol, tf = key.split("_")
        bt = data["backtest"]
        rows += f"""
        <tr>
            <td><a href="{key}.html">{symbol}</a></td>
            <td>{tf}</td>
            <td class="{'positive' if bt['total_return_pct'] > 0 else 'negative'}">{bt['total_return_pct']:.2f}%</td>
            <td>{bt['sharpe_ratio']:.2f}</td>
            <td>{bt['win_rate']:.1f}%</td>
            <td>{bt['num_trades']}</td>
        </tr>
"""

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Multi-Band DCA Strategy - Experiment Results</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background: #f5f5f5;
        }}
        h1 {{ color: #333; }}
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
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }}
        th {{ background: #f5f5f5; font-weight: 600; }}
        tr:hover {{ background: #f9f9f9; }}
        a {{ color: #2196F3; text-decoration: none; }}
        a:hover {{ text-decoration: underline; }}
        .positive {{ color: #4CAF50; font-weight: bold; }}
        .negative {{ color: #f44336; font-weight: bold; }}
    </style>
</head>
<body>
    <h1>Multi-Band DCA Strategy - Experiment Results</h1>
    <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>

    <div class="card">
        <h2>All Experiments</h2>
        <table>
            <tr>
                <th>Symbol</th>
                <th>Timeframe</th>
                <th>Return</th>
                <th>Sharpe</th>
                <th>Win Rate</th>
                <th>Trades</th>
            </tr>
            {rows}
        </table>
    </div>

    <div class="card">
        <h2>Strategy Description</h2>
        <p><strong>Entry:</strong> Buy when price drops below Bollinger Band lower band (customizable sigma, e.g., -2.5σ)</p>
        <p><strong>Exit:</strong> Sell when price rises above upper band (customizable sigma, e.g., +0.7σ)</p>
        <p><strong>DCA:</strong> Add positions when price drops further (n-splits with avg-down percentage)</p>
        <p><strong>Risk:</strong> 2-month max hold period, no stop-loss</p>
    </div>
</body>
</html>
"""

    output_path = output_dir / "index.html"
    output_path.write_text(html_content)

    return str(output_path)
