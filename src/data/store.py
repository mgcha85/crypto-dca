import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any

import polars as pl


class ResultStore:
    def __init__(self, db_path: str | Path = "results/results.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS experiments (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    symbol TEXT NOT NULL,
                    timeframe TEXT NOT NULL,
                    start_date TEXT,
                    end_date TEXT,
                    experiment_type TEXT NOT NULL,
                    description TEXT
                );

                CREATE TABLE IF NOT EXISTS backtest_params (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    experiment_id INTEGER NOT NULL,
                    bb_period INTEGER,
                    bb_entry_sigma REAL,
                    bb_exit_sigma REAL,
                    n_splits INTEGER,
                    avg_down_pct REAL,
                    max_hold_days INTEGER,
                    initial_capital REAL,
                    position_size_pct REAL,
                    FOREIGN KEY (experiment_id) REFERENCES experiments(id)
                );

                CREATE TABLE IF NOT EXISTS backtest_results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    experiment_id INTEGER NOT NULL,
                    params_id INTEGER NOT NULL,
                    total_return_pct REAL,
                    win_rate REAL,
                    avg_hold_days REAL,
                    num_trades INTEGER,
                    sharpe_ratio REAL,
                    max_drawdown_pct REAL,
                    final_capital REAL,
                    FOREIGN KEY (experiment_id) REFERENCES experiments(id),
                    FOREIGN KEY (params_id) REFERENCES backtest_params(id)
                );

                CREATE TABLE IF NOT EXISTS trades (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    experiment_id INTEGER NOT NULL,
                    params_id INTEGER NOT NULL,
                    position_id INTEGER,
                    entry_num INTEGER,
                    entry_time TIMESTAMP,
                    entry_price REAL,
                    quantity REAL,
                    exit_time TIMESTAMP,
                    exit_price REAL,
                    avg_entry_price REAL,
                    pnl_pct REAL,
                    hold_days REAL,
                    status TEXT,
                    FOREIGN KEY (experiment_id) REFERENCES experiments(id),
                    FOREIGN KEY (params_id) REFERENCES backtest_params(id)
                );

                CREATE TABLE IF NOT EXISTS optimization_results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    experiment_id INTEGER NOT NULL,
                    best_params_id INTEGER,
                    metric TEXT,
                    best_score REAL,
                    n_trials INTEGER,
                    method TEXT,
                    FOREIGN KEY (experiment_id) REFERENCES experiments(id),
                    FOREIGN KEY (best_params_id) REFERENCES backtest_params(id)
                );

                CREATE TABLE IF NOT EXISTS ml_results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    experiment_id INTEGER NOT NULL,
                    model_type TEXT,
                    accuracy REAL,
                    precision_score REAL,
                    recall REAL,
                    f1 REAL,
                    auc_roc REAL,
                    feature_importance TEXT,
                    model_path TEXT,
                    FOREIGN KEY (experiment_id) REFERENCES experiments(id)
                );

                CREATE TABLE IF NOT EXISTS dl_results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    experiment_id INTEGER NOT NULL,
                    model_type TEXT,
                    sequence_length INTEGER,
                    hidden_size INTEGER,
                    num_layers INTEGER,
                    epochs_trained INTEGER,
                    best_val_loss REAL,
                    test_accuracy REAL,
                    test_f1 REAL,
                    model_path TEXT,
                    FOREIGN KEY (experiment_id) REFERENCES experiments(id)
                );

                CREATE INDEX IF NOT EXISTS idx_exp_symbol ON experiments(symbol);
                CREATE INDEX IF NOT EXISTS idx_exp_timeframe ON experiments(timeframe);
                CREATE INDEX IF NOT EXISTS idx_trades_exp ON trades(experiment_id);
            """)

    def create_experiment(
        self,
        symbol: str,
        timeframe: str,
        experiment_type: str,
        start_date: str | None = None,
        end_date: str | None = None,
        description: str | None = None,
    ) -> int:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                """
                INSERT INTO experiments (symbol, timeframe, start_date, end_date, experiment_type, description)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (symbol, timeframe, start_date, end_date, experiment_type, description),
            )
            return cursor.lastrowid

    def save_backtest_params(self, experiment_id: int, params: dict) -> int:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                """
                INSERT INTO backtest_params 
                (experiment_id, bb_period, bb_entry_sigma, bb_exit_sigma, n_splits, 
                 avg_down_pct, max_hold_days, initial_capital, position_size_pct)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    experiment_id,
                    params.get("bb_period"),
                    params.get("bb_entry_sigma"),
                    params.get("bb_exit_sigma"),
                    params.get("n_splits"),
                    params.get("avg_down_pct"),
                    params.get("max_hold_days"),
                    params.get("initial_capital"),
                    params.get("position_size_pct"),
                ),
            )
            return cursor.lastrowid

    def save_backtest_result(
        self,
        experiment_id: int,
        params_id: int,
        result: dict,
    ) -> int:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                """
                INSERT INTO backtest_results
                (experiment_id, params_id, total_return_pct, win_rate, avg_hold_days,
                 num_trades, sharpe_ratio, max_drawdown_pct, final_capital)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    experiment_id,
                    params_id,
                    result.get("total_return_pct"),
                    result.get("win_rate"),
                    result.get("avg_hold_days"),
                    result.get("num_trades"),
                    result.get("sharpe_ratio"),
                    result.get("max_drawdown_pct"),
                    result.get("final_capital"),
                ),
            )
            return cursor.lastrowid

    def save_trades(
        self,
        experiment_id: int,
        params_id: int,
        trades_df: pl.DataFrame,
    ) -> None:
        with sqlite3.connect(self.db_path) as conn:
            for row in trades_df.iter_rows(named=True):
                conn.execute(
                    """
                    INSERT INTO trades
                    (experiment_id, params_id, position_id, entry_num, entry_time,
                     entry_price, quantity, exit_time, exit_price, avg_entry_price,
                     pnl_pct, hold_days, status)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        experiment_id,
                        params_id,
                        row.get("position_id"),
                        row.get("entry_num"),
                        str(row.get("entry_time")),
                        row.get("entry_price"),
                        row.get("quantity"),
                        str(row.get("exit_time")),
                        row.get("exit_price"),
                        row.get("avg_entry_price"),
                        row.get("pnl_pct"),
                        row.get("hold_days"),
                        row.get("status"),
                    ),
                )

    def save_optimization_result(
        self,
        experiment_id: int,
        best_params_id: int,
        metric: str,
        best_score: float,
        n_trials: int,
        method: str,
    ) -> int:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                """
                INSERT INTO optimization_results
                (experiment_id, best_params_id, metric, best_score, n_trials, method)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (experiment_id, best_params_id, metric, best_score, n_trials, method),
            )
            return cursor.lastrowid

    def save_ml_result(
        self,
        experiment_id: int,
        model_type: str,
        metrics: dict,
        feature_importance: str | None = None,
        model_path: str | None = None,
    ) -> int:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                """
                INSERT INTO ml_results
                (experiment_id, model_type, accuracy, precision_score, recall, f1, auc_roc,
                 feature_importance, model_path)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    experiment_id,
                    model_type,
                    metrics.get("accuracy"),
                    metrics.get("precision"),
                    metrics.get("recall"),
                    metrics.get("f1"),
                    metrics.get("auc_roc"),
                    feature_importance,
                    model_path,
                ),
            )
            return cursor.lastrowid

    def save_dl_result(
        self,
        experiment_id: int,
        model_type: str,
        config: dict,
        metrics: dict,
        model_path: str | None = None,
    ) -> int:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                """
                INSERT INTO dl_results
                (experiment_id, model_type, sequence_length, hidden_size, num_layers,
                 epochs_trained, best_val_loss, test_accuracy, test_f1, model_path)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    experiment_id,
                    model_type,
                    config.get("sequence_length"),
                    config.get("hidden_size"),
                    config.get("num_layers"),
                    metrics.get("epochs_trained"),
                    metrics.get("best_val_loss"),
                    metrics.get("test_accuracy"),
                    metrics.get("test_f1"),
                    model_path,
                ),
            )
            return cursor.lastrowid

    def get_experiments(
        self,
        symbol: str | None = None,
        timeframe: str | None = None,
        experiment_type: str | None = None,
    ) -> pl.DataFrame:
        query = "SELECT * FROM experiments WHERE 1=1"
        params = []

        if symbol:
            query += " AND symbol = ?"
            params.append(symbol)
        if timeframe:
            query += " AND timeframe = ?"
            params.append(timeframe)
        if experiment_type:
            query += " AND experiment_type = ?"
            params.append(experiment_type)

        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(query, params).fetchall()
            return pl.DataFrame([dict(row) for row in rows])

    def get_backtest_results(self, experiment_id: int) -> pl.DataFrame:
        query = """
            SELECT br.*, bp.*
            FROM backtest_results br
            JOIN backtest_params bp ON br.params_id = bp.id
            WHERE br.experiment_id = ?
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(query, (experiment_id,)).fetchall()
            return pl.DataFrame([dict(row) for row in rows])

    def get_trades(self, experiment_id: int, params_id: int | None = None) -> pl.DataFrame:
        query = "SELECT * FROM trades WHERE experiment_id = ?"
        params = [experiment_id]

        if params_id:
            query += " AND params_id = ?"
            params.append(params_id)

        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(query, params).fetchall()
            return pl.DataFrame([dict(row) for row in rows])

    def get_best_results(
        self,
        symbol: str | None = None,
        timeframe: str | None = None,
        metric: str = "sharpe_ratio",
        limit: int = 10,
    ) -> pl.DataFrame:
        query = f"""
            SELECT e.symbol, e.timeframe, e.start_date, e.end_date,
                   bp.bb_entry_sigma, bp.bb_exit_sigma, bp.n_splits, bp.avg_down_pct,
                   br.total_return_pct, br.win_rate, br.sharpe_ratio, br.max_drawdown_pct, br.num_trades
            FROM backtest_results br
            JOIN backtest_params bp ON br.params_id = bp.id
            JOIN experiments e ON br.experiment_id = e.id
            WHERE 1=1
        """
        params = []

        if symbol:
            query += " AND e.symbol = ?"
            params.append(symbol)
        if timeframe:
            query += " AND e.timeframe = ?"
            params.append(timeframe)

        query += f" ORDER BY br.{metric} DESC LIMIT ?"
        params.append(limit)

        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(query, params).fetchall()
            return pl.DataFrame([dict(row) for row in rows])
