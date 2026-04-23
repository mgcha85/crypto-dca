from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional

import numpy as np
import polars as pl


class PositionStatus(Enum):
    OPEN = "open"
    CLOSED_PROFIT = "closed_profit"
    CLOSED_LOSS = "closed_loss"
    EXPIRED = "expired"


@dataclass
class TradeEntry:
    entry_time: datetime
    entry_price: float
    quantity: float
    entry_idx: int


@dataclass
class Position:
    id: int
    entries: list[TradeEntry] = field(default_factory=list)
    exit_time: Optional[datetime] = None
    exit_price: Optional[float] = None
    exit_idx: Optional[int] = None
    status: PositionStatus = PositionStatus.OPEN
    target_exit_price: float = 0.0

    @property
    def total_quantity(self) -> float:
        return sum(e.quantity for e in self.entries)

    @property
    def avg_entry_price(self) -> float:
        if not self.entries:
            return 0.0
        total_cost = sum(e.entry_price * e.quantity for e in self.entries)
        return total_cost / self.total_quantity

    @property
    def first_entry_time(self) -> datetime:
        return self.entries[0].entry_time

    @property
    def hold_days(self) -> float:
        if self.exit_time is None:
            return 0.0
        delta = self.exit_time - self.first_entry_time
        return delta.total_seconds() / 86400

    @property
    def pnl_pct(self) -> float:
        if self.exit_price is None:
            return 0.0
        return ((self.exit_price - self.avg_entry_price) / self.avg_entry_price) * 100

    @property
    def num_entries(self) -> int:
        return len(self.entries)


@dataclass
class BacktestResult:
    positions: list[Position]
    equity_curve: pl.Series
    initial_capital: float
    final_capital: float

    @property
    def total_return_pct(self) -> float:
        return ((self.final_capital - self.initial_capital) / self.initial_capital) * 100

    @property
    def win_rate(self) -> float:
        closed = [p for p in self.positions if p.status != PositionStatus.OPEN]
        if not closed:
            return 0.0
        wins = sum(1 for p in closed if p.pnl_pct > 0)
        return wins / len(closed) * 100

    @property
    def avg_hold_days(self) -> float:
        closed = [p for p in self.positions if p.status != PositionStatus.OPEN]
        if not closed:
            return 0.0
        return sum(p.hold_days for p in closed) / len(closed)

    @property
    def num_trades(self) -> int:
        return len([p for p in self.positions if p.status != PositionStatus.OPEN])

    @property
    def sharpe_ratio(self) -> float:
        if len(self.equity_curve) < 2:
            return 0.0
        returns = self.equity_curve.pct_change().drop_nulls()
        std = returns.std()
        if std is None or std == 0:
            return 0.0
        mean = returns.mean()
        if mean is None:
            return 0.0
        return np.sqrt(252 * 24 * 12) * mean / std

    @property
    def max_drawdown_pct(self) -> float:
        equity_arr = self.equity_curve.to_numpy()
        peak = np.maximum.accumulate(equity_arr)
        drawdown = (equity_arr - peak) / peak * 100
        return float(np.min(drawdown))

    def to_dict(self) -> dict:
        return {
            "total_return_pct": self.total_return_pct,
            "win_rate": self.win_rate,
            "avg_hold_days": self.avg_hold_days,
            "num_trades": self.num_trades,
            "sharpe_ratio": self.sharpe_ratio,
            "max_drawdown_pct": self.max_drawdown_pct,
            "initial_capital": self.initial_capital,
            "final_capital": self.final_capital,
        }


class MultiBandDCABacktester:
    def __init__(
        self,
        bb_period: int = 20,
        bb_entry_sigma: float = -2.5,
        bb_exit_sigma: float = 0.7,
        n_splits: int = 3,
        avg_down_pct: float = 10.0,
        max_hold_days: int = 60,
        initial_capital: float = 10000.0,
        position_size_pct: float = 10.0,
    ):
        self.bb_period = bb_period
        self.bb_entry_sigma = bb_entry_sigma
        self.bb_exit_sigma = bb_exit_sigma
        self.n_splits = int(n_splits)
        self.avg_down_pct = avg_down_pct
        self.max_hold_days = max_hold_days
        self.initial_capital = initial_capital
        self.position_size_pct = position_size_pct

    def _prepare_data(self, df: pl.DataFrame) -> pl.DataFrame:
        return df.with_columns(
            pl.col("close").rolling_mean(window_size=self.bb_period, min_samples=self.bb_period).alias("_bb_middle"),
            pl.col("close").rolling_std(window_size=self.bb_period, min_samples=self.bb_period).alias("_bb_std"),
        ).with_columns(
            (pl.col("_bb_middle") + pl.col("_bb_std") * self.bb_entry_sigma).alias("entry_band"),
            (pl.col("_bb_middle") + pl.col("_bb_std") * self.bb_exit_sigma).alias("exit_band"),
        ).drop(["_bb_middle", "_bb_std"])

    def _get_next_avg_down_threshold(self, position: Position) -> float:
        num_entries = position.num_entries
        if num_entries == 1:
            return position.avg_entry_price * (1 - self.avg_down_pct / 200)
        return position.avg_entry_price * (1 - self.avg_down_pct / 100)

    def _can_add_entry(self, position: Position, current_price: float) -> bool:
        if position.num_entries >= self.n_splits:
            return False
        threshold = self._get_next_avg_down_threshold(position)
        return current_price <= threshold

    def run(self, df: pl.DataFrame) -> BacktestResult:
        df = self._prepare_data(df)

        datetimes = df["datetime"].to_list()
        closes = df["close"].to_numpy()
        entry_bands = df["entry_band"].to_numpy()
        exit_bands = df["exit_band"].to_numpy()

        capital = self.initial_capital
        positions: list[Position] = []
        current_position: Optional[Position] = None
        equity_history = []
        position_counter = 0

        split_capital = self.initial_capital * (self.position_size_pct / 100)

        for i in range(self.bb_period, len(df)):
            current_time = datetimes[i]
            current_price = closes[i]
            entry_price = entry_bands[i]
            exit_price = exit_bands[i]

            if np.isnan(entry_price) or np.isnan(exit_price):
                equity_history.append(capital)
                continue

            if current_position is not None:
                days_held = (current_time - current_position.first_entry_time).total_seconds() / 86400
                if days_held >= self.max_hold_days:
                    current_position.exit_time = current_time
                    current_position.exit_price = current_price
                    current_position.exit_idx = i
                    current_position.status = PositionStatus.EXPIRED
                    capital += current_position.total_quantity * current_price
                    positions.append(current_position)
                    current_position = None

            if current_position is not None:
                if current_price >= current_position.target_exit_price:
                    current_position.exit_time = current_time
                    current_position.exit_price = current_price
                    current_position.exit_idx = i
                    if current_price >= current_position.avg_entry_price:
                        current_position.status = PositionStatus.CLOSED_PROFIT
                    else:
                        current_position.status = PositionStatus.CLOSED_LOSS
                    capital += current_position.total_quantity * current_price
                    positions.append(current_position)
                    current_position = None

            if current_position is not None:
                if self._can_add_entry(current_position, current_price):
                    if capital >= split_capital:
                        qty = split_capital / current_price
                        capital -= split_capital
                        current_position.entries.append(
                            TradeEntry(
                                entry_time=current_time,
                                entry_price=current_price,
                                quantity=qty,
                                entry_idx=i,
                            )
                        )
                        current_position.target_exit_price = exit_price

            if current_position is None and current_price <= entry_price:
                if capital >= split_capital:
                    position_counter += 1
                    qty = split_capital / current_price
                    capital -= split_capital
                    current_position = Position(
                        id=position_counter,
                        entries=[
                            TradeEntry(
                                entry_time=current_time,
                                entry_price=current_price,
                                quantity=qty,
                                entry_idx=i,
                            )
                        ],
                        target_exit_price=exit_price,
                    )

            position_value = 0.0
            if current_position is not None:
                position_value = current_position.total_quantity * current_price

            equity_history.append(capital + position_value)

        if current_position is not None:
            final_price = closes[-1]
            current_position.exit_time = datetimes[-1]
            current_position.exit_price = final_price
            current_position.exit_idx = len(df) - 1
            current_position.status = PositionStatus.OPEN
            capital += current_position.total_quantity * final_price
            positions.append(current_position)

        equity_curve = pl.Series("equity", equity_history)

        return BacktestResult(
            positions=positions,
            equity_curve=equity_curve,
            initial_capital=self.initial_capital,
            final_capital=capital,
        )

    def get_trades_df(self, result: BacktestResult) -> pl.DataFrame:
        records = []
        for pos in result.positions:
            for i, entry in enumerate(pos.entries):
                records.append({
                    "position_id": pos.id,
                    "entry_num": i + 1,
                    "entry_time": entry.entry_time,
                    "entry_price": entry.entry_price,
                    "quantity": entry.quantity,
                    "exit_time": pos.exit_time,
                    "exit_price": pos.exit_price,
                    "avg_entry_price": pos.avg_entry_price,
                    "pnl_pct": pos.pnl_pct,
                    "hold_days": pos.hold_days,
                    "status": pos.status.value,
                })
        return pl.DataFrame(records)
