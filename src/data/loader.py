from pathlib import Path

import polars as pl


TIMEFRAME_MINUTES = {
    "1m": 1,
    "5m": 5,
    "15m": 15,
    "30m": 30,
    "1h": 60,
    "4h": 240,
    "1d": 1440,
}


class CryptoDataLoader:
    def __init__(self, base_path: str | Path):
        self.base_path = Path(base_path)

    def _get_symbol_path(self, symbol: str) -> Path:
        return self.base_path / symbol

    def load_raw(
        self,
        symbol: str,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> pl.DataFrame:
        symbol_path = self._get_symbol_path(symbol)
        if not symbol_path.exists():
            raise ValueError(f"Symbol {symbol} not found at {symbol_path}")

        date_dirs = sorted(symbol_path.glob("date=*"))

        if start_date:
            date_dirs = [d for d in date_dirs if d.name >= f"date={start_date}"]
        if end_date:
            date_dirs = [d for d in date_dirs if d.name <= f"date={end_date}"]

        if not date_dirs:
            raise ValueError(f"No data found for {symbol} in date range")

        parquet_files = []
        for date_dir in date_dirs:
            parquet_files.extend(date_dir.glob("*.parquet"))

        df = pl.concat([pl.read_parquet(pf) for pf in parquet_files])
        df = df.sort("datetime").unique(subset=["datetime"], maintain_order=True)

        return df

    def resample_ohlcv(
        self,
        df: pl.DataFrame,
        timeframe: str,
    ) -> pl.DataFrame:
        if timeframe not in TIMEFRAME_MINUTES:
            raise ValueError(f"Invalid timeframe: {timeframe}")

        minutes = TIMEFRAME_MINUTES[timeframe]
        if minutes == 1:
            return df.clone()

        return (
            df.sort("datetime")
            .group_by_dynamic("datetime", every=f"{minutes}m", closed="left", label="left")
            .agg(
                pl.col("open").first(),
                pl.col("high").max(),
                pl.col("low").min(),
                pl.col("close").last(),
                pl.col("volume").sum(),
                pl.col("quote_volume").sum(),
                pl.col("trades").sum(),
                pl.col("taker_buy_base").sum(),
                pl.col("taker_buy_quote").sum(),
            )
            .drop_nulls(subset=["open"])
        )

    def load(
        self,
        symbol: str,
        timeframe: str = "5m",
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> pl.DataFrame:
        df = self.load_raw(symbol, start_date, end_date)
        return self.resample_ohlcv(df, timeframe)

    def load_multi_timeframe(
        self,
        symbol: str,
        timeframes: list[str],
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> dict[str, pl.DataFrame]:
        raw_df = self.load_raw(symbol, start_date, end_date)

        return {tf: self.resample_ohlcv(raw_df, tf) for tf in timeframes}

    def get_available_symbols(self) -> list[str]:
        return sorted([d.name for d in self.base_path.iterdir() if d.is_dir()])

    def get_date_range(self, symbol: str) -> tuple[str, str]:
        symbol_path = self._get_symbol_path(symbol)
        date_dirs = sorted(symbol_path.glob("date=*"))

        if not date_dirs:
            raise ValueError(f"No data found for {symbol}")

        start_date = date_dirs[0].name.replace("date=", "")
        end_date = date_dirs[-1].name.replace("date=", "")

        return start_date, end_date
