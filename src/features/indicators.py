import polars as pl


def calc_sma(df: pl.DataFrame, col: str, period: int, alias: str | None = None) -> pl.Expr:
    alias = alias or f"{col}_sma_{period}"
    return pl.col(col).rolling_mean(window_size=period, min_samples=period).alias(alias)


def calc_ema(df: pl.DataFrame, col: str, period: int, alias: str | None = None) -> pl.Expr:
    alias = alias or f"{col}_ema_{period}"
    return pl.col(col).ewm_mean(span=period, min_samples=period).alias(alias)


def add_rsi(df: pl.DataFrame, period: int = 14, col: str = "close") -> pl.DataFrame:
    return df.with_columns(
        pl.col(col).diff().alias("_delta")
    ).with_columns(
        pl.when(pl.col("_delta") > 0).then(pl.col("_delta")).otherwise(0.0).alias("_gain"),
        pl.when(pl.col("_delta") < 0).then(-pl.col("_delta")).otherwise(0.0).alias("_loss"),
    ).with_columns(
        pl.col("_gain").ewm_mean(alpha=1/period, min_samples=period, adjust=False).alias("_avg_gain"),
        pl.col("_loss").ewm_mean(alpha=1/period, min_samples=period, adjust=False).alias("_avg_loss"),
    ).with_columns(
        (100 - (100 / (1 + pl.col("_avg_gain") / pl.col("_avg_loss")))).alias("rsi")
    ).drop(["_delta", "_gain", "_loss", "_avg_gain", "_avg_loss"])


def add_bollinger_bands(
    df: pl.DataFrame,
    period: int = 20,
    std_dev: float = 2.0,
    col: str = "close",
) -> pl.DataFrame:
    return df.with_columns(
        pl.col(col).rolling_mean(window_size=period, min_samples=period).alias("bb_middle"),
        pl.col(col).rolling_std(window_size=period, min_samples=period).alias("_bb_std"),
    ).with_columns(
        (pl.col("bb_middle") - pl.col("_bb_std") * std_dev).alias("bb_lower"),
        (pl.col("bb_middle") + pl.col("_bb_std") * std_dev).alias("bb_upper"),
    ).drop("_bb_std")


def add_bb_custom_sigma(
    df: pl.DataFrame,
    period: int,
    sigma: float,
    col: str = "close",
    alias: str = "bb_band",
) -> pl.DataFrame:
    return df.with_columns(
        (
            pl.col(col).rolling_mean(window_size=period, min_samples=period)
            + pl.col(col).rolling_std(window_size=period, min_samples=period) * sigma
        ).alias(alias)
    )


def add_ma_distance(df: pl.DataFrame, ma_col: str, close_col: str = "close") -> pl.DataFrame:
    dist_col = f"{ma_col}_dist"
    return df.with_columns(
        ((pl.col(close_col) - pl.col(ma_col)) / pl.col(ma_col) * 100).alias(dist_col)
    )


def add_bb_distance(df: pl.DataFrame) -> pl.DataFrame:
    return df.with_columns(
        (
            (pl.col("close") - (pl.col("bb_upper") + pl.col("bb_lower")) / 2)
            / ((pl.col("bb_upper") - pl.col("bb_lower")) / 2)
            * 100
        ).alias("bb_dist")
    )


class FeatureEngineer:
    def __init__(
        self,
        ma_periods: list[int] | None = None,
        rsi_period: int = 14,
        bb_period: int = 20,
        bb_std: float = 2.0,
    ):
        self.ma_periods = ma_periods or [25, 50, 100, 200, 400]
        self.rsi_period = rsi_period
        self.bb_period = bb_period
        self.bb_std = bb_std

    def add_basic_features(self, df: pl.DataFrame) -> pl.DataFrame:
        ma_exprs = []
        for period in self.ma_periods:
            ma_exprs.append(
                pl.col("close").rolling_mean(window_size=period, min_samples=period).alias(f"ma_{period}")
            )

        df = df.with_columns(ma_exprs)

        for period in self.ma_periods:
            df = add_ma_distance(df, f"ma_{period}")

        df = add_rsi(df, self.rsi_period)
        df = add_bollinger_bands(df, self.bb_period, self.bb_std)
        df = add_bb_distance(df)

        df = df.with_columns(
            ((pl.col("bb_upper") - pl.col("bb_lower")) / pl.col("bb_middle") * 100).alias("bb_width")
        )

        return df

    def add_entry_exit_bands(
        self,
        df: pl.DataFrame,
        entry_sigma: float = -2.5,
        exit_sigma: float = 0.7,
        bb_period: int = 20,
    ) -> pl.DataFrame:
        df = add_bb_custom_sigma(df, bb_period, entry_sigma, alias="entry_band")
        df = add_bb_custom_sigma(df, bb_period, exit_sigma, alias="exit_band")
        return df

    def add_multi_tf_features(
        self,
        primary_df: pl.DataFrame,
        tf_dfs: dict[str, pl.DataFrame],
    ) -> pl.DataFrame:
        result = primary_df

        for tf_name, tf_df in tf_dfs.items():
            tf_features = self.add_basic_features(tf_df)

            primary_cols = set(primary_df.columns)
            feature_cols = [c for c in tf_features.columns if c not in primary_cols]

            rename_map = {c: f"{tf_name}_{c}" for c in feature_cols}
            tf_subset = tf_features.select(["datetime"] + feature_cols).rename(rename_map)

            result = result.join_asof(
                tf_subset,
                on="datetime",
                strategy="backward",
            )

        return result

    def generate_features(
        self,
        df: pl.DataFrame,
        entry_sigma: float = -2.5,
        exit_sigma: float = 0.7,
        bb_period: int = 20,
        tf_dfs: dict[str, pl.DataFrame] | None = None,
    ) -> pl.DataFrame:
        result = self.add_basic_features(df)
        result = self.add_entry_exit_bands(result, entry_sigma, exit_sigma, bb_period)

        if tf_dfs:
            result = self.add_multi_tf_features(result, tf_dfs)

        return result

    def get_feature_columns(self) -> list[str]:
        cols = []
        for period in self.ma_periods:
            cols.extend([f"ma_{period}", f"ma_{period}_dist"])
        cols.extend(["rsi", "bb_lower", "bb_middle", "bb_upper", "bb_dist", "bb_width"])
        return cols
