from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml


@dataclass
class DataConfig:
    base_path: Path
    default_symbol: str
    base_timeframe: str


@dataclass
class StrategyConfig:
    bb_period: int
    bb_entry_sigma: float
    bb_exit_sigma: float
    n_splits: int
    avg_down_pct: float
    max_hold_days: int
    stop_loss: Optional[float]
    initial_capital: float
    position_size_pct: float


@dataclass
class TimeframeConfig:
    primary: list[str]
    feature_tfs: list[str]


@dataclass
class FeatureConfig:
    ma_periods: list[int]
    rsi_period: int
    bb_period: int
    bb_std: float


@dataclass
class MLConfig:
    hold_days_threshold: int
    test_size: float
    val_size: float
    random_state: int
    undersample: bool
    undersample_strategy: str


@dataclass
class DLConfig:
    sequence_length: int
    hidden_size: int
    num_layers: int
    dropout: float
    batch_size: int
    epochs: int
    learning_rate: float
    early_stopping_patience: int


@dataclass
class OptimizationConfig:
    param_ranges: dict
    n_trials: int
    metric: str


@dataclass
class OutputConfig:
    results_dir: Path
    docs_dir: Path
    models_dir: Path


@dataclass
class Config:
    data: DataConfig
    strategy: StrategyConfig
    timeframes: TimeframeConfig
    features: FeatureConfig
    ml: MLConfig
    dl: DLConfig
    optimization: OptimizationConfig
    output: OutputConfig

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Config":
        with open(path) as f:
            raw = yaml.safe_load(f)

        return cls(
            data=DataConfig(
                base_path=Path(raw["data"]["base_path"]),
                default_symbol=raw["data"]["default_symbol"],
                base_timeframe=raw["data"]["base_timeframe"],
            ),
            strategy=StrategyConfig(**raw["strategy"]),
            timeframes=TimeframeConfig(**raw["timeframes"]),
            features=FeatureConfig(**raw["features"]),
            ml=MLConfig(**raw["ml"]),
            dl=DLConfig(**raw["dl"]),
            optimization=OptimizationConfig(**raw["optimization"]),
            output=OutputConfig(
                results_dir=Path(raw["output"]["results_dir"]),
                docs_dir=Path(raw["output"]["docs_dir"]),
                models_dir=Path(raw["output"]["models_dir"]),
            ),
        )


def load_config(path: str | Path = "configs/default.yaml") -> Config:
    return Config.from_yaml(path)
