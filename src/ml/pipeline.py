from dataclasses import dataclass
from pathlib import Path
import json

import numpy as np
import polars as pl
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
)
from imblearn.under_sampling import RandomUnderSampler
import joblib

from src.features.indicators import FeatureEngineer
from src.backtest.engine import MultiBandDCABacktester
from src.data.store import ResultStore


@dataclass
class MLResult:
    model_name: str
    accuracy: float
    precision: float
    recall: float
    f1: float
    auc_roc: float
    feature_importance: dict | None
    confusion_mat: np.ndarray
    classification_rep: str


class MLPipeline:
    def __init__(
        self,
        hold_days_threshold: float = 5.0,
        test_size: float = 0.2,
        val_size: float = 0.1,
        random_state: int = 42,
        undersample: bool = True,
    ):
        self.hold_days_threshold = hold_days_threshold
        self.test_size = test_size
        self.val_size = val_size
        self.random_state = random_state
        self.undersample = undersample
        self.scaler = StandardScaler()
        self.feature_cols: list[str] = []

    def prepare_labels(
        self,
        df: pl.DataFrame,
        trades_df: pl.DataFrame,
    ) -> pl.DataFrame:
        entry_labels = trades_df.filter(pl.col("entry_num") == 1).select([
            "entry_time",
            "hold_days",
            "pnl_pct",
            "status",
        ]).with_columns(
            pl.when(pl.col("hold_days") > self.hold_days_threshold)
            .then(0)
            .otherwise(1)
            .alias("label")
        )

        result = df.join(
            entry_labels.select(["entry_time", "label"]),
            left_on="datetime",
            right_on="entry_time",
            how="left",
        )

        return result

    def prepare_features(
        self,
        df: pl.DataFrame,
        feature_engineer: FeatureEngineer,
    ) -> pl.DataFrame:
        result = feature_engineer.add_basic_features(df)
        result = feature_engineer.add_entry_exit_bands(result)

        self.feature_cols = feature_engineer.get_feature_columns()
        self.feature_cols = [c for c in self.feature_cols if c in result.columns]

        return result

    def create_dataset(
        self,
        df: pl.DataFrame,
        trades_df: pl.DataFrame,
        feature_engineer: FeatureEngineer,
    ) -> tuple[np.ndarray, np.ndarray, pl.DataFrame]:
        featured_df = self.prepare_features(df, feature_engineer)
        labeled_df = self.prepare_labels(featured_df, trades_df)

        entry_samples = labeled_df.filter(pl.col("label").is_not_null())

        if entry_samples.is_empty():
            raise ValueError("No labeled samples found")

        X = entry_samples.select(self.feature_cols).to_numpy()
        y = entry_samples["label"].to_numpy()

        valid_mask = ~np.isnan(X).any(axis=1)
        X = X[valid_mask]
        y = y[valid_mask]
        entry_samples = entry_samples.filter(pl.Series(valid_mask))

        return X, y, entry_samples

    def split_data(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state, stratify=y
        )

        val_ratio = self.val_size / (1 - self.test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_ratio, random_state=self.random_state, stratify=y_temp
        )

        if self.undersample:
            rus = RandomUnderSampler(random_state=self.random_state)
            X_train, y_train = rus.fit_resample(X_train, y_train)

        X_train = self.scaler.fit_transform(X_train)
        X_val = self.scaler.transform(X_val)
        X_test = self.scaler.transform(X_test)

        return X_train, X_val, X_test, y_train, y_val, y_test

    def train_model(
        self,
        model_type: str,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
    ):
        if model_type == "random_forest":
            model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=self.random_state,
                n_jobs=-1,
            )
        elif model_type == "gradient_boosting":
            model = GradientBoostingClassifier(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                random_state=self.random_state,
            )
        elif model_type == "logistic_regression":
            model = LogisticRegression(
                max_iter=1000,
                random_state=self.random_state,
                n_jobs=-1,
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        model.fit(X_train, y_train)
        return model

    def evaluate_model(
        self,
        model,
        model_name: str,
        X_test: np.ndarray,
        y_test: np.ndarray,
    ) -> MLResult:
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else y_pred

        feature_importance = None
        if hasattr(model, "feature_importances_"):
            feature_importance = dict(zip(self.feature_cols, model.feature_importances_.tolist()))
        elif hasattr(model, "coef_"):
            feature_importance = dict(zip(self.feature_cols, np.abs(model.coef_[0]).tolist()))

        return MLResult(
            model_name=model_name,
            accuracy=accuracy_score(y_test, y_pred),
            precision=precision_score(y_test, y_pred, zero_division=0),
            recall=recall_score(y_test, y_pred, zero_division=0),
            f1=f1_score(y_test, y_pred, zero_division=0),
            auc_roc=roc_auc_score(y_test, y_proba) if len(np.unique(y_test)) > 1 else 0.0,
            feature_importance=feature_importance,
            confusion_mat=confusion_matrix(y_test, y_pred),
            classification_rep=classification_report(y_test, y_pred),
        )

    def run_pipeline(
        self,
        df: pl.DataFrame,
        trades_df: pl.DataFrame,
        feature_engineer: FeatureEngineer,
        model_types: list[str] | None = None,
    ) -> dict[str, MLResult]:
        model_types = model_types or ["random_forest", "gradient_boosting", "logistic_regression"]

        X, y, _ = self.create_dataset(df, trades_df, feature_engineer)
        print(f"Dataset: {X.shape[0]} samples, {X.shape[1]} features")
        print(f"Label distribution: positive={np.sum(y)}, negative={len(y) - np.sum(y)}")

        X_train, X_val, X_test, y_train, y_val, y_test = self.split_data(X, y)
        print(f"Train: {len(y_train)}, Val: {len(y_val)}, Test: {len(y_test)}")

        results = {}
        for model_type in model_types:
            print(f"\nTraining {model_type}...")
            model = self.train_model(model_type, X_train, y_train, X_val, y_val)
            result = self.evaluate_model(model, model_type, X_test, y_test)
            results[model_type] = result
            print(f"  Accuracy: {result.accuracy:.4f}")
            print(f"  F1: {result.f1:.4f}")
            print(f"  AUC-ROC: {result.auc_roc:.4f}")

        return results

    def save_model(self, model, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"model": model, "scaler": self.scaler, "feature_cols": self.feature_cols}, path)

    def load_model(self, path: str | Path):
        data = joblib.load(path)
        self.scaler = data["scaler"]
        self.feature_cols = data["feature_cols"]
        return data["model"]
