from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import polars as pl
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.features.indicators import FeatureEngineer
from src.backtest.engine import MultiBandDCABacktester


@dataclass
class DLResult:
    model_name: str
    epochs_trained: int
    best_val_loss: float
    test_accuracy: float
    test_f1: float
    train_losses: list[float]
    val_losses: list[float]


class SequenceDataset(Dataset):
    def __init__(
        self,
        sequences: np.ndarray,
        features: np.ndarray,
        labels: np.ndarray,
    ):
        self.sequences = torch.FloatTensor(sequences)
        self.features = torch.FloatTensor(features)
        self.labels = torch.LongTensor(labels)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.sequences[idx], self.features[idx], self.labels[idx]


class LSTMModel(nn.Module):
    def __init__(
        self,
        seq_features: int,
        static_features: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=seq_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True,
        )

        self.fc_static = nn.Sequential(
            nn.Linear(static_features, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        combined_size = hidden_size * 2 + 64
        self.classifier = nn.Sequential(
            nn.Linear(combined_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 2),
        )

    def forward(
        self,
        seq: torch.Tensor,
        static: torch.Tensor,
    ) -> torch.Tensor:
        lstm_out, (h_n, _) = self.lstm(seq)
        lstm_features = torch.cat([h_n[-2], h_n[-1]], dim=1)

        static_features = self.fc_static(static)
        combined = torch.cat([lstm_features, static_features], dim=1)

        return self.classifier(combined)


class TransformerModel(nn.Module):
    def __init__(
        self,
        seq_features: int,
        static_features: int,
        seq_length: int = 100,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()

        self.input_proj = nn.Linear(seq_features, d_model)
        self.pos_embedding = nn.Parameter(torch.randn(1, seq_length, d_model))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.fc_static = nn.Sequential(
            nn.Linear(static_features, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        combined_size = d_model + 64
        self.classifier = nn.Sequential(
            nn.Linear(combined_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 2),
        )

    def forward(
        self,
        seq: torch.Tensor,
        static: torch.Tensor,
    ) -> torch.Tensor:
        x = self.input_proj(seq)
        x = x + self.pos_embedding[:, :x.size(1), :]
        x = self.transformer(x)
        transformer_features = x.mean(dim=1)

        static_features = self.fc_static(static)
        combined = torch.cat([transformer_features, static_features], dim=1)

        return self.classifier(combined)


class DLPipeline:
    def __init__(
        self,
        sequence_length: int = 100,
        hold_days_threshold: float = 5.0,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
        batch_size: int = 32,
        epochs: int = 50,
        learning_rate: float = 0.001,
        early_stopping_patience: int = 10,
        test_size: float = 0.2,
        val_size: float = 0.1,
        random_state: int = 42,
        device: str = "auto",
    ):
        self.sequence_length = sequence_length
        self.hold_days_threshold = hold_days_threshold
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.early_stopping_patience = early_stopping_patience
        self.test_size = test_size
        self.val_size = val_size
        self.random_state = random_state

        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.seq_scaler = StandardScaler()
        self.feature_scaler = StandardScaler()
        self.seq_cols = ["open", "high", "low", "close", "volume"]
        self.feature_cols: list[str] = []

    def prepare_sequences(
        self,
        df: pl.DataFrame,
        trades_df: pl.DataFrame,
        feature_engineer: FeatureEngineer,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        featured_df = feature_engineer.add_basic_features(df)
        featured_df = feature_engineer.add_entry_exit_bands(featured_df)

        self.feature_cols = feature_engineer.get_feature_columns()

        entry_labels = trades_df.filter(pl.col("entry_num") == 1).select([
            "entry_time",
            "hold_days",
        ]).with_columns(
            pl.when(pl.col("hold_days") > self.hold_days_threshold)
            .then(0)
            .otherwise(1)
            .alias("label")
        )

        entry_times = set(entry_labels["entry_time"].to_list())
        entry_label_map = dict(zip(
            entry_labels["entry_time"].to_list(),
            entry_labels["label"].to_list(),
        ))

        datetime_col = featured_df["datetime"].to_list()
        target_indices = []
        target_labels = []
        for i in range(self.sequence_length, len(datetime_col)):
            if datetime_col[i] in entry_times:
                target_indices.append(i)
                target_labels.append(entry_label_map[datetime_col[i]])

        if not target_indices:
            return np.array([]), np.array([]), np.array([])

        seq_df = featured_df.select(self.seq_cols)
        feat_df = featured_df.select([c for c in self.feature_cols if c in featured_df.columns])

        sequences = []
        features = []
        labels = []

        batch_size = 1000
        for batch_start in range(0, len(target_indices), batch_size):
            batch_end = min(batch_start + batch_size, len(target_indices))
            batch_indices = target_indices[batch_start:batch_end]
            batch_labels = target_labels[batch_start:batch_end]

            for idx, label in zip(batch_indices, batch_labels):
                seq_slice = seq_df.slice(idx - self.sequence_length, self.sequence_length)
                seq = seq_slice.to_numpy()

                feat_slice = feat_df.row(idx)
                feat = np.array(feat_slice)

                if not np.isnan(seq).any() and not np.isnan(feat).any():
                    sequences.append(seq)
                    features.append(feat)
                    labels.append(label)

        return np.array(sequences), np.array(features), np.array(labels)

    def create_dataloaders(
        self,
        sequences: np.ndarray,
        features: np.ndarray,
        labels: np.ndarray,
    ) -> tuple[DataLoader, DataLoader, DataLoader]:
        np.random.seed(self.random_state)

        indices = np.arange(len(labels))
        train_val_idx, test_idx = train_test_split(
            indices, test_size=self.test_size, stratify=labels, random_state=self.random_state
        )

        val_ratio = self.val_size / (1 - self.test_size)
        train_idx, val_idx = train_test_split(
            train_val_idx, test_size=val_ratio, stratify=labels[train_val_idx], random_state=self.random_state
        )

        seq_shape = sequences.shape
        sequences_flat = sequences.reshape(-1, seq_shape[-1])
        self.seq_scaler.fit(sequences_flat[train_idx.repeat(seq_shape[1])])
        sequences_scaled = self.seq_scaler.transform(sequences_flat).reshape(seq_shape)

        self.feature_scaler.fit(features[train_idx])
        features_scaled = self.feature_scaler.transform(features)

        train_dataset = SequenceDataset(
            sequences_scaled[train_idx], features_scaled[train_idx], labels[train_idx]
        )
        val_dataset = SequenceDataset(
            sequences_scaled[val_idx], features_scaled[val_idx], labels[val_idx]
        )
        test_dataset = SequenceDataset(
            sequences_scaled[test_idx], features_scaled[test_idx], labels[test_idx]
        )

        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)

        return train_loader, val_loader, test_loader

    def train_model(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
    ) -> tuple[nn.Module, list[float], list[float], int]:
        model = model.to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate)
        criterion = nn.CrossEntropyLoss()

        train_losses = []
        val_losses = []
        best_val_loss = float("inf")
        patience_counter = 0
        best_state = None

        for epoch in range(self.epochs):
            model.train()
            train_loss = 0.0
            for seq, feat, label in train_loader:
                seq, feat, label = seq.to(self.device), feat.to(self.device), label.to(self.device)

                optimizer.zero_grad()
                output = model(seq, feat)
                loss = criterion(output, label)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()

            train_loss /= len(train_loader)
            train_losses.append(train_loss)

            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for seq, feat, label in val_loader:
                    seq, feat, label = seq.to(self.device), feat.to(self.device), label.to(self.device)
                    output = model(seq, feat)
                    loss = criterion(output, label)
                    val_loss += loss.item()

            val_loss /= len(val_loader)
            val_losses.append(val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = model.state_dict().copy()
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= self.early_stopping_patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break

        if best_state:
            model.load_state_dict(best_state)

        return model, train_losses, val_losses, epoch + 1

    def evaluate_model(
        self,
        model: nn.Module,
        test_loader: DataLoader,
    ) -> tuple[float, float]:
        model.eval()
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for seq, feat, label in test_loader:
                seq, feat = seq.to(self.device), feat.to(self.device)
                output = model(seq, feat)
                preds = output.argmax(dim=1).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(label.numpy())

        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)

        accuracy = (all_preds == all_labels).mean()

        tp = ((all_preds == 1) & (all_labels == 1)).sum()
        fp = ((all_preds == 1) & (all_labels == 0)).sum()
        fn = ((all_preds == 0) & (all_labels == 1)).sum()

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        return accuracy, f1

    def run_pipeline(
        self,
        df: pl.DataFrame,
        trades_df: pl.DataFrame,
        feature_engineer: FeatureEngineer,
        model_type: str = "lstm",
    ) -> DLResult:
        print("Preparing sequences...")
        sequences, features, labels = self.prepare_sequences(df, trades_df, feature_engineer)
        print(f"Dataset: {len(labels)} samples, {sequences.shape[1]} timesteps, {sequences.shape[2]} seq features, {features.shape[1]} static features")
        print(f"Label distribution: positive={labels.sum()}, negative={len(labels) - labels.sum()}")

        if len(labels) < 20:
            raise ValueError(f"Not enough samples for training: {len(labels)}")

        train_loader, val_loader, test_loader = self.create_dataloaders(sequences, features, labels)
        print(f"Train: {len(train_loader.dataset)}, Val: {len(val_loader.dataset)}, Test: {len(test_loader.dataset)}")

        if model_type == "lstm":
            model = LSTMModel(
                seq_features=sequences.shape[2],
                static_features=features.shape[1],
                hidden_size=self.hidden_size,
                num_layers=self.num_layers,
                dropout=self.dropout,
            )
        elif model_type == "transformer":
            model = TransformerModel(
                seq_features=sequences.shape[2],
                static_features=features.shape[1],
                seq_length=self.sequence_length,
                d_model=64,
                nhead=4,
                num_layers=self.num_layers,
                dropout=self.dropout,
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        print(f"\nTraining {model_type}...")
        model, train_losses, val_losses, epochs_trained = self.train_model(model, train_loader, val_loader)

        accuracy, f1 = self.evaluate_model(model, test_loader)
        print(f"Test Accuracy: {accuracy:.4f}, F1: {f1:.4f}")

        return DLResult(
            model_name=model_type,
            epochs_trained=epochs_trained,
            best_val_loss=min(val_losses),
            test_accuracy=accuracy,
            test_f1=f1,
            train_losses=train_losses,
            val_losses=val_losses,
        )

    def save_model(self, model: nn.Module, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "model_state_dict": model.state_dict(),
            "seq_scaler": self.seq_scaler,
            "feature_scaler": self.feature_scaler,
            "seq_cols": self.seq_cols,
            "feature_cols": self.feature_cols,
            "config": {
                "sequence_length": self.sequence_length,
                "hidden_size": self.hidden_size,
                "num_layers": self.num_layers,
                "dropout": self.dropout,
            },
        }, path)
