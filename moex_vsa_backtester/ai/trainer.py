from pathlib import Path
from typing import Optional, Tuple
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)
from sklearn.preprocessing import StandardScaler
from loguru import logger

from ai.model import get_model


class ModelTrainer:
    def __init__(
        self,
        model_type: str = "mlp",
        hidden_sizes: list = None,
        dropout: float = 0.3,
        learning_rate: float = 0.001,
        batch_size: int = 32,
        epochs: int = 100,
        early_stopping_patience: int = 15,
        use_attention: bool = True,
    ):
        self.model_type = model_type
        self.hidden_sizes = hidden_sizes
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.early_stopping_patience = early_stopping_patience
        self.use_attention = use_attention

        self.model: Optional[nn.Module] = None
        self.scaler = None
        self.history = {"train_loss": [], "val_loss": [], "val_auc": []}

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        val_size: float = 0.2,
        scale: bool = True,
        random_state: int = 42,
    ) -> dict:
        if X.size == 0 or y.size == 0:
            raise ValueError("Empty dataset provided")

        logger.info(f"Training on {len(X)} samples, {X.shape[1]} features")

        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=val_size, random_state=random_state, stratify=y
        )

        if scale:
            self.scaler = StandardScaler()
            X_train = self.scaler.fit_transform(X_train)
            X_val = self.scaler.transform(X_val)

        self.model = get_model(
            self.model_type,
            input_size=X_train.shape[1],
            hidden_sizes=self.hidden_sizes,
            dropout=self.dropout,
            use_attention=self.use_attention,
        )

        train_dataset = TensorDataset(
            torch.FloatTensor(X_train), torch.FloatTensor(y_train)
        )
        val_dataset = TensorDataset(
            torch.FloatTensor(X_val), torch.FloatTensor(y_val)
        )

        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)

        pos_weight = torch.FloatTensor([len(y_train) / (y_train.sum() + 1) - 1])
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=0.5, patience=5
        )

        best_auc = 0
        patience_counter = 0

        for epoch in range(self.epochs):
            train_loss = self._train_epoch(self.model, train_loader, criterion, optimizer)
            val_loss, val_metrics = self._evaluate(self.model, val_loader, criterion)

            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["val_auc"].append(val_metrics["auc"])

            scheduler.step(val_metrics["auc"])

            if val_metrics["auc"] > best_auc:
                best_auc = val_metrics["auc"]
                patience_counter = 0
                self._save_best_state()
            else:
                patience_counter += 1

            if (epoch + 1) % 10 == 0:
                logger.info(
                    f"Epoch {epoch+1}/{self.epochs} - "
                    f"Train Loss: {train_loss:.4f}, "
                    f"Val Loss: {val_loss:.4f}, "
                    f"Val AUC: {val_metrics['auc']:.4f}"
                )

            if patience_counter >= self.early_stopping_patience:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break

        self._load_best_state()

        final_metrics = self._evaluate(
            self.model, val_loader, criterion
        )[1]

        logger.info(f"Training complete. Best Val AUC: {best_auc:.4f}")
        return final_metrics

    def _train_epoch(
        self,
        model: nn.Module,
        loader: DataLoader,
        criterion: nn.Module,
        optimizer: optim.Optimizer,
    ) -> float:
        model.train()
        total_loss = 0

        for X_batch, y_batch in loader:
            optimizer.zero_grad()
            outputs = model(X_batch).squeeze()
            if outputs.ndim == 0:
                outputs = outputs.unsqueeze(0)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        return total_loss / len(loader)

    def _evaluate(
        self,
        model: nn.Module,
        loader: DataLoader,
        criterion: nn.Module,
    ) -> Tuple[float, dict]:
        model.eval()
        total_loss = 0
        all_preds = []
        all_probs = []
        all_labels = []

        with torch.no_grad():
            for X_batch, y_batch in loader:
                outputs = model(X_batch).squeeze()
                if outputs.ndim == 0:
                    outputs = outputs.unsqueeze(0)
                loss = criterion(outputs, y_batch)
                total_loss += loss.item()

                probs = torch.sigmoid(outputs).numpy()
                preds = (probs >= 0.5).astype(int)

                if probs.ndim == 0:
                    probs = probs.reshape(1)
                    preds = preds.reshape(1)

                all_probs.extend(probs.tolist())
                all_preds.extend(preds.tolist())
                all_labels.extend(y_batch.numpy().tolist())

        all_probs = np.array(all_probs)
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)

        metrics = {
            "loss": total_loss / len(loader),
            "accuracy": accuracy_score(all_labels, all_preds),
            "precision": precision_score(all_labels, all_preds, zero_division=0),
            "recall": recall_score(all_labels, all_preds, zero_division=0),
            "f1": f1_score(all_labels, all_preds, zero_division=0),
            "auc": roc_auc_score(all_labels, all_probs),
            "confusion_matrix": confusion_matrix(all_labels, all_preds).tolist(),
        }

        return total_loss / len(loader), metrics

    def _save_best_state(self):
        self._best_state = {
            k: v.cpu().clone()
            for k, v in self.model.state_dict().items()
        }

    def _load_best_state(self):
        if hasattr(self, "_best_state"):
            self.model.load_state_dict(self._best_state)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Model not trained")
        
        self.model.eval()
        
        if self.scaler is not None:
            X = self.scaler.transform(X)
        
        X_tensor = torch.FloatTensor(X)
        with torch.no_grad():
            logits = self.model(X_tensor).squeeze()
            probs = torch.sigmoid(logits).numpy()
        
        if probs.ndim == 0:
            probs = probs.reshape(1)
        
        return probs

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        probs = self.predict_proba(X)
        return (probs >= threshold).astype(int)

    def save(self, path: Path):
        if self.model is None:
            raise RuntimeError("No model to save")
        
        state = {
            "model_state_dict": self.model.state_dict(),
            "model_type": self.model_type,
            "input_size": self.model.input_size,
            "hidden_sizes": self.hidden_sizes,
            "dropout": self.dropout,
            "history": self.history,
        }
        if self.scaler is not None:
            state["scaler_mean"] = self.scaler.mean_
            state["scaler_scale"] = self.scaler.scale_
        torch.save(state, path)
        logger.info(f"Model saved to {path}")

    def load(self, path: Path):
        state = torch.load(path, map_location="cpu", weights_only=False)
        
        self.model_type = state["model_type"]
        self.hidden_sizes = state["hidden_sizes"]
        self.dropout = state["dropout"]
        
        self.model = get_model(
            self.model_type,
            input_size=state["input_size"],
            hidden_sizes=self.hidden_sizes,
            dropout=self.dropout,
        )
        self.model.load_state_dict(state["model_state_dict"])
        self.history = state.get("history", {})
        
        if "scaler_mean" in state and "scaler_scale" in state:
            self.scaler = StandardScaler()
            self.scaler.mean_ = state["scaler_mean"]
            self.scaler.scale_ = state["scaler_scale"]
            self.scaler.var_ = state["scaler_scale"] ** 2
            self.scaler.n_features_in_ = state["scaler_mean"].shape[0]
            self.scaler.n_samples_seen_ = state["scaler_mean"].shape[0]
        else:
            self.scaler = None
        
        logger.info(f"Model loaded from {path}")