"""AI model architectures for VSA signal classification."""

from abc import ABC, abstractmethod
from typing import Optional, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger

from domain import ModelArchitecture, DEFAULT_DROPOUT


class BaseModel(ABC, nn.Module):
    """Abstract base class for all classifier models."""
    
    def __init__(self, input_size: int, dropout: float = DEFAULT_DROPOUT):
        super().__init__()
        self.input_size = input_size
        self.dropout = dropout
    
    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network."""
        pass
    
    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        """Predict class probabilities.
        
        Args:
            x: Input features array
            
        Returns:
            Probability predictions
        """
        self.eval()
        with torch.no_grad():
            x_tensor = torch.FloatTensor(x).to(next(self.parameters()).device)
            logits = self.forward(x_tensor)
            probs = torch.sigmoid(logits)
            return probs.cpu().numpy()
    
    def predict(self, x: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """Predict class labels.
        
        Args:
            x: Input features array
            threshold: Classification threshold
            
        Returns:
            Binary class predictions
        """
        probs = self.predict_proba(x)
        return (probs >= threshold).astype(int)
    
    def _init_weights(self, module: nn.Module) -> None:
        """Initialize weights using Xavier initialization."""
        if isinstance(module, (nn.Linear, nn.Conv1d)):
            if module.weight is not None:
                nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)


class TradeClassifier(BaseModel):
    """MLP-based trade classifier with optional attention mechanism."""
    
    def __init__(
        self,
        input_size: int,
        hidden_sizes: Optional[List[int]] = None,
        dropout: float = DEFAULT_DROPOUT,
        use_attention: bool = True,
    ):
        super().__init__(input_size, dropout)
        
        self.use_attention = use_attention
        hidden_sizes = hidden_sizes or [128, 64, 32]
        
        if use_attention:
            self.attention = nn.Sequential(
                nn.Linear(input_size, hidden_sizes[0] // 2),
                nn.Tanh(),
                nn.Linear(hidden_sizes[0] // 2, 1)
            )
        
        layers = []
        prev_size = input_size
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.BatchNorm1d(hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            prev_size = hidden_size
        
        layers.append(nn.Linear(prev_size, 1))
        self.features = nn.Sequential(*layers)
        
        # Initialize weights manually to avoid _apply issues with BatchNorm
        for module in self.modules():
            self._init_weights(module)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_attention:
            attn_weights = self.attention(x)
            attn_weights = F.softmax(attn_weights, dim=1)
            x = x * attn_weights
        
        return self.features(x)


class Conv1DClassifier(BaseModel):
    """1D CNN-based classifier for sequential feature extraction."""
    
    def __init__(
        self,
        input_size: int,
        context_window: int = 24,
        hidden_sizes: Optional[List[int]] = None,
        dropout: float = DEFAULT_DROPOUT,
    ):
        super().__init__(input_size, dropout)
        
        self.context_window = context_window
        hidden_sizes = hidden_sizes or [64, 32]
        
        features_per_step = input_size // context_window
        
        self.conv1 = nn.Conv1d(
            in_channels=features_per_step,
            out_channels=hidden_sizes[0],
            kernel_size=3,
            padding=1
        )
        self.conv2 = nn.Conv1d(
            in_channels=hidden_sizes[0],
            out_channels=hidden_sizes[1],
            kernel_size=3,
            padding=1
        )
        
        self.pool = nn.AdaptiveMaxPool1d(1)
        
        fc_input_size = hidden_sizes[1] + input_size
        
        self.fc = nn.Sequential(
            nn.Linear(fc_input_size, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1)
        )
        
        self._apply(self._init_weights)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)
        
        x_flat = x.view(batch_size, -1)
        
        x_conv = x.view(batch_size, -1, self.context_window)
        
        x_conv = F.relu(self.conv1(x_conv))
        x_conv = F.relu(self.conv2(x_conv))
        x_conv = self.pool(x_conv).squeeze(-1)
        
        x_combined = torch.cat([x_conv, x_flat], dim=1)
        
        return self.fc(x_combined)


class LSTMClassifier(BaseModel):
    """LSTM-based sequence classifier."""
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = DEFAULT_DROPOUT,
    ):
        super().__init__(input_size, dropout)
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_size=1,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )
        
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1)
        )
        
        self._init_lstm_weights()
    
    def _init_lstm_weights(self) -> None:
        """Initialize LSTM weights with orthogonal initialization."""
        for name, param in self.lstm.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(param)
            elif "weight_hh" in name:
                nn.init.orthogonal_(param)
            elif "bias" in name:
                nn.init.zeros_(param)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)
        
        x = x.unsqueeze(-1)
        
        lstm_out, (h_n, _) = self.lstm(x)
        
        last_output = lstm_out[:, -1, :]
        
        return self.fc(last_output)


def get_model(model_type: str, input_size: int, **kwargs) -> BaseModel:
    """Factory function to create model instances.
    
    Args:
        model_type: Type of model ('mlp', 'conv1d', 'lstm')
        input_size: Input feature dimension
        **kwargs: Additional model-specific arguments
        
    Returns:
        Initialized model instance
        
    Raises:
        ValueError: If model_type is not recognized
    """
    model_type = model_type.lower()
    
    if model_type == "mlp":
        return TradeClassifier(input_size, **kwargs)
    elif model_type == "conv1d":
        return Conv1DClassifier(input_size, **kwargs)
    elif model_type == "lstm":
        return LSTMClassifier(input_size, **kwargs)
    else:
        logger.warning(f"Неизвестный тип модели {model_type}, используется MLP")
        return TradeClassifier(input_size, **kwargs)
