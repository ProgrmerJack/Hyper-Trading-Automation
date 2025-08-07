from __future__ import annotations

"""Transformer-based model for high-frequency trading predictions."""

from dataclasses import dataclass
from typing import Optional

try:  # pragma: no cover - optional dependency
    import torch
    import torch.nn as nn
except Exception:  # pragma: no cover - gracefully handle missing torch
    torch = None  # type: ignore
    nn = None  # type: ignore


@dataclass
class TransformerConfig:
    """Configuration for :class:`HFTTransformer`"""

    d_model: int = 64
    nhead: int = 4
    num_layers: int = 2


class HFTTransformer(nn.Module):
    """Minimal wrapper around ``nn.Transformer`` for price prediction.

    The model consumes a sequence of feature vectors and outputs the
    probability that the next price movement is positive.  It is a very
    small architecture intended as a starting point for experimentation
    with transformer-based sequence models in this project.
    """

    def __init__(self, config: Optional[TransformerConfig] = None) -> None:
        if torch is None:  # pragma: no cover - executed only when torch missing
            raise ImportError("PyTorch is required to use HFTTransformer")
        super().__init__()
        config = config or TransformerConfig()
        self.transformer = nn.Transformer(
            d_model=config.d_model,
            nhead=config.nhead,
            num_encoder_layers=config.num_layers,
            num_decoder_layers=config.num_layers,
        )
        self.fc = nn.Linear(config.d_model, 1)

    def forward(self, src: "torch.Tensor") -> "torch.Tensor":  # pragma: no cover - simple wrapper
        """Run a forward pass.

        Parameters
        ----------
        src: torch.Tensor
            Input tensor of shape ``[seq_len, batch, features]``.
        """

        out = self.transformer(src, src)
        # Use the mean of the last dimension as a compact sequence summary
        return torch.sigmoid(self.fc(out.mean(dim=0)))


def train_transformer_model(df, config: Optional[TransformerConfig] = None) -> HFTTransformer:
    """Return a transformer model instance trained on price data.

    The current implementation is intentionally lightweight and does not
    perform real training to keep the dependency surface small.  It
    simply initialises the model and leaves training to the caller.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing at least a ``close`` column.  Additional
        feature engineering should be performed by the user before
        training a serious model.
    config : TransformerConfig, optional
        Custom configuration for the model.
    """

    if torch is None:  # pragma: no cover - optional dependency
        raise ImportError("PyTorch is required to train the transformer model")

    model = HFTTransformer(config)
    # Placeholder: users should implement their own training loop.
    model.eval()
    return model
