import pytest
import torch
from lit_auto_encoder.auto_encoder import LitAutoEncoder
from torch import nn


@pytest.fixture
def autoencoder() -> LitAutoEncoder:
    """Pytest fixture to instantiate LitAutoEncoder."""
    encoder = nn.Sequential(nn.Linear(28 * 28, 64), nn.ReLU(), nn.Linear(64, 3))
    decoder = nn.Sequential(nn.Linear(3, 64), nn.ReLU(), nn.Linear(64, 28 * 28))
    return LitAutoEncoder(encoder, decoder)


def test_training_step(autoencoder) -> None:
    """Test training loop based on loss and output as a tensor."""
    batch = (torch.rand(32, 1, 28, 28), torch.rand(32, 1, 28, 28))
    batch_idx = 0
    loss = autoencoder.training_step(batch, batch_idx)
    assert loss is not None
    assert isinstance(loss, torch.Tensor)


def test_configure_optimizers(autoencoder) -> None:
    """Test optimizer."""
    optimizer = autoencoder.configure_optimizers()
    assert optimizer is not None
    assert isinstance(optimizer, torch.optim.Adam)
