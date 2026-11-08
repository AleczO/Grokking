"""Every registered experiment builds and trains for a few steps on CPU.

Run with `pytest` from the repository root.
"""
import math

import pytest
import torch

from configs import EXPERIMENTS
from core import BuildDatasets, GetModel, TrainEpoch, Validate, to_tensors

SMALL_N = 2   # operand range [-2, 2] for complex tasks
SMALL_P = 7   # modulus for modular addition


def small_size(dataset_name):
    return SMALL_P if dataset_name == "M" else SMALL_N


@pytest.mark.parametrize("name", sorted(EXPERIMENTS))
def test_experiment_trains(name):
    torch.manual_seed(0)
    config = EXPERIMENTS[name](small_size(EXPERIMENTS[name](1)["experiment"]["dataset"]))
    exp = config["experiment"]

    train_ds, val_ds = BuildDatasets(exp["dataset"], exp["N"], train_frac=0.5)
    assert len(train_ds) > 0 and len(val_ds) > 0

    train_x, train_y = to_tensors(train_ds, "cpu")
    val_x, val_y = to_tensors(val_ds, "cpu")

    model = GetModel(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.1)

    for _ in range(3):
        loss, acc = TrainEpoch(model, train_x, train_y, optimizer)
        assert math.isfinite(loss) and 0.0 <= acc <= 1.0

    loss, acc = Validate(model, val_x, val_y)
    assert math.isfinite(loss) and 0.0 <= acc <= 1.0