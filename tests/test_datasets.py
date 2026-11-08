"""Datasets produce correct targets, valid token ids and clean train/val splits.

Run with `pytest` from the repository root.
"""
import pytest
import torch

from core import DATASET_REGISTRY, SPLIT_AWARE

SMALL_N = 2   # operand range [-2, 2] for complex tasks
SMALL_P = 7   # modulus for modular addition


def small_size(dataset_name):
    return SMALL_P if dataset_name == "M" else SMALL_N


def all_examples(name):
    """Concatenate train and val of a dataset into (inputs, outputs)."""
    size = small_size(name)
    cls = DATASET_REGISTRY[name]
    parts = [cls(size, split="train"), cls(size, split="val")] if name in SPLIT_AWARE else [cls(size)]
    return torch.cat([p.inputs for p in parts]), torch.cat([p.outputs for p in parts])


@pytest.mark.parametrize("name", ["CN", "CNA", "CNC"])
def test_complex_targets(name):
    off = 2 * SMALL_N ** 2
    x, y = all_examples(name)
    a, b, c, d = (x - off).T
    assert torch.equal(y[:, 0] - off, a * c - b * d)
    assert torch.equal(y[:, 1] - off, a * d + b * c)


@pytest.mark.parametrize("name", ["CNFM", "CNFMA"])
def test_matrix_targets(name):
    off = 2 * SMALL_N ** 2
    x, y = all_examples(name)
    A = (x[:, :4] - off).view(-1, 2, 2)
    B = (x[:, 4:] - off).view(-1, 2, 2)
    assert torch.equal((y - off).view(-1, 2, 2), A @ B)


def test_modular_targets():
    x, y = all_examples("M")
    assert torch.equal(y, (x[:, 0] + x[:, 1]) % SMALL_P)


@pytest.mark.parametrize("name", sorted(DATASET_REGISTRY))
def test_tokens_in_vocab(name):
    size = small_size(name)
    voc_size = size if name == "M" else 4 * size ** 2 + 1
    x, y = all_examples(name)
    for t in (x, y):
        assert t.min() >= 0 and t.max() < voc_size


def test_split_sizes():
    n = (2 * SMALL_N + 1) ** 2  # number of complex numbers
    for name in ["CNA", "CNC"]:
        x, _ = all_examples(name)
        assert len(x) == n ** 2  # train + val cover every pair exactly once
        assert len(torch.unique(x, dim=0)) == n ** 2

    train = DATASET_REGISTRY["CNFMA"](SMALL_N, split="train")
    val = DATASET_REGISTRY["CNFMA"](SMALL_N, split="val")
    assert len(train) == n * (n + 1) // 2
    assert len(val) == n * (n - 1) // 2