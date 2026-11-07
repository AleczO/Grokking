"""Synthetic datasets for the grokking experiments.

All complex-number datasets enumerate every combination of
a, b, c, d in [-N, N] and encode values as token ids by adding an offset
of 2*N^2, so that every operand and every result is a non-negative id
smaller than voc_size = 4*N^2 + 1.

Datasets with a deterministic split take ``split="train"`` or ``split="val"``.
The others are split randomly with ``DivideDataset``.
"""
import torch
from torch.utils.data import DataLoader, Dataset, random_split

SPLITS = ("train", "val")


def _check_split(split):
    if split not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS}, got {split!r}")


class _TensorPairDataset(Dataset):
    """Base class: stores (inputs, outputs) tensors and indexes into them."""

    def __len__(self):
        return self.inputs.shape[0]

    def __getitem__(self, index):
        return self.inputs[index], self.outputs[index]


# ---------------------------------------------------------- Modular Addition ----------------------------------------------------------

class DatasetM(_TensorPairDataset):
    """Modular addition (a + b) mod p.

    Input: [a, b] with a, b in [0, p). Output: a single token (a + b) mod p.
    Contains all p^2 pairs; split randomly.
    """

    def __init__(self, p):
        self.p = p
        self.inputs, self.outputs = self.GenerateDataset(p)

    def GenerateDataset(self, p):
        inputs = torch.zeros([p * p, 2], dtype=torch.long)
        outputs = torch.zeros(p * p, dtype=torch.long)

        i = 0
        for a in range(p):
            for b in range(p):
                inputs[i] = torch.tensor([a, b])
                outputs[i] = (a + b) % p
                i += 1

        return inputs, outputs


# ---------------------------------------------------------- Complex Numbers ----------------------------------------------------------

class DatasetCN(_TensorPairDataset):
    """Complex multiplication (a+bi)(c+di).

    Input: [a, b, c, d]. Output: [re, im] = [ac - bd, ad + bc].
    Contains all (2N+1)^4 examples; split randomly.
    """

    def __init__(self, N):
        self.N = N
        self.offset = 2 * (N * N)
        self.inputs, self.outputs = self.GenerateDataset(N)

    def GenerateDataset(self, N):
        inputs = torch.zeros([(2 * N + 1)**4, 4], dtype=torch.long)
        outputs = torch.zeros([(2 * N + 1)**4, 2], dtype=torch.long)

        i = 0
        for a in range(-N, N + 1):
            for b in range(-N, N + 1):
                for c in range(-N, N + 1):
                    for d in range(-N, N + 1):
                        inputs[i] = torch.tensor([
                            a + self.offset,
                            b + self.offset,
                            c + self.offset,
                            d + self.offset
                        ])
                        outputs[i] = torch.tensor([
                            a * c - b * d + self.offset,
                            a * d + b * c + self.offset
                        ])
                        i += 1

        return inputs, outputs


class DatasetCNA(_TensorPairDataset):
    """Complex multiplication with an operand-order split.

    Same task and encoding as DatasetCN.
    train: examples with (a, b) <= (c, d) in lexicographic order.
    val:   all remaining examples, i.e. (a, b) > (c, d).
    """

    def __init__(self, N, split="train"):
        _check_split(split)
        self.N = N
        self.split = split
        self.offset = 2 * (N * N)
        self.inputs, self.outputs = self.GenerateDataset(N)

    def GenerateDataset(self, N):
        inputs, outputs = [], []

        for a in range(-N, N + 1):
            for b in range(-N, N + 1):
                for c in range(-N, N + 1):
                    for d in range(-N, N + 1):
                        in_train = (a, b) <= (c, d)
                        if in_train != (self.split == "train"):
                            continue

                        inputs.append([
                            a + self.offset,
                            b + self.offset,
                            c + self.offset,
                            d + self.offset
                        ])
                        outputs.append([
                            a * c - b * d + self.offset,
                            a * d + b * c + self.offset
                        ])

        return torch.tensor(inputs, dtype=torch.long), torch.tensor(outputs, dtype=torch.long)


class DatasetCNC(_TensorPairDataset):
    """Complex multiplication with a sign-based split.

    Same task and encoding as DatasetCN.
    train: examples where both imaginary parts are positive (b > 0 and d > 0).
    val:   all remaining examples.
    """

    def __init__(self, N, split="train"):
        _check_split(split)
        self.N = N
        self.split = split
        self.offset = 2 * (N * N)
        self.inputs, self.outputs = self.GenerateDataset(N)

    def GenerateDataset(self, N):
        inputs, outputs = [], []

        for a in range(-N, N + 1):
            for b in range(-N, N + 1):
                for c in range(-N, N + 1):
                    for d in range(-N, N + 1):
                        in_train = b > 0 and d > 0
                        if in_train != (self.split == "train"):
                            continue

                        inputs.append([
                            a + self.offset,
                            b + self.offset,
                            c + self.offset,
                            d + self.offset
                        ])
                        outputs.append([
                            a * c - b * d + self.offset,
                            a * d + b * c + self.offset
                        ])

        return torch.tensor(inputs, dtype=torch.long), torch.tensor(outputs, dtype=torch.long)


# ---------------------------------------------------------- Complex Numbers, Matrix Form ----------------------------------------------------------

class DatasetCNFM(_TensorPairDataset):
    """Complex multiplication with complex numbers as 2x2 real matrices.

    a+bi is encoded as [[a, -b], [b, a]], flattened row-wise.
    Input: both operand matrices (8 tokens). Output: the product matrix
    [re, -im, im, re] (4 tokens). Contains all (2N+1)^4 examples; split randomly.
    """

    def __init__(self, N):
        self.N = N
        self.offset = 2 * (N * N)
        self.inputs, self.outputs = self.GenerateDataset(N)

    def GenerateDataset(self, N):
        inputs = torch.zeros([(2 * N + 1)**4, 8], dtype=torch.long)
        outputs = torch.zeros([(2 * N + 1)**4, 4], dtype=torch.long)

        i = 0
        for a in range(-N, N + 1):
            for b in range(-N, N + 1):
                for c in range(-N, N + 1):
                    for d in range(-N, N + 1):
                        inputs[i] = torch.tensor([
                            a + self.offset, -b + self.offset, b + self.offset, a + self.offset,
                            c + self.offset, -d + self.offset, d + self.offset, c + self.offset
                        ])

                        re_val = a * c - b * d + self.offset
                        im_val = a * d + b * c + self.offset
                        neg_im_val = -(a * d + b * c) + self.offset
                        outputs[i] = torch.tensor([re_val, neg_im_val, im_val, re_val])
                        i += 1

        return inputs, outputs


class DatasetCNFMA(_TensorPairDataset):
    """Matrix-form complex multiplication with an operand-order split.

    Same encoding as DatasetCNFM. Pairs of complex numbers are taken once
    per unordered pair (i <= j over a fixed enumeration).
    train: each pair in forward order (z_i, z_j), including i == j.
    val:   the same pairs with operands swapped (z_j, z_i), excluding i == j.
    """

    def __init__(self, N, split="train"):
        _check_split(split)
        self.N = N
        self.split = split
        self.offset = 2 * (N * N)
        self.inputs, self.outputs = self.GenerateDataset(N)

    def GenerateDataset(self, N):
        numbers = [(x, y) for x in range(-N, N + 1) for y in range(-N, N + 1)]

        inputs, outputs = [], []

        for i in range(len(numbers)):
            for j in range(i, len(numbers)):
                if self.split == "val" and i == j:
                    continue

                a, b = numbers[i]
                c, d = numbers[j]

                re_val = a * c - b * d + self.offset
                im_val = a * d + b * c + self.offset
                neg_im_val = -(a * d + b * c) + self.offset

                first = [a + self.offset, -b + self.offset, b + self.offset, a + self.offset]
                second = [c + self.offset, -d + self.offset, d + self.offset, c + self.offset]

                inputs.append(first + second if self.split == "train" else second + first)
                outputs.append([re_val, neg_im_val, im_val, re_val])

        return torch.tensor(inputs, dtype=torch.long), torch.tensor(outputs, dtype=torch.long)


# ---------------------------------------------------------- Registry & helpers ----------------------------------------------------------

DATASET_REGISTRY = {
    "M": DatasetM,
    "CN": DatasetCN,
    "CNA": DatasetCNA,
    "CNC": DatasetCNC,
    "CNFM": DatasetCNFM,
    "CNFMA": DatasetCNFMA,
}

# Datasets that build train/val deterministically via split="train"/"val".
# All others are split randomly with DivideDataset.
SPLIT_AWARE = {"CNA", "CNC", "CNFMA"}


def DivideDataset(dataset, train_frac):
    """Random train/validation split of a dataset."""
    train_size = int(train_frac * len(dataset))
    return random_split(dataset, [train_size, len(dataset) - train_size])


def to_tensors(dataset, device):
    """Load a whole dataset (Dataset or Subset) as (inputs, targets) on device."""
    inputs, targets = next(iter(DataLoader(dataset, batch_size=len(dataset))))
    return inputs.to(device), targets.to(device)


def BuildDatasets(name, N, train_frac):
    """Return (train_dataset, val_dataset) for a registered dataset name."""
    cls = DATASET_REGISTRY[name]

    if name in SPLIT_AWARE:
        return cls(N, split="train"), cls(N, split="val")

    return DivideDataset(cls(N), train_frac)