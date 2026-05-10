import torch
from torch.utils.data import Dataset

class complexNumbers(Dataset):
    def __init__(self, N):
        self.N = N
        self.size = (2 * N + 1)**4

        self.offset = N
        self.output_offset = 2 * (N * N)

        self.inputs, self.outputs = self.GenerateDataset(N)

    def __len__(self):
        return self.size
    
    def __getitem__(self, index):
        return self.inputs[index], self.outputs[index]
    
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
                            a * c - b * d + self.output_offset,
                            a * d + b * c + self.output_offset
                        ])
                        i += 1

        return inputs, outputs
