import torch
from torch.utils.data import Dataset

class ModuloDataset(Dataset):
    def __init__(self, p):
        self.input, self.output = self.generateData(p)
        self.size = p * p
        self.p = p

    def __len__(self):
        return self.size
    
    def __getitem__(self, index):
        return self.input[index], self.output[index]

    def generateData(self, p):
        i = 0
        input = torch.zeros([p * p, 2], dtype=torch.long)
        output = torch.zeros(p * p, dtype=torch.long)

        for a in range(p):
            for b in range(p):
                input[i][0] = a
                input[i][1] = b
                output[i] = (a + b) % p
                i += 1
                
        return input, output