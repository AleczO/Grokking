import torch
from torch.utils.data import Dataset

# ---------------------------------------------------------- Modulo Numbers Dataset ----------------------------------------------------------

class DatasetM(Dataset): # Modulo Numbers Dataset
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

# ---------------------------------------------------------- Complex Numbers Datasets ----------------------------------------------------------

class DatasetCN(Dataset): # Complex Numbers Dataset
    def __init__(self, N):
        self.N = N
        self.size = (2 * N + 1)**4

        self.offset = 2 * (N * N)
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


class DatasetCNUC(Dataset): # Complex Numbers on Unit Circle Dataset (Work in progress)
    def __init__(self, N):
        self.N = N

        self.offset = 2 * (N * N)
        self.output_offset = 2 * (N * N)


    def __len__(self):
        return self.N

    def __getitem__(self, index):
        return self.inputs[index], self.output[index]

    def GenerateDataset(self, N):
        inputs = torch.zeros([N, 4], dtype=torch.float32)
        outputs = torch.zeros([N, 2], dtype=torch.float32)

        i = 0
        for i in range(N):
            theta1 = 2 * torch.pi * torch.rand(1)
            a, b = torch.cos(theta1), torch.sin(theta1)

            theta2 = 2 * torch.pi * torch.rand(1)
            c, d = torch.cos(theta2), torch.sin(theta2)

            inputs[i] = torch.tensor([a, b, c, d])
            
            outputs[i] = torch.tensor([
                a * c - b * d,
                a * d + b * c
            ])

        return inputs, outputs


class DatasetCNFM(Dataset): # Complex Numbers Full Matrix Output
    def __init__(self, N):
        self.N = N
        self.size = (2 * N + 1)**4
        self.offset = 2 * (N * N)

        self.inputs, self.outputs = self.GenerateDataset(N)

    def __len__(self):
        return self.size
    
    def __getitem__(self, index):
        return self.inputs[index], self.outputs[index]
    
    def GenerateDataset(self, N):
        inputs = torch.zeros([(2 * N + 1)**4, 8], dtype=torch.long)
        outputs = torch.zeros([(2 * N + 1)**4, 4 ], dtype=torch.long)

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
    

class DatasetCNM(Dataset): # Complex Numbers Real-Imaginary Matrix Output
    def __init__(self, N):
        self.N = N
        self.size = (2 * N + 1)**4
        self.offset = 2 * (N * N)
        self.output_offset = 2 * (N * N)

        self.inputs, self.outputs = self.GenerateDataset(N)

    def __len__(self):
        return self.size
    
    def __getitem__(self, index):
        return self.inputs[index], self.outputs[index]
    
    def GenerateDataset(self, N):
        inputs = torch.zeros([(2 * N + 1)**4, 8], dtype=torch.long)
        outputs = torch.zeros([(2 * N + 1)**4, 2], dtype=torch.long)

        i = 0
        for a in range(-N, N + 1):
            for b in range(-N, N + 1):
                for c in range(-N, N + 1):
                    for d in range(-N, N + 1):
                        
                        inputs[i] = torch.tensor([
                            a + self.offset, -b + self.offset, b + self.offset, a + self.offset,
                            c + self.offset, -d + self.offset, d + self.offset, c + self.offset
                        ])

                        re_val = a * c - b * d + self.output_offset
                        im_val = a * d + b * c + self.output_offset

                        outputs[i] = torch.tensor([re_val, im_val])
                        i += 1

        return inputs, outputs
    

