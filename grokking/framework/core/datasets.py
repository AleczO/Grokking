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
    

class DatasetCNA(Dataset): # Complex Numbers Dataset Asymetry
    def __init__(self, N, split="train"):
        self.N = N
        self.split = split
        
        self.offset = 2 * (N * N)
        self.output_offset = 2 * (N * N)

        self.inputs, self.outputs = self.GenerateDataset(N)
        self.size = self.inputs.shape[0]

    def __len__(self):
        return self.size
    
    def __getitem__(self, index):
        return self.inputs[index], self.outputs[index]
    
    def GenerateDataset(self, N):
        train_inputs, train_outputs = [], []
        test_inputs, test_outputs = [], []

        for a in range(-N, N + 1):
            for b in range(-N, N + 1):
                for c in range(-N, N + 1):
                    for d in range(-N, N + 1):
                        
                        inp = [
                            a + self.offset,
                            b + self.offset,
                            c + self.offset,
                            d + self.offset
                        ]

                        out = [
                            a * c - b * d + self.output_offset,
                            a * d + b * c + self.output_offset
                        ]

                        if (a, b) <= (c, d):
                            train_inputs.append(inp)
                            train_outputs.append(out)
                        else:
                            test_inputs.append(inp)
                            test_outputs.append(out)

        if self.split == "train":
            return torch.tensor(train_inputs, dtype=torch.long), torch.tensor(train_outputs, dtype=torch.long)
        else:
            return torch.tensor(test_inputs, dtype=torch.long), torch.tensor(test_outputs, dtype=torch.long)


class DatasetCNC(Dataset):
    def __init__(self, N, split="train"):
        self.N = N
        self.split = split
        
        self.offset = 2 * (N * N)
        self.output_offset = 2 * (N * N)

        self.inputs, self.outputs = self.GenerateDataset(N)
        self.size = self.inputs.shape[0]

    def __len__(self):
        return self.size
    
    def __getitem__(self, index):
        return self.inputs[index], self.outputs[index]
    
    def GenerateDataset(self, N):
        train_inputs, train_outputs = [], []
        test_inputs, test_outputs = [], []

        for a in range(-N, N + 1):
            for b in range(-N, N + 1):
                for c in range(-N, N + 1):
                    for d in range(-N, N + 1):
                        
                        inp = [
                            a + self.offset,
                            b + self.offset,
                            c + self.offset,
                            d + self.offset
                        ]

                        out = [
                            a * c - b * d + self.output_offset,
                            a * d + b * c + self.output_offset
                        ]

                        if b > 0 and d > 0:
                            train_inputs.append(inp)
                            train_outputs.append(out)
                        else:
                            test_inputs.append(inp)
                            test_outputs.append(out)

        if self.split == "train":
            return torch.tensor(train_inputs, dtype=torch.long), torch.tensor(train_outputs, dtype=torch.long)
        else:
            return torch.tensor(test_inputs, dtype=torch.long), torch.tensor(test_outputs, dtype=torch.long)

# ---------------------------------------------------------- Complex Numbers Matrix Form Datasets ----------------------------------------------------------


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

class DatasetCNFMA(Dataset): # Complex Numbers Full Matrix Output Assymetry
    def __init__(self, N, split="train"):
        self.N = N
        self.offset = 2 * (N * N)
        self.split = split
        
        self.inputs, self.outputs = self.GenerateDataset(N)
        self.size = self.inputs.shape[0]

    def __len__(self):
        return self.size
    
    def __getitem__(self, index):
        return self.inputs[index], self.outputs[index]
    
    def GenerateDataset(self, N):
        all_complex_numbers = []
        for x in range(-N, N + 1):
            for y in range(-N, N + 1):
                all_complex_numbers.append((x, y))

        target_inputs = []
        target_outputs = []

        for i in range(len(all_complex_numbers)):
            for j in range(i, len(all_complex_numbers)):
                
                a, b = all_complex_numbers[i]
                c, d = all_complex_numbers[j]
                
                re_val = a * c - b * d + self.offset
                im_val = a * d + b * c + self.offset
                neg_im_val = -(a * d + b * c) + self.offset
                out = [re_val, neg_im_val, im_val, re_val]

                inp_forward = [
                    a + self.offset, -b + self.offset, b + self.offset, a + self.offset,
                    c + self.offset, -d + self.offset, d + self.offset, c + self.offset
                ]
                
                inp_reversed = [
                    c + self.offset, -d + self.offset, d + self.offset, c + self.offset,
                    a + self.offset, -b + self.offset, b + self.offset, a + self.offset
                ]

                if self.split == "train":
                    target_inputs.append(inp_forward)
                    target_outputs.append(out)
                
                elif self.split == "val":
                    if i != j:
                        target_inputs.append(inp_reversed)
                        target_outputs.append(out)


        return torch.tensor(target_inputs, dtype=torch.long), torch.tensor(target_outputs, dtype=torch.long)
