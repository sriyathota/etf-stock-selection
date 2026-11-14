import torch
import torch.nn as nn
import torch.nn.functional as F

class Matryoshka_Linear_Layer(nn.Module):
    def __init__(self, input_dim, output_dim, nesting_list, efficient=False):
        super(Matryoshka_Linear_Layer, self).__init__()
        self.nesting_list = nesting_list
        self.efficient = efficient

        if self.efficient:
            # Efficient mode: Use a single weight matrix and slice it
            max_output_dim = max(nesting_list)
            self.linear = nn.Linear(input_dim, max_output_dim)
        else:
            # Standard mode: Separate linear layers for each nesting level
            self.linears = nn.ModuleList([nn.Linear(input_dim, dim) for dim in nesting_list])

    def forward(self, x):
        outputs = []
        if self.efficient:
            full_output = self.linear(x)
            for dim in self.nesting_list:
                outputs.append(torch.nn.functional.linear(x[:, :dim], self.linear.weight[:, :dim], self.linear.bias))
        else:
            for linear in self.linears:
                outputs.append(linear(x))
        return outputs

class MatryoshkaLoss(nn.Module):
    def __init__(self, relative_importance=None):
        super(MatryoshkaLoss, self).__init__()
        self.criterion = nn.CrossEntropyLoss()
        self.relative_importance = relative_importance

    def forward(self, outputs, target):
        total_loss = 0
        for i, output in enumerate(outputs):
            loss = self.criterion(output, target)
            if self.relative_importance:
                loss *= self.relative_importance[i]
            total_loss += loss
        return total_loss
