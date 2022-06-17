import torch.nn as nn
import torch.nn.functional as F


class MlpNeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(MlpNeuralNet, self).__init__()
        self.l1 = nn.Linear(input_size, hidden_size)
        self.l2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = F.relu(self.l1(x))
        out = self.l2(out)
        return out