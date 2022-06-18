import torch
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


class LstmNeuralNet(nn.Module):
    def __init__(self, input_size,
                lstm_hidden_size,
                lstm_num_layers,
                linear_hidden_size_1,
                linear_hidden_size_2,
                num_classes,
                cfg):
        super(LstmNeuralNet, self).__init__()
        self.input_size = input_size
        self.lstm_num_layers = lstm_num_layers
        self.CFG = cfg
        self.lstm_hidden_size = lstm_hidden_size
        self.lstm_layer = nn.LSTM(input_size, lstm_hidden_size, lstm_num_layers)
        self.linear_layer_1 = nn.Linear(lstm_hidden_size, linear_hidden_size_1)
        self.linear_layer_2 = nn.Linear(linear_hidden_size_1, linear_hidden_size_2)
        self.linear_layer_3 = nn.Linear(linear_hidden_size_2, num_classes)

    def init_h0(self, batch_size):
        return torch.randn(self.lstm_num_layers, batch_size, self.lstm_hidden_size)

    def init_c0(self, batch_size):
        return torch.randn(self.lstm_num_layers, batch_size, self.lstm_hidden_size)

    def forward(self, X):
        X = X.view(self.CFG['seq_length'], -1, self.input_size)
        h0 = self.init_h0(X.size()[1])
        c0 = self.init_c0(X.size()[1])

        output, (hn, cn) = self.lstm_layer(X, (h0, c0))
        hidden_state = self.linear_layer_1(hn[-1])
        hidden_state = self.linear_layer_2(F.relu(hidden_state))
        out = self.linear_layer_3(F.relu(hidden_state))
        return out
