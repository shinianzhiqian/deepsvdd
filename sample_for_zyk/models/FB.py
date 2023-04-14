from sklearn.preprocessing import minmax_scale
import numpy as np
import torch.nn as nn
import torch
import torch.optim as optim
from time import time
from sklearn.metrics import mean_squared_error as mse, mean_absolute_error as mae
from tqdm import trange

class FB(nn.Module):
    def __init__(self, input_dim, win_len, hidden_dim, num_layers):
        super(FB, self).__init__()
        self.input_dim = input_dim
        self.win_len = win_len
        self.hidden_dim = hidden_dim

        self.lstm = nn.LSTM(
            self.input_dim, 
            hidden_dim, 
            num_layers=num_layers,
            batch_first=True,
            )
        self.hiddent2out = nn.Linear(hidden_dim, self.input_dim)

    def forward(self, seq):
        lstm_out, _ = self.lstm(seq.view(-1, self.win_len, self.input_dim))
        predict = self.hiddent2out(lstm_out)
        return predict[:, -1, :]
