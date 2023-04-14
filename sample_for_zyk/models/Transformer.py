from sklearn.preprocessing import minmax_scale
import numpy as np
import torch.nn as nn
import torch
import torch.optim as optim
from time import time
from sklearn.metrics import mean_squared_error as mse, mean_absolute_error as mae
from tqdm import trange

class Transformer(nn.Module):
    def __init__(self, input_dim, nhead,  win_len, hidden_dim, num_layers):
        super(Transformer, self).__init__()
        self.input_dim = input_dim
        self.win_len = win_len
        self.hidden_dim = hidden_dim

        self.transformer = nn.Transformer(
            d_model=input_dim,
            nhead=nhead,
            batch_first=True,
            dim_feedforward=hidden_dim,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
        )
        self.hiddent2out = nn.Linear(hidden_dim, self.input_dim)

    def forward(self, x):
        return self.transformer(x, x)

