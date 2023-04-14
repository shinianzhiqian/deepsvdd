import torch
from torch import nn
import torch.nn.functional as F
from torch.optim import Adam
from tqdm import tqdm, trange
import numpy as np



class FBVAE(nn.Module):
    def __init__(self, num_metric, win_len=100, h_dim=500, z_dim=20):
        super(FBVAE, self).__init__()

        self.num_metric = num_metric
        self.win_len = win_len

        self.lstm = nn.LSTM(
            num_metric, 
            h_dim, 
            num_layers=1,
            batch_first=True,
            )

        # self.fc1 = nn.Linear(num_metric, h_dim)
        self.fc2 = nn.Linear(h_dim, z_dim)  # 均值 向量
        self.fc3 = nn.Linear(h_dim, z_dim)  # 保准方差 向量
        self.fc4 = nn.Linear(z_dim, h_dim)
        self.fc5 = nn.Linear(h_dim, num_metric)

    def encode(self, x):
        lstm_out, _ = self.lstm(x.view(-1, self.win_len, self.num_metric))
        x = lstm_out[:, -1, :]

        h = F.relu(x)
        # h = F.relu(self.fc1(x))
        return self.fc2(h), self.fc3(h)

    def reparameterize(self, mu, log_var):
        std = torch.exp(log_var / 2)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = F.relu(self.fc4(z))
        return torch.sigmoid(self.fc5(h))

    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        x_reconst = self.decode(z)
        return x_reconst, mu, log_var

    @staticmethod
    def loss_function(x, x_reconst, mu, log_var):
        reconst_loss = F.binary_cross_entropy(x_reconst, x, reduction='mean')
        kl_div = - 0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())

        loss = reconst_loss + kl_div

        return loss


def train(params, dataloader, device='cpu'):
    log_interval = 1

    win_len = params.get("win_len")
    h_dim = params.get("win_len")
    z_dim = params.get("z_dim")
    epoch_cnt = params.get("epoch_cnt")
    lr = params.get("lr")


    model = FBVAE(win_len=win_len, h_dim=h_dim, z_dim=z_dim).to(device)
    model.train()

    optimizer = Adam(model.parameters(), lr=lr)
    loss_fn = model.loss_function


    loss_ls = []
    for epoch in (pbar := trange(epoch_cnt)):
        for step, x in enumerate(dataloader):
            x = x.to(device)
            x_recons, mu, log_var = model(x)

            loss = loss_fn(x, x_recons, mu, log_var)

            loss_ls.append(loss.item())
            if (step + 1) % log_interval == 0:
                pbar.set_description(f"Epoch: {epoch+1}, Loss: {np.average(loss_ls): .4f}")
                loss_ls.clear()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    return model


def test(model: FBVAE, dataloader, device='cpu'):
    labels, raw_seq, est_seq, loss = [], [], [], []
    model.eval()
    with torch.no_grad():
        for x, y in dataloader:
            labels.append(y.numpy())

            x = x.to(device)
            x_recons, mu, var = model(x)

            loss.append(model.loss_function(x, x_recons, mu, var).cpu().item())
            raw_seq.append(x[:, -1].cpu().numpy())
            est_seq.append(x_recons[:, -1].cpu().numpy())

    
    raw_seq = np.concatenate(raw_seq, axis=0)
    est_seq = np.concatenate(est_seq, axis=0)
    labels = np.concatenate(labels, axis=0)

    return raw_seq, est_seq, np.average(loss), labels
