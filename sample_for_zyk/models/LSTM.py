import numpy as np
import torch.nn as nn
import torch
import torch.optim as optim
from tqdm import trange

from .utils.evaluate import evaluate_by_entity
from .utils.log import record_res

class LSTM(nn.Module):
    def __init__(self, input_dim, win_len, hidden_dim, num_layers):
        super(LSTM, self).__init__()
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


def train(dataloader, params, writer, test_loaders=None):
    epoch_cnt = params.get("epoch_cnt")
    num_metric = params.get("num_metric")
    win_len = params.get("win_len")
    hidden_dim = params.get("hidden_dim")
    num_layers = params.get("num_layers")
    lr = params.get("lr")
    device = params.get("device")

    log_interval = 1


    model = LSTM(num_metric, win_len, hidden_dim, num_layers).to(device)  #type: LSTM
    optimizer = optim.Adam(model.parameters(), lr=lr)    
    loss_fn = nn.MSELoss()

    loss_ls = []
    global_step = 0
    for epoch in (pbar := trange(epoch_cnt)):
        model.train()
        for step, x in enumerate(dataloader):
            x = x.to(device).view(-1, win_len, num_metric)
            x_pred = model(x)
            loss = loss_fn(x[:, -1], x_pred)
            loss_ls.append(loss.item())

            if (step + 1) % log_interval == 0:
                avg_loss = np.average(loss_ls)
                pbar.set_description(f"Epoch: {epoch+1}, Loss: {avg_loss: .4f}")
                writer.add_scalar("loss", avg_loss, global_step=global_step)
                global_step += 1 
                
                loss_ls.clear()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # test
        if test_loaders is not None:
            res_ls = evaluate_by_entity(model, test_loaders, test, params)
            record_res(writer, res_ls, epoch+1)


    return model


def test(model: LSTM, dataloader, params):
    num_metric = params.get("num_metric")
    device = params.get("device")

    labels, raw_seq, est_seq, loss = [], [], [], []
    model.eval()
    loss_function = nn.MSELoss()
    with torch.no_grad():
        for x, y in dataloader:
            labels.append(y.numpy())

            x = x.to(device).view(-1, model.win_len, num_metric)
            x_pred = model(x)

            loss.append(loss_function(x[:, -1], x_pred).cpu().item())
            raw_seq.append(x.squeeze()[:, -1].cpu().numpy())
            est_seq.append(x_pred.squeeze().cpu().numpy())

    
    raw_seq = np.concatenate(raw_seq, axis=0)
    est_seq = np.concatenate(est_seq, axis=0)
    labels = np.concatenate(labels, axis=0)

    return raw_seq, est_seq, np.average(loss), labels
