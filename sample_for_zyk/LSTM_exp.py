
import torch
from torch import optim
from torch import nn
from torch.utils.data import DataLoader
import numpy as np

from models.LSTM import train, test, LSTM
from utils.dataset import get_concat_ChinaMobileDataset, get_test_by_entity, get_feedback_data
from models.utils.evaluate import evaluate_by_entity
from utils.log import record_res

from torch.utils.tensorboard import SummaryWriter 

import time

note = "LSTM模型"

starttime = time.strftime("%Y-%m-%d_%H-%M")
print("Start experiment:", starttime)
writer = SummaryWriter(log_dir="./log/" + note + starttime, comment=starttime, flush_secs=60)

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic=True

setup_seed(2023)

params = {
    'num_metric': 25,
    'win_len': 20,
    'hidden_dim': 128,
    'lr': 1e-3,
    'batch_size': 5120,
    'epoch_cnt': 200,
    'num_layers': 1,
    'delay': 0,
    'model': "LSTM",
    'device': 'cpu',
}


win_len = params.get("win_len")
batch_size = params.get("batch_size")

writer.add_hparams(params, {"exp": 0.1})


data_path="data/ChinaMobile_online_data.npy"
label_path="data/ChinaMobile_online_label.npy"


train_dataset, test_dataset = get_concat_ChinaMobileDataset(win_len=win_len, data_path=data_path, label_path=label_path)
test_entities = get_test_by_entity(win_len=win_len, data_path=data_path, label_path=label_path)
feedback_dataset = get_feedback_data(win_len=win_len, data_path=data_path, label_path=label_path)


train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False)
test_loaders = [DataLoader(i, batch_size=batch_size, shuffle=False, drop_last=False) for i in test_entities]


# ======================= Train ================================
model = train(train_loader, params, writer, test_loaders=test_loaders) # type: LSTM
torch.save(model.state_dict(), "./saved_models/LSTM.pt")


model = LSTM(
    input_dim=params.get("num_metric"),
    win_len=win_len,
    hidden_dim=params.get("hidden_dim"),
    num_layers=params.get("num_layers"),
)


model.load_state_dict(torch.load("./saved_models/LSTM.pt"))
model.to(params.get('device'))
model.eval()

res = evaluate_by_entity(model, test_loaders, test, params)
point_res, range_res = record_res(writer, res, step=params.get('epoch_cnt'))

writer.add_hparams(params, range_res)

print("point res:")
print(point_res)

print("range res:")
print(range_res)

fine_tune_loader = DataLoader(feedback_dataset, batch_size=batch_size, shuffle=True, drop_last=False)
model = train(fine_tune_loader, params, writer, test_loaders=test_loaders)
res = evaluate_by_entity(model, test_loaders, test, params)
