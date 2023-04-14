import numpy as np
from torch.utils.data import Dataset, ConcatDataset
import torch
from sklearn.preprocessing import minmax_scale
import pandas as pd


class MTSDataset(Dataset):
    def __init__(self, raw_seqs, win_len=20, selected_features=None, minmax=True) -> None:
        super().__init__()

        if minmax:
            raw_seqs = minmax_scale(raw_seqs)

        self._raw_seqs = torch.tensor(raw_seqs, dtype=torch.float32)
        self._win_len = win_len


    def set_win_len(self, win_len):
        self._win_len = win_len
    

    def __len__(self):
        return len(self._raw_seqs) - self._win_len + 1

    
    def __getitem__(self, index):
        return self._raw_seqs[index : index + self._win_len]
    

class MTSLabeledDataset(Dataset):
    def __init__(self, raw_seqs, labels, win_len=20, minmax=True) -> None:
        super().__init__()

        if minmax:
            raw_seqs = minmax_scale(raw_seqs)

        self._raw_seqs = torch.tensor(raw_seqs, dtype=torch.float32)
        self._labels = torch.tensor(labels, dtype=torch.int).view(-1)
        assert len(self._raw_seqs) >= len(self._labels)

        # label: 0 normal, 1 anomaly, -1 unkonwn
        if len(self._labels) < len(self._raw_seqs):
            padding_size = len(self._raw_seqs) - len(self._labels)
            padding = torch.zeros(padding_size, dtype=torch.int) - 1
            self._labels = torch.cat([padding, self._labels])
        
        self._win_len = win_len


    def set_win_len(self, win_len):
        self._win_len = win_len
    

    def __len__(self):
        return len(self._raw_seqs) - self._win_len + 1

    
    def __getitem__(self, index):
        return self._raw_seqs[index : index + self._win_len], self._labels[index + self._win_len - 1]
    

class UTSDataset(Dataset):
    def __init__(self, raw_seqs, win_len=20, minmax=True) -> None:
        super().__init__()

        if minmax:
            raw_seqs = minmax_scale(raw_seqs)

        self._raw_seqs = torch.tensor(raw_seqs, dtype=torch.float32)
        self._win_len = win_len

    def __len__(self):
        return len(self._raw_seqs) - self._win_len + 1

    
    def __getitem__(self, index):
        return self._raw_seqs[index : index + self._win_len]

    def set_win_len(self, win_len):
        self._win_len = win_len


class UTSTestDataset(Dataset):
    def __init__(self, raw_seqs, labels, win_len=20, minmax=True) -> None:
        super().__init__()

        if minmax:
            raw_seqs = minmax_scale(raw_seqs)

        self._raw_seqs = torch.tensor(raw_seqs, dtype=torch.float32)
        self._labels = torch.tensor(labels, dtype=torch.bool)
        self._win_len = win_len

    def __len__(self):
        return len(self._raw_seqs) - self._win_len + 1

    
    def __getitem__(self, index):
        return self._raw_seqs[index : index + self._win_len], self._labels[index + self._win_len - 1]


    def set_win_len(self, win_len):
        self._win_len = win_len


def select_feedback_data(data, labels, pos, win_len):
    trian_cnt = data.shape[1] - labels.shape[1]
    
    feedback_pos = pos + trian_cnt
    begin_pos = feedback_pos - win_len
    end_pos = feedback_pos + win_len

    feedback_range = np.concatenate([begin_pos.reshape(-1, 1), end_pos.reshape(-1, 1)], axis=1)
    feedback_pos = np.array([np.arange(i[0], i[1]) for i in feedback_range])

    rows = np.tile(np.arange(feedback_pos.shape[0]).reshape(-1, 1), feedback_pos.shape[1])
    
    feedback_data = data[rows, feedback_pos]
    feedback_pos = feedback_pos - trian_cnt
    feedback_labels = labels[rows, feedback_pos]

    return feedback_data, feedback_labels


def get_feedback_data(win_len,
                      minmax=True,         
                      data_path="/root/Feedback/data/ChinaMobile_data.npy", 
                      label_path="/root/Feedback/data/ChinaMobile_label.npy"):
    
    data_mat = np.load(data_path)
    label_mat = np.load(label_path)
    pos = np.argmax(label_mat, axis=1)

    if minmax:
        for i in range(data_mat.shape[0]):
            data_mat[i, :, :] = minmax_scale(data_mat[i, :, :])

    feedback_data, feedback_labels = select_feedback_data(data_mat, label_mat, pos, win_len)
    train_datasets = [
            MTSDataset(
            raw_seqs=feedback_data[i],
            win_len=win_len,
            minmax=False
        ) for i in range(len(data_mat))
    ]

    # test_datasets = [
    #         MTSLabeledDataset(
    #         raw_seqs=feedback_data[i],
    #         labels=feedback_labels[i],
    #         win_len=win_len,
    #         minmax=False
    #     ) for i in range(len(data_mat))
    # ]

    return ConcatDataset(train_datasets)
    

def  get_concat_ChinaMobileDataset(
        win_len=20, 
        minmax=True,
        train_end_pos = -1,
        data_path="/root/Feedback/data/ChinaMobile_data.npy", 
        label_path="/root/Feedback/data/ChinaMobile_label.npy"
        ):

        data_mat = np.load(data_path)
        label_mat = np.load(label_path)

        if minmax:
            for i in range(data_mat.shape[0]):
                data_mat[i, :, :] = minmax_scale(data_mat[i, :, :])

        label_begin_pos = data_mat.shape[1] - label_mat.shape[1]
        if train_end_pos == -1:
            train_end_pos = label_begin_pos

        train_datasets = [
                MTSDataset(
                raw_seqs=data_mat[i][:train_end_pos],
                win_len=win_len,
                minmax=False
            ) for i in range(len(data_mat))
        ]

        test_datasets = [
                MTSLabeledDataset(
                raw_seqs=data_mat[i, train_end_pos: ],
                labels=label_mat[i, train_end_pos - label_begin_pos: ],
                win_len=win_len,
                minmax=False
            ) for i in range(len(data_mat))
        ]

        return ConcatDataset(train_datasets), ConcatDataset(test_datasets)


def get_test_by_entity(
        win_len=20, 
        minmax=True,
        train_end_pos = -1,
        data_path="/root/Feedback/data/ChinaMobile_data.npy", 
        label_path="/root/Feedback/data/ChinaMobile_label.npy"
        ):
    
        data_mat = np.load(data_path)
        label_mat = np.load(label_path)

        if minmax:
            for i in range(data_mat.shape[0]):
                data_mat[i, :, :] = minmax_scale(data_mat[i, :, :])

        label_begin_pos = data_mat.shape[1] - label_mat.shape[1]
        if train_end_pos == -1:
            train_end_pos = label_begin_pos

        test_datasets = [
                MTSLabeledDataset(
                raw_seqs=data_mat[i, train_end_pos: ],
                labels=label_mat[i, train_end_pos - label_begin_pos: ],
                win_len=win_len,
                minmax=False
            ) for i in range(len(data_mat))
        ]

        return test_datasets
    
