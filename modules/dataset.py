import os
import pickle
import torch
from torch.utils.data import TensorDataset
from torch.nn.utils.rnn import pad_sequence
import numpy as np

def prepare_data(series_list):
    N = len(series_list)
    F = series_list[0].shape[1]
    Tmax = max(seq.shape[0] for seq in series_list)

    X = np.full((N, Tmax, F), np.nan, dtype=np.float32)
    for i, seq in enumerate(series_list):
        T = seq.shape[0]
        X[i, :T, :] = seq

    return X

class Dataset:
    def __init__(self, path, device, filter=['bhairavi'], mode='train'):
        with open(os.path.join(path), 'rb') as f:
            dataset = pickle.load(f)

        self.device = device

        svaras = ['S', 'R', 'G', 'M', 'P', 'D', 'N', 'x']

        series_list = [x['curr'][0] for x in dataset if x['dataset'] in filter and x['fold'] == mode and x['curr'][1] != 'x']
        X = prepare_data(series_list)

        self.x = torch.tensor(X, dtype=torch.float32)

        self.targets = torch.tensor([svaras.index(x['curr'][1]) for x in dataset if x['dataset'] in filter and x['fold'] == mode and x['curr'][1] != 'x'], dtype=torch.int64)
        self.modes = torch.tensor([0 for x in dataset if x['dataset'] in filter and x['fold'] == mode and x['curr'][1] != 'x'], dtype=torch.int64)


    def __getitem__(self, architecture):
        if architecture == 'ts2vec':
            return TensorDataset(
                self.x.to(self.device),
                self.targets.to(self.device),
                self.modes.to(self.device)
            )
        else:
            raise ValueError


    def __len__(self):
        return len(self.targets)


    @property
    def num_classes(self):
        return len(torch.unique(self.targets))


def pad(sequences):
    padded = pad_sequence(sequences, batch_first=True)
    return torch.nan_to_num(padded, nan=0).to(torch.float32)


def mask_padding(sequences):
    padded = pad_sequence(sequences, batch_first=True, padding_value=torch.inf)
    return padded == torch.inf


def mask_silence(sequences):
    padded = pad_sequence(sequences, batch_first=True)
    return torch.isnan(padded).to(torch.float32)


def lengths(sequences):
    return torch.tensor([len(sequence) for sequence in sequences], dtype=torch.int64)
