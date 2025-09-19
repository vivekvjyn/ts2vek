'''
Experiment 2: Pretrain on Carnatic Varnam and Bhairavi, Finetune on Bhairavi
'''


import argparse
import os
import random
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import pickle
from ts2vec import TS2Vec
import json
from models.classifier import Classifier

from modules.dataset import Dataset
from modules.logger import Logger
from modules.trainer import Trainer

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

lr = 0.001
batch_size = 256
input_dims = 2
hidden_dims = 64
output_dims = 32
depth = 4
dropout = 0.3
mask_mode = 'binomial'
exp = 'pretrain-varnam+bhairavi_finetune-bhairavi'
pretrain_epochs = 100
finetune_epochs = 300
num_classes = 7
catchup_epochs = 20

params = {
    'num_classes': num_classes,
    'batch_size': batch_size,
    'exp': exp,
    'hidden_dims': hidden_dims,
    'output_dims': output_dims,
    'depth': depth,
    'dropout': dropout,
    'mask_mode': mask_mode,
    'model_path': f'checkpoints/finetuned/{exp}.pth',
    'input_dims': input_dims,
    'train_path': 'dataset/train.pkl',
    'test_path': 'dataset/test.pkl',
    'lr': lr,
    'pretrain_on': ['carnatic_varnam', 'bhairavi'],
    'finetune_on': ['bhairavi'],
    'pretrain_epochs': pretrain_epochs,
    'finetune_epochs': finetune_epochs,
    'catchup_epochs': catchup_epochs,
}


logger = Logger(exp)

def prepare_data(series_list):
    N = len(series_list)
    F = series_list[0].shape[1]
    Tmax = max(seq.shape[0] for seq in series_list)

    X = np.full((N, Tmax, F), np.nan, dtype=np.float32)
    for i, seq in enumerate(series_list):
        T = seq.shape[0]
        X[i, :T, :] = seq

    return X

def main():
    pretrain()

    train()

def pretrain():
    dataset_path = params['train_path']

    with open(dataset_path, 'rb') as f:
        all_svaras = pickle.load(f)

    series_list = [x['curr'][0] for x in all_svaras if x['dataset'] in params['pretrain_on']]

    X = prepare_data(series_list)
    print(f"Data shape: {X.shape}")
    model = TS2Vec(
        input_dims=input_dims,
        device='cuda',
        output_dims=output_dims,
        hidden_dims=hidden_dims,
        depth=depth,
        lr=lr,
        batch_size=batch_size)

    print("Starting pretraining...")

    loss_log = model.fit(
        X,
        name=exp,
        n_epochs=pretrain_epochs,
        verbose=True
    )


def train():
    train_path = params['train_path']
    train_dataset = Dataset(train_path, device, filter=params['finetune_on'], mode='train')
    train_loader = DataLoader(train_dataset['ts2vec'], batch_size=batch_size)


    val_dataset = Dataset(train_path, device, filter=params['finetune_on'], mode='validation')
    val_loader = DataLoader(val_dataset['ts2vec'], batch_size=batch_size)

    model = Classifier(name=exp, num_classes=num_classes, input_dims=input_dims, output_dims=output_dims, hidden_dims=hidden_dims, depth=depth, mask_mode=mask_mode, dropout=dropout).to(device)

    model.load_encoder(device, path=f"checkpoints/pretrained/{exp}.pth")

    trainer = Trainer(model, exp, logger)
    f1 = trainer(train_loader, val_loader, F.cross_entropy, finetune_epochs, lr / 10, catchup_epochs=catchup_epochs)

    params['f1'] = f1

    os.makedirs(f"params", exist_ok=True)

    with open(f'params/{exp}.json', 'w') as f:
        json.dump(params, f, indent=4)


if __name__ == '__main__':
    main()
