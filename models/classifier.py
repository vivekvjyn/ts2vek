from models.encoder import TSEncoder
import torch
from torch import nn
import torch.nn.functional as F
import os


class Classifier(nn.Module):
    def __init__(self, name, num_classes, input_dims=2, output_dims=16, hidden_dims=32, depth=1, mask_mode='binomial', dropout=0.3):
        super(Classifier, self).__init__()

        self.dir = 'checkpoints/finetuned'
        self.filename = f'{name}.pth'

        self.encoder = TSEncoder(
            input_dims=input_dims,
            output_dims=output_dims,
            hidden_dims=hidden_dims,
            depth=depth,
            mask_mode=mask_mode
        )


        self.classifier = nn.Sequential(
            nn.BatchNorm1d(output_dims),
            nn.Dropout(dropout),
            nn.Linear(output_dims, hidden_dims),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dims),
            nn.Dropout(dropout),
            nn.Linear(hidden_dims, num_classes)
        )


    def forward(self, x, mask=None):
        z = self.encode(x, mask)

        out = self.classifier(z)
        return out


    def save(self):
        os.makedirs(self.dir, exist_ok=True)
        torch.save(self.encoder.state_dict(), os.path.join(self.dir, self.filename))


    def load(self, device):
        self.load_state_dict(torch.load(os.path.join(self.dir, self.filename), map_location=device))


    def load_encoder(self, device, path):
        state_dict = torch.load(path, map_location=device)
        self.encoder.load_state_dict(state_dict)


    def set_trainable(self, mode=True):
        for param in self.encoder.parameters():
            param.requires_grad = mode


    def encode(self, x, mask=None, encoding_window="full_series"):
        z = self.encoder(x, mask)

        if encoding_window == "full_series":
            reprs = F.max_pool1d(
                z.transpose(1, 2).contiguous(), kernel_size=z.size(1)
            ).squeeze(-1)
        elif isinstance(encoding_window, int):
            reprs = F.max_pool1d(
                z.transpose(1, 2).contiguous(), kernel_size=encoding_window, stride=1
            ).transpose(1, 2)
        else:
            reprs = z.mean(dim=1)

        return reprs
