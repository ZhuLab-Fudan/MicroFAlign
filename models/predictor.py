"""Sequence, kingdom and gLM2 fusion for GO prediction."""
from utils.data import model_folder
from pathlib import Path
import torch
from utils.data import load
from torch import nn


def projection(inputs, outputs):
    return nn.Sequential(nn.Linear(inputs, outputs), nn.BatchNorm1d(outputs),
                         nn.ReLU(), nn.Dropout(.5))


class DNN(nn.Module):
    def __init__(self, out_dim, sequence_dim=1024, glm_dim=1280):
        super().__init__()
        self.dnn_go = projection(sequence_dim, 1024)
        self.dnn_sp = projection(9, 32)
        self.dnn_glm = projection(glm_dim, 1024)
        self.dnn_pre = nn.Sequential(nn.Linear(2080, out_dim), nn.Sigmoid())

    def forward(self, sequence, glm, kingdom):
        features = torch.cat((self.dnn_go(sequence), self.dnn_sp(kingdom.float()),
                              self.dnn_glm(glm)), dim=1)
        return self.dnn_pre(features)


def load_predictor(task, device, data_path='data', version='trained'):
    saved = load(model_folder(data_path, version) / f'stage3/{task}.pt')
    model = DNN(len(saved['labels']))
    model.load_state_dict(saved['state'])
    return model.to(device), saved['labels']


def save_predictor(model, labels, args, score, threshold):
    folder = Path(args.data_path) / 'model/stage3'
    folder.mkdir(parents=True, exist_ok=True)
    state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    torch.save(dict(state=state, labels=labels, task=args.task, score=score,
                    threshold=threshold, args=vars(args)), folder / f'{args.task}.pt')
