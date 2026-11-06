"""Data for domain and text pretraining."""
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader
from utils.data import read_lines, read_domains, read_embedding, read_annotations, label_matrix
from utils.ontology import GeneOntology


class FeatureDataset(Dataset):
    def __init__(self, task, feature, split, meta, data_path='data'):
        folder = Path(f'{data_path}/anno/cluster/{task}')
        self.ontology = GeneOntology(f'{data_path}/anno/go-basic.obo')
        self.ic_path = Path(data_path) / 'anno/IA.tsv'
        annotation = 'val_stage1.txt' if split == 'val' else f'{split}.txt'
        self.truth = read_annotations(folder / annotation, self.ontology)
        self.pids = [p for p in read_lines(folder / f'{split}.pid') if p in self.truth]
        self.feature = feature
        self.input_dim = len(meta['domains'])
        if feature == 'text':
            embeddings = Path(data_path) / 'data/embeddings/pubmedbert'
            self.pids = [p for p in self.pids if (embeddings / f'{p}.pt').is_file()]
            self.vectors = torch.stack([read_embedding(embeddings, p) for p in self.pids])
            self.input_dim = self.vectors.shape[1]
        else:
            self.domains = read_domains(f'{data_path}/data/interpro/{task}/supervised.feature')
        self.y = label_matrix(self.pids, self.truth, meta['labels'])

    def __len__(self):
        return len(self.pids)

    def __getitem__(self, i):
        if self.feature == 'text':
            x = self.vectors[i]
        else:
            x = torch.zeros(self.input_dim)
            x[self.domains[self.pids[i]]] = 1
        return {'index': i, 'input': x, 'y': self.y[i]}


class FeatureData:
    def __init__(self, task, feature, batch_size, hidden, data_path='data'):
        self.meta = dict(task=task, hidden=hidden, architecture='three_fc_layernorm',
                         domains=read_lines(f'{data_path}/data/interpro/{task}/interpro.table'),
                         labels=read_lines(f'{data_path}/anno/cluster/{task}/label_list_{task}.txt'))
        train = FeatureDataset(task, feature, 'train', self.meta, data_path)
        val = FeatureDataset(task, feature, 'val', self.meta, data_path)
        self.input_dim = train.input_dim
        self.n_labels = len(self.meta['labels'])
        self.train_loader = DataLoader(train, batch_size=batch_size, shuffle=True)
        self.val_loader = DataLoader(val, batch_size=batch_size)
