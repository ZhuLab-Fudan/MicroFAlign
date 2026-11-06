"""Sequences, domain indices and text embeddings for alignment."""
import json
from functools import partial
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader
from utils.data import read_lines, read_fasta, read_domains, read_embedding, load
from utils.contrastive import alignment_batch


class AlignmentData(Dataset):
    def __init__(self, task, batch_size, text_weight, data_path='data'):
        folder = Path(f'{data_path}/data/interpro/{task}')
        stage1 = Path(f'{data_path}/model/stage1/{task}')
        self.meta = json.loads((stage1 / 'meta.json').read_text())
        self.pids = read_lines(folder / f'{task}.pid')
        sequences = read_fasta(Path(data_path) / 'data/sequences.fasta')
        self.sequences = [sequences[pid] for pid in self.pids]
        self.domains = read_domains(folder / 'interpro.feature')
        self.text_valid = torch.zeros(len(self.pids), dtype=torch.bool)
        self.text = torch.empty(len(self.pids), 0)
        if text_weight:
            saved = load(stage1 / 'text.pt')
            self.meta['text_dim'] = saved['state']['fc1.weight'].shape[1]
            self.text = torch.zeros(len(self.pids), self.meta['text_dim'])
            embeddings = Path(data_path) / 'data/embeddings/pubmedbert'
            for i, pid in enumerate(self.pids):
                if (embeddings / f'{pid}.pt').is_file():
                    self.text[i] = read_embedding(embeddings, pid)
                    self.text_valid[i] = True
        batch = partial(alignment_batch, meta=self.meta)
        self.train_loader = DataLoader(self, batch_size=batch_size, shuffle=True, collate_fn=batch)
        self.export_loader = DataLoader(self, batch_size=batch_size, collate_fn=batch)

    def __len__(self):
        return len(self.pids)

    def __getitem__(self, i):
        return i, self.domains[self.pids[i]], self.text_valid[i], self.text[i]
