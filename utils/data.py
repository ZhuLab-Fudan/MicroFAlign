"""Shared file readers."""
from collections import defaultdict
from pathlib import Path
import torch
from Bio import SeqIO

ROOTS = {'GO:0008150', 'GO:0003674', 'GO:0005575'}
KINGDOMS = ['3379134', '1783272', '2731360', '3366610', '33208', '33090', '4751', '0']


def read_lines(path):
    return Path(path).read_text().splitlines()


def load(path):
    return torch.load(path, map_location='cpu', weights_only=False)


def read_embedding(directory, pid):
    x = load(Path(directory) / f'{pid}.pt')
    if isinstance(x, dict):
        reps = x['mean_representations']
        x = reps[max(reps)]
    return x.float().reshape(-1)


def move(batch, device):
    return {k: v.to(device) for k, v in batch.items()}


def read_fasta(path):
    return {r.id.split('|')[1] if '|' in r.id else r.id: str(r.seq)
            for r in SeqIO.parse(path, 'fasta')}


def read_domains(path):
    domains = {}
    for line in read_lines(path):
        pid, *tokens = line.split()
        domains[pid] = sorted(int(t.split(':')[0]) for t in tokens)
    return domains


def read_annotations(path, ontology):
    annotations = defaultdict(set)
    for line in read_lines(path):
        pid, term = line.split()[:2]
        annotations[pid].add(term)
    return {pid: ontology.transfer(terms) - ROOTS for pid, terms in annotations.items()}


def label_matrix(pids, truth, labels):
    index = {term: i for i, term in enumerate(labels)}
    y = torch.zeros(len(pids), len(labels))
    for i, pid in enumerate(pids):
        for term in truth[pid]:
            if term in index:
                y[i, index[term]] = 1
    return y


def model_folder(data_path, version='trained'):
    folder = Path(data_path) / 'model'
    return folder if version == 'trained' else folder / version
