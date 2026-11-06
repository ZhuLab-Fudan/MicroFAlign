"""Data loading and GO prediction."""
from pathlib import Path
import torch
from torch.utils.data import Dataset
from utils.data import model_folder, KINGDOMS, read_lines, read_embedding, read_annotations, read_fasta, label_matrix
from utils.ontology import GeneOntology
from utils.evaluation import fmax


class PredictionDataset(Dataset):
    def __init__(self, task, split, labels=None, fasta=None, data_path='data', kingdom_file=None, version='trained', benchmark='temporal'):
        folder = Path(f'{data_path}/anno/cluster/{task}')
        self.ontology = GeneOntology(f'{data_path}/anno/go-basic.obo')
        self.ic_path = Path(data_path) / 'anno/IA.tsv'
        self.labels = read_lines(folder / 'labels.txt') if labels is None else labels
        self.sequence_dir = model_folder(data_path, version) / f'stage2/{task}/embeddings/sequence'
        self.glm_dir = Path(data_path) / 'data/embeddings/glm2'
        kingdom_path = Path(kingdom_file) if kingdom_file else Path(data_path) / 'data/kingdom.tsv'
        kingdoms = dict(line.split() for line in read_lines(kingdom_path))
        if split == 'predict':
            self.pids = list(read_fasta(fasta)) if fasta else read_lines(f'{data_path}/anno/{benchmark}/{task}/test.pid')
            self.truth, self.y = {}, None
        else:
            self.truth = read_annotations(folder / f'{split}.txt', self.ontology)
            self.pids = [p for p in read_lines(folder / f'{split}.pid') if p in self.truth]
            self.y = label_matrix(self.pids, self.truth, self.labels)
        mapping = {k: i for i, k in enumerate(KINGDOMS)}
        self.kingdom = torch.zeros(len(self.pids), 9)
        for i, pid in enumerate(self.pids):
            self.kingdom[i, mapping.get(kingdoms[pid], 8)] = 1

    def __len__(self):
        return len(self.pids)

    def __getitem__(self, i):
        pid = self.pids[i]
        batch = dict(index=i, sequence=read_embedding(self.sequence_dir, pid),
                     glm=read_embedding(self.glm_dir, pid), kingdom=self.kingdom[i])
        if self.y is not None:
            batch['y'] = self.y[i]
        return batch


@torch.no_grad()
def predict_scores(model, loader, args, propagate=True):
    model.eval()
    scores = {}
    for batch in loader:
        probabilities = model(batch['sequence'].to(args.device), batch['glm'].to(args.device),
                              batch['kingdom'].to(args.device)).cpu()
        for index, row in zip(batch['index'].tolist(), probabilities):
            raw = {go: float(value) for go, value in zip(loader.dataset.labels, row) if value >= args.prob}
            if propagate:
                raw = loader.dataset.ontology.transfer_scores(raw)
            scores[loader.dataset.pids[index]] = dict(sorted(raw.items(), key=lambda x: x[1], reverse=True)[:args.top_k])
    return scores


def score_predictions(scores, dataset):
    ic = {go: float(value) for go, value in (line.split() for line in read_lines(dataset.ic_path))}
    return fmax(scores, dataset.truth, dataset.pids, ic)


def save_predictions(scores, ontology, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w') as handle:
        for pid, terms in scores.items():
            for go, value in terms.items():
                handle.write(f'{pid}\t{go}\t{value:.3f}\t{ontology[go].ns}\n')
