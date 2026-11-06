"""Predict GO terms using trained MicroFAlign models."""
import argparse
from pathlib import Path
from types import SimpleNamespace
from utils.data import read_fasta
from utils.sequence_embeddings import export_sequences
import torch
from torch.utils.data import DataLoader
from models.predictor import load_predictor
from utils.prediction import PredictionDataset, predict_scores, save_predictions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-path', '--data_path', default='data', help='Root directory containing anno, data, model and res')
    parser.add_argument('--task', choices=['mf', 'bp', 'cc'], default='mf', help='GO aspect')
    parser.add_argument('--fasta', help='Query FASTA; defaults to the temporal test set')
    parser.add_argument('--device', default='cuda:0', help='Compute device, e.g. cuda:0 or cpu')
    parser.add_argument('--batch-size', type=int, default=128, help='Proteins per batch')
    parser.add_argument('--prob', type=float, default=0.01, help='Minimum GO prediction score')
    parser.add_argument('--top-k', type=int, default=100, help='Maximum GO terms per protein')
    parser.add_argument('--kingdom-file', '--kingdom_file', help='Protein-to-kingdom TSV; default: <data-path>/data/kingdom.tsv')
    parser.add_argument('--version', choices=['domain', 'domain+text', 'trained'], default='domain+text', help='Pretrained variant (domain;domain+text) or locally trained models')
    parser.add_argument('--split', choices=['cluster', 'temporal'], default='temporal', help='Benchmark test set')
    parser.add_argument('--out', help='Output prediction file')
    parser.add_argument('--esm-batch-size', type=int, default=4, help='Batch size for missing sequence embeddings')
    args = parser.parse_args()
    torch.set_num_threads(4)

    # load model
    model, labels = load_predictor(args.task, args.device, args.data_path, args.version)

    # load data
    data = PredictionDataset(args.task, 'predict', labels, args.fasta,
                             data_path=args.data_path, kingdom_file=args.kingdom_file,
                             version=args.version, benchmark=args.split)
    loader = DataLoader(data, batch_size=args.batch_size)

    # extract missing sequence embeddings
    missing = [pid for pid in data.pids if not (Path(data.sequence_dir) / f'{pid}.pt').exists()]
    if missing:
        sequences = read_fasta(args.fasta or f'{args.data_path}/res/{args.split}/all.fasta')
        export_args = SimpleNamespace(**vars(args))
        export_args.batch_size = args.esm_batch_size
        export_sequences(((pid, sequences[pid]) for pid in missing), export_args)

    # predict
    scores = predict_scores(model, loader, args)

    # save results
    output = args.out or f'{args.data_path}/res/{args.split}/{args.version}/{args.task}.txt'
    save_predictions(scores, data.ontology, output)
    print(f'Saved {len(scores)} proteins to {output}')


if __name__ == '__main__':
    main()
