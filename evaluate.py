"""Evaluate prediction files using weighted Fmax."""
import argparse
import json
from collections import defaultdict
from pathlib import Path
from utils.data import read_lines, read_annotations
from utils.ontology import GeneOntology
from utils.evaluation import fmax


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-path', '--data_path', default='data', help='Data root directory')
    parser.add_argument('--task', choices=['mf', 'bp', 'cc'], default='mf', help='GO aspect')
    parser.add_argument('--split', choices=['cluster', 'temporal'], default='temporal', help='Benchmark test set')
    parser.add_argument('--result', required=True, help='Prediction file: protein ID, GO term, score, optional aspect')
    parser.add_argument('--anno', help='Custom ground-truth annotation file')
    parser.add_argument('--pids', help='Custom evaluation protein ID file')
    args = parser.parse_args()

    # load data
    folder = Path(args.data_path) / 'anno'
    ontology = GeneOntology(folder / 'go-basic.obo')
    truth = read_annotations(args.anno or folder / args.split / args.task / 'test.txt', ontology)
    truth = {pid: {go for go in terms if ontology[go].ns == args.task} for pid, terms in truth.items()}
    pids = read_lines(args.pids) if args.pids else (
        list(truth) if args.anno else read_lines(folder / args.split / args.task / 'test.pid'))
    ic = {go: float(value) for go, value in (line.split() for line in read_lines(folder / 'IA.tsv'))}
    scores = defaultdict(dict)
    for line in read_lines(args.result):
        pid, go, value, *_ = line.split()
        if ontology[go].ns == args.task:
            scores[pid][go] = max(scores[pid].get(go, 0), float(value))

    # evaluate
    score, threshold = fmax(scores, truth, pids, ic)
    metrics = dict(task=args.task, proteins=len(pids),
                   evaluated_proteins=sum(len(truth.get(pid, set())) > 1 for pid in pids),
                   proteins_with_predictions=sum(pid in scores for pid in pids),
                   wFmax=score, threshold=threshold)
    output = Path(args.result).with_suffix('.metrics.json')
    output.write_text(json.dumps(metrics, indent=2) + '\n')
    print(json.dumps(metrics, indent=2))


if __name__ == '__main__':
    main()
