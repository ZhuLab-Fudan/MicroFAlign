"""Export pooled ESM2 and aligned sequence embeddings."""
import argparse
import torch
from Bio import SeqIO
from utils.sequence_embeddings import export_sequences


def read_sequences(args):
    if args.sequence:
        yield args.pid, ''.join(args.sequence.split()).upper()
    else:
        for record in SeqIO.parse(args.fasta, 'fasta'):
            pid = record.id.split('|')[1] if '|' in record.id else record.id
            yield pid, str(record.seq).upper()


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-path', '--data_path', default='data', help='Root directory containing anno, data, model and res')
    parser.add_argument('--task', choices=['mf', 'bp', 'cc'], default='mf', help='GO aspect')
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--fasta', help='Input FASTA file')
    source.add_argument('--sequence', help='Input amino-acid sequence')
    parser.add_argument('--pid', default='protein1', help='Protein ID for a single input sequence')
    parser.add_argument('--device', default='cuda:0', help='Compute device, e.g. cuda:0 or cpu')
    parser.add_argument('--batch-size', type=int, default=24, help='Proteins per batch')
    parser.add_argument('--version', choices=['domain', 'domain+text', 'trained'], default='text', help='Pretrained variant or locally trained models')
    args = parser.parse_args()
    torch.set_num_threads(4)

    export_sequences(read_sequences(args), args)


if __name__ == '__main__':
    main()
