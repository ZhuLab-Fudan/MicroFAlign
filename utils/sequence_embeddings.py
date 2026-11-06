"""Export sequence representations from a selected alignment model."""
from contextlib import nullcontext
from itertools import islice
import torch
from models.alignment import load_sequence_model
from utils.data import model_folder


@torch.inference_mode()
def export_sequences(records, args):
    # load model
    esm, projection, max_length = load_sequence_model(args.task, args.device, args.data_path, args.version)

    # load data
    records = iter(records)
    folder = model_folder(args.data_path, args.version) / f'stage2/{args.task}/embeddings'
    for name in ('esm2', 'sequence'):
        (folder / name).mkdir(parents=True, exist_ok=True)

    # extract embeddings
    count = 0
    while rows := list(islice(records, args.batch_size)):
        pids, sequences = zip(*rows)
        tokens = esm.tokenize(list(sequences), max_length, args.device)
        precision = torch.autocast('cuda', dtype=torch.bfloat16) if args.device.startswith('cuda') else nullcontext()
        with precision:
            pooled = esm(tokens)
            projected = projection(pooled)

        # save embeddings
        vectors = {'esm2': pooled.float().cpu(), 'sequence': projected.float().cpu()}
        for i, pid in enumerate(pids):
            for name, values in vectors.items():
                torch.save(values[i].clone(), folder / name / f'{pid}.pt')
        count += len(pids)
        print(f'Exported {count} proteins', flush=True)

