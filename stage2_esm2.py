"""Stage 2: function-aware ESM2 adaptation with CLIP and MLM."""
import argparse
import time
from pathlib import Path
from contextlib import nullcontext
import torch
from models.alignment import load_models, save_alignment
from utils.alignment_data import AlignmentData
from utils.contrastive import multi_positive
from utils.data import move


def precision(args):
    return torch.autocast('cuda', dtype=torch.bfloat16) if args.device.startswith('cuda') else nullcontext()


def contrastive_loss(model, batch, args, z):
    d = model.domain(batch['domain_input'])
    valid = (batch['domain'] >= 0).any(1)
    groups = [tuple(row[row >= 0].tolist()) for row in batch['domain'][valid]]
    loss = multi_positive(z[valid].float(), d[valid].float(), groups, args.temperature)
    if args.text_weight:
        valid_text = batch['text_valid']
        raw = batch['text_input'][valid_text]
        t = model.text(raw)
        _, groups_text = torch.unique(raw, dim=0, return_inverse=True)
        loss = loss + args.text_weight * multi_positive(
            z[valid_text].float(), t.float(), groups_text.tolist(), args.temperature)
    return loss


@torch.no_grad()
def extract(esm, model, data, loader, args):
    """Export pooled and projected sequence embeddings."""
    esm.eval()
    model.eval()
    folder = Path(f'{args.data_path}/model/stage2/{args.task}/embeddings')
    for name in ('esm2', 'sequence'):
        (folder / name).mkdir(parents=True, exist_ok=True)
    for batch in loader:
        ids = batch['index'].tolist()
        tokens = esm.tokenize([data.sequences[i] for i in ids], args.max_length, args.device)
        with precision(args):
            pooled = esm(tokens)
            projected = model.sequence(pooled)
        vectors = {'esm2': pooled.float().cpu(), 'sequence': projected.float().cpu()}
        for row, i in enumerate(ids):
            for name, values in vectors.items():
                torch.save(values[row].clone(), folder / name / f'{data.pids[i]}.pt')
    (folder / f'{args.task}.pid').write_text('\n'.join(data.pids) + '\n')


def train_epoch(esm, model, data, optimizer, args):
    esm.train()
    model.train()
    optimizer.zero_grad(set_to_none=True)
    totals = torch.zeros(3, device=args.device)
    for step, batch in enumerate(data.train_loader):
        ids = batch['index'].tolist()
        tokens = esm.tokenize([data.sequences[i] for i in ids], args.max_length, args.device)
        batch = move(batch, args.device)
        labels = None
        if args.mlm_weight:
            tokens, labels = esm.mask_tokens(tokens, args.mlm_prob)
        with precision(args):
            if labels is None:
                x, mlm = esm(tokens), torch.zeros((), device=args.device)
            else:
                x, mlm = esm(tokens, labels)
            clip = contrastive_loss(model, batch, args, model.sequence(x))
            loss = args.clip_weight * clip + args.mlm_weight * mlm
        start = (step // args.accum_steps) * args.accum_steps * args.batch_size
        group_size = min(args.accum_steps * args.batch_size, len(data) - start)
        (loss * len(ids) / group_size).backward()
        if (step + 1) % args.accum_steps == 0 or step + 1 == len(data.train_loader):
            torch.nn.utils.clip_grad_norm_([p for g in optimizer.param_groups for p in g['params']], 1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        totals += torch.stack([loss.detach(), clip.detach(), mlm.detach()]) * len(ids)
    return (totals / len(data)).tolist()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-path', '--data_path', default='data', help='Root directory containing anno, data, model and res')
    parser.add_argument('--task', choices=['mf', 'bp', 'cc'], default='mf', help='GO aspect')
    parser.add_argument('--device', default='cuda:0', help='Compute device, e.g. cuda:0 or cpu')
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=24, help='Proteins per batch')
    parser.add_argument('--accum-steps', type=int, default=2, help='Gradient accumulation steps')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--esm-lr', type=float, default=1e-4, help='LoRA learning rate')
    parser.add_argument('--last-layers', type=int, default=3, help='Number of final ESM2 layers with LoRA')
    parser.add_argument('--rank', type=int, default=8, help='LoRA rank')
    parser.add_argument('--alpha', type=int, default=16, help='LoRA scaling factor')
    parser.add_argument('--max-length', type=int, default=1000, help='Maximum number of amino acids')
    parser.add_argument('--checkpoint', action='store_true', help='Enable gradient checkpointing')
    parser.add_argument('--temperature', type=float, default=0.1, help='Contrastive temperature')
    parser.add_argument('--clip-weight', type=float, default=0.1, help='Contrastive loss weight')
    parser.add_argument('--mlm-weight', type=float, default=1.0, help='MLM loss weight')
    parser.add_argument('--mlm-prob', type=float, default=0.15, help='Probability of selecting a residue for MLM')
    parser.add_argument('--text-weight', type=float, default=1.0, help='Text alignment weight; 0 disables text alignment')
    args = parser.parse_args()
    torch.set_num_threads(4)

    # load data
    data = AlignmentData(args.task, args.batch_size, args.text_weight, args.data_path)

    # load models and define optimizer
    esm, model = load_models(args, data.meta)
    optimizer = torch.optim.AdamW([
        {'params': [p for p in esm.parameters() if p.requires_grad], 'lr': args.esm_lr},
        {'params': model.sequence.parameters(), 'lr': args.lr},
    ])

    # train
    for epoch in range(args.epochs):
        started = time.perf_counter()
        scores = train_epoch(esm, model, data, optimizer, args)
        print(f'epoch={epoch+1} total/clip/mlm={scores} seconds={time.perf_counter()-started:.1f}', flush=True)

    # save model and embeddings
    save_alignment(esm, model, data.meta, args)
    extract(esm, model, data, data.export_loader, args)


if __name__ == '__main__':
    main()
