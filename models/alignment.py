"""Sequence projection and frozen auxiliary encoders."""
from utils.data import model_folder
from pathlib import Path
import torch
from torch import nn
from models.features import FeatureBackbone
from models.esm2 import ESM2Encoder, load_finetuned
from utils.data import load


class AlignmentModel(nn.Module):
    def __init__(self, inputs, meta):
        super().__init__()
        hidden = meta['hidden']
        self.sequence = nn.Linear(inputs, hidden)
        self.domain = FeatureBackbone(len(meta['domains']), hidden)
        self.domain.requires_grad_(False)
        if 'text_dim' in meta:
            self.text = FeatureBackbone(meta['text_dim'], hidden)
            self.text.requires_grad_(False)


def load_models(args, meta):
    esm = ESM2Encoder(f'{args.data_path}/model/pretrained/esm2')
    esm.configure(args.last_layers, args.rank, args.alpha, args.checkpoint)
    model = AlignmentModel(esm.dim, meta)
    names = ['domain', 'text'] if args.text_weight else ['domain']
    for name in names:
        saved = load(f'{args.data_path}/model/stage1/{args.task}/{name}.pt')
        state = {k: v for k, v in saved['state'].items() if not k.startswith('fc3.')}
        getattr(model, name).load_state_dict(state)
    return esm.to(args.device), model.to(args.device)


def save_alignment(esm, model, meta, args):
    folder = Path(f'{args.data_path}/model/stage2/{args.task}')
    folder.mkdir(parents=True, exist_ok=True)
    config = dict(vars(args), esm_model=f'{args.data_path}/model/pretrained/esm2', tune='lora')
    meta = dict(meta, feature_dim=meta['hidden'], output_dim=meta['hidden'],
                input_dim=esm.dim, projection='linear', esm_model=config['esm_model'],
                max_length=args.max_length)
    torch.save(dict(state=esm.trainable_state(), args=config), folder / 'esm2_tuned.pt')
    state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    torch.save(dict(state=state, meta=meta, args=config, epochs=args.epochs), folder / 'align.pt')


def load_sequence_model(task, device, data_path='data', version='trained'):
    esm, max_length = load_finetuned(task, device, data_path, version)
    saved = load(model_folder(data_path, version) / f'stage2/{task}/align.pt')
    state = {k.removeprefix('sequence.'): v for k, v in saved['state'].items()
             if k.startswith('sequence.')}
    outputs, inputs = state['weight'].shape
    projection = nn.Linear(inputs, outputs)
    projection.load_state_dict(state)
    return esm, projection.to(device).eval(), max_length
