"""ESM2 with LoRA and masked language modeling."""
from utils.data import model_folder
import math
import torch
from torch import nn
from transformers import AutoTokenizer, EsmForMaskedLM
from torch.nn import functional as F


class LoRALinear(nn.Module):
    def __init__(self, base, rank=8, alpha=16):
        super().__init__()
        self.base = base
        self.a = nn.Linear(base.in_features, rank, bias=False)
        self.b = nn.Linear(rank, base.out_features, bias=False)
        nn.init.kaiming_uniform_(self.a.weight, a=math.sqrt(5))
        nn.init.zeros_(self.b.weight)
        self.scale = alpha / rank

    def forward(self, x):
        return self.base(x) + self.b(self.a(x)) * self.scale


class ESM2Encoder(nn.Module):
    def __init__(self, path):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
        pretrained = EsmForMaskedLM.from_pretrained(path, local_files_only=True)
        self.esm = pretrained.esm
        self.lm_head = pretrained.lm_head
        self.requires_grad_(False)
        self.dim = self.esm.config.hidden_size

    def configure(self, last_layers=3, rank=8, alpha=16, checkpoint=False):
        for layer in self.esm.encoder.layer[-last_layers:]:
            attention = layer.attention.self
            for name in ('query', 'value'):
                setattr(attention, name, LoRALinear(getattr(attention, name), rank, alpha))
        if checkpoint:
            self.esm.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={'use_reentrant': False})

    def tokenize(self, sequences, max_length, device):
        tokens = self.tokenizer(sequences, padding=True, truncation=True,
                                max_length=max_length + 2, return_tensors='pt')
        return {k: v.to(device) for k, v in tokens.items()}

    def forward(self, tokens, labels=None):
        h = self.esm(**tokens).last_hidden_state
        mask = tokens['attention_mask'].bool()
        mask = mask & (tokens['input_ids'] != self.tokenizer.cls_token_id)
        mask = mask & (tokens['input_ids'] != self.tokenizer.eos_token_id)
        pooled = (h.float() * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True)
        if labels is None:
            return pooled
        selected = labels != -100
        mlm = F.cross_entropy(self.lm_head(h[selected]).float(), labels[selected]) if selected.any() else h.sum() * 0
        return pooled, mlm

    def mask_tokens(self, tokens, probability=0.15):
        """Mask selected residues with the 80/10/10 sampling rule."""
        ids = tokens['input_ids']
        eligible = tokens['attention_mask'].bool()
        for token_id in self.tokenizer.all_special_ids:
            eligible = eligible & (ids != token_id)
        selected = (torch.rand(ids.shape, device=ids.device) < probability) & eligible
        labels = ids.clone().masked_fill(~selected, -100)
        masked = ids.clone()
        choice = torch.rand(ids.shape, device=ids.device)
        masked[selected & (choice < 0.8)] = self.tokenizer.mask_token_id
        residues = torch.tensor(self.tokenizer.convert_tokens_to_ids(list('LAGVSERTIDPKQNFYMHWCXBUZO')),
                                device=ids.device)
        random_ids = residues[torch.randint(len(residues), ids.shape, device=ids.device)]
        replace = selected & (choice >= 0.8) & (choice < 0.9)
        masked[replace] = random_ids[replace]
        return dict(tokens, input_ids=masked), labels

    def trainable_state(self):
        return {n: p.detach().cpu() for n, p in self.named_parameters() if p.requires_grad}


def load_finetuned(task, device, data_path='data', version='trained'):
    saved = torch.load(model_folder(data_path, version) / f'stage2/{task}/esm2_tuned.pt', map_location='cpu', weights_only=False)
    config = saved['args']
    model = ESM2Encoder(f'{data_path}/model/pretrained/esm2')
    model.configure(config['last_layers'], config['rank'], config['alpha'])
    model.load_state_dict(saved['state'], strict=False)
    return model.to(device).eval(), config['max_length']
