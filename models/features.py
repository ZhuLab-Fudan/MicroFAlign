"""GO-supervised auxiliary encoders shared by Stages 1 and 2."""
import json
from pathlib import Path
import torch
from torch import nn


class FeatureBackbone(nn.Module):
    def __init__(self, inputs, hidden=1024):
        super().__init__()
        self.fc1 = nn.Linear(inputs, hidden, bias=False)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc1_norm = nn.LayerNorm(hidden)
        self.fc2_norm = nn.LayerNorm(hidden)

    def forward(self, x):
        x = self.fc1_norm(self.fc1(x)).relu()
        return self.fc2_norm(self.fc2(x)).relu()


class FeatureClassifier(FeatureBackbone):
    def __init__(self, inputs, hidden, labels):
        super().__init__(inputs, hidden)
        self.fc3 = nn.Linear(hidden, labels)

    def encode(self, x):
        return super().forward(x)

    def forward(self, x):
        return self.fc3(self.encode(x))


def save_features(model, meta, args, score, threshold):
    folder = Path(f'{args.data_path}/model/stage1/{args.task}')
    folder.mkdir(parents=True, exist_ok=True)
    name = 'domain' if args.feature == 'interpro' else 'text'
    state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    torch.save(dict(state=state, meta=meta, score=score, threshold=threshold,
                    args=vars(args)), folder / f'{name}.pt')
    (folder / 'meta.json').write_text(json.dumps(meta, indent=2))
