"""Stage 1: function-aware domain and text pretraining."""
import argparse
import torch
from torch.nn import functional as F
from models.features import FeatureClassifier, save_features
from utils.feature_data import FeatureData
from utils.evaluation import evaluate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-path', '--data_path', default='data', help='Root directory containing anno, data, model and res')
    parser.add_argument('--task', choices=['mf', 'bp', 'cc'], default='mf', help='GO aspect')
    parser.add_argument('--feature', choices=['interpro', 'text'], required=True, help='Auxiliary feature to train')
    parser.add_argument('--device', default='cuda:0', help='Compute device, e.g. cuda:0 or cpu')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--patience', type=int, default=5, help='Epochs without validation improvement before stopping')
    parser.add_argument('--batch-size', type=int, default=128, help='Proteins per batch')
    parser.add_argument('--hidden', type=int, default=1024, help='Auxiliary hidden dimension')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    args = parser.parse_args()
    torch.set_num_threads(4)

    # load data
    data = FeatureData(args.task, args.feature, args.batch_size, args.hidden, args.data_path)

    # define model and optimizer
    model = FeatureClassifier(data.input_dim, args.hidden, data.n_labels).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    best, stale = -1., 0

    for epoch in range(args.epochs):
        # train
        model.train()
        total = 0.
        for batch in data.train_loader:
            logits = model(batch['input'].to(args.device))
            loss = F.binary_cross_entropy_with_logits(logits, batch['y'].to(args.device))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += loss.item()

        # validate
        score, threshold = evaluate(model, data.val_loader, args.device, data.meta['labels'])
        print(f'epoch={epoch+1} BCE={total/len(data.train_loader):.5f} val_wFmax={score:.5f}', flush=True)

        # save model
        if score > best:
            best, stale = score, 0
            save_features(model, data.meta, args, score, threshold)
        else:
            stale += 1
        if stale >= args.patience:
            break


if __name__ == '__main__':
    main()
