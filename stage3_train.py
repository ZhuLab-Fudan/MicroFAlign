"""Stage 3: train the sequence, gLM2 and kingdom GO predictor."""
import argparse
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from models.predictor import DNN, load_predictor, save_predictor
from utils.prediction import PredictionDataset, predict_scores, score_predictions, save_predictions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-path', '--data_path', default='data', help='Root directory containing anno, data, model and res')
    parser.add_argument('--task', choices=['mf', 'bp', 'cc'], default='mf', help='GO aspect')
    parser.add_argument('--device', default='cuda:0', help='Compute device, e.g. cuda:0 or cpu')
    parser.add_argument('--batch-size', type=int, default=128, help='Proteins per batch')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--patience', type=int, default=6, help='Epochs without validation improvement before stopping')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--prob', type=float, default=.01, help='Minimum GO prediction score')
    parser.add_argument('--top-k', type=int, default=100, help='Maximum GO terms per protein')
    args = parser.parse_args()
    torch.set_num_threads(4)

    # load data
    train_data = PredictionDataset(args.task, 'train', data_path=args.data_path)
    val_data = PredictionDataset(args.task, 'val', train_data.labels, data_path=args.data_path)
    test_data = PredictionDataset(args.task, 'test', train_data.labels, data_path=args.data_path)
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True,
                              drop_last=(len(train_data) % args.batch_size == 1))
    val_loader = DataLoader(val_data, batch_size=args.batch_size)
    test_loader = DataLoader(test_data, batch_size=args.batch_size)

    # define model and optimizer
    model = DNN(len(train_data.labels)).to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    best, stale = -1., 0

    for epoch in range(args.epochs):
        # train
        model.train()
        total = 0.
        for batch in train_loader:
            probabilities = model(batch['sequence'].to(args.device), batch['glm'].to(args.device),
                                  batch['kingdom'].to(args.device))
            loss = F.binary_cross_entropy(probabilities, batch['y'].to(args.device))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += loss.item()

        # validate
        scores = predict_scores(model, val_loader, args, propagate=False)
        score, threshold = score_predictions(scores, val_data)
        print(f'epoch={epoch+1} BCE={total/len(train_loader):.5f} val_wFmax={score:.5f}', flush=True)

        # save model
        if score > best:
            best, stale = score, 0
            save_predictor(model, train_data.labels, args, score, threshold)
        else:
            stale += 1
        if stale >= args.patience:
            break

    # test
    model, _ = load_predictor(args.task, args.device, args.data_path)
    scores = predict_scores(model, test_loader, args)
    print('Test wFmax, threshold:', score_predictions(scores, test_data), flush=True)
    save_predictions(scores, test_data.ontology, f'{args.data_path}/res/cluster/{args.task}.txt')


if __name__ == '__main__':
    main()
