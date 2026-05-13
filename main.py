import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from sklearn.metrics import accuracy_score
from torch_geometric.datasets import TUDataset
from sklearn.model_selection import StratifiedKFold
import argparse
import numpy as np
import torch
import time
from model import *

def get_argparse():
    parser = argparse.ArgumentParser(description='QS-GCN: Quadratic Spectral Graph Convolution Network for Graph Classification.')
    parser.add_argument('--bmname', dest='bmname',
                        help='Name of the benchmark dataset')
    parser.add_argument('--num-conv', dest='num_conv', type=int,
                        help='number of conv layers')
    parser.add_argument('--output-dim', dest='output_dim',
                        help='output_dim')
    parser.add_argument('--cuda', dest='cuda',
                        help='CUDA.')
    parser.add_argument('--lr', dest='lr', type=float,
                        help='Learning rate.')
    parser.add_argument('--weight_decay', dest='weight_decay', type=float,
                        help='weight_decay.')
    parser.add_argument('--clip', dest='clip', type=float,
                        help='Gradient clipping.')
    parser.add_argument('--batch-size', dest='batch_size', type=int,
                        help='Batch size.')
    parser.add_argument('--epochs', dest='num_epochs', type=int,
                        help='Number of epochs to train.')
    parser.add_argument('--feature', dest='feature_type',
                        help='Feature used for encoder. Can be: id, deg')
    parser.add_argument('--bn', dest='bn', action='store_const',
                        const=False, default=True,
                        help='Whether batch normalization is used')
    parser.add_argument('--dropout', dest='dropout', type=float,
                        help='Dropout rate.')
    parser.add_argument('--nobias', dest='bias', action='store_const',
                        const=False, default=True,
                        help='Whether to add bias. Default to True.')
    parser.add_argument('--rand_seed', dest='rand_seed', type=int, default="0",
                        help='random seed (default: 0)')
    parser.set_defaults(datadir='data',
                        bmname="MUTAG",
                        cuda=0,
                        feature_type='default',
                        lr=0.001,
                        weight_decay=0.0001,
                        clip=2.0,
                        bn=True,
                        dropout=0.1,
                        batch_size=64,
                        num_epochs=200,
                        num_conv=2,
                        output_dim=64,
                        rand_seed=0
                        )

    return parser.parse_args()

def train(model, loader, optimizer, device):
    model.train()
    total_loss = 0
    total_time = 0
    for data in loader:
        begin_time = time.time()
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data.x, data.edge_index, data.batch)
        loss = F.cross_entropy(out, data.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        elapsed = time.time() - begin_time
        total_time += elapsed
    return total_loss / len(loader), total_time

def test(model, loader, device):
    model.eval()
    correct = 0
    for data in loader:
        data = data.to(device)
        out = model(data.x, data.edge_index, data.batch)
        pred = out.argmax(dim=1)
        correct += int((pred == data.y).sum())
    return correct / len(loader.dataset)

def main():
    args = get_argparse()
    print(args)
    device = torch.device('cuda:' + str(args.cuda) if torch.cuda.is_available() else 'cpu')
    print(device)
    rand_seed = args.rand_seed
    torch.manual_seed(rand_seed)
    np.random.seed(rand_seed)
    dataset = TUDataset(root='/tmp/TUDataset', name=args.bmname)
    in_channels = dataset.num_node_features
    out_channels = dataset.num_classes
    max_deg = 20
    new_dataset = []
    for data in dataset:
        if data.x is None:
            in_channels = max_deg+1
            degs = torch.bincount(data.edge_index[0], minlength=data.num_nodes)
            degs[degs > max_deg] = max_deg
            x = torch.zeros((data.num_nodes, max_deg + 1))
            x[torch.arange(data.num_nodes), degs] = 1
            data.x = x
        new_dataset.append(data)

    dataset = new_dataset

    labels = [data.y.item() for data in dataset]
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=args.rand_seed)
    fold, (train_idx, test_idx) = next(iter(enumerate(skf.split(dataset, labels))))
    train_dataset = [dataset[i] for i in train_idx]
    test_dataset = [dataset[i] for i in test_idx]
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size)

    model = SpectralNet(input_dim=in_channels, num_conv=args.num_conv, hidden_dim=args.output_dim,
                        num_classes=out_channels).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    for epoch in range(args.num_epochs):
        loss, epoch_time = train(model, train_loader, optimizer, device)
        if epoch % 10 == 0:
            model.eval()
            acc = test(model, train_loader, device)
            test_acc = test(model, test_loader, device)
            print('Epoch: ', epoch, ' Avg loss: ', loss, '; train acc: ', acc, '; test acc: ', test_acc,
                  '; epoch time: ', epoch_time)
    model.eval()
    test_acc = test(model, test_loader, device)
    print('Test acc: ', test_acc)

if __name__ == "__main__":
    main()
