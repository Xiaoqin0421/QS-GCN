import torch
import torch.nn as nn
from torch_geometric.nn import global_mean_pool
from torch_geometric.datasets import TUDataset
from torch_geometric.utils import to_scipy_sparse_matrix
import torch.nn.functional as F
from torch_sparse import SparseTensor
from torch_geometric.utils import get_laplacian, add_self_loops, remove_self_loops
from torch_scatter import scatter_add
from torch_geometric.nn import GCNConv

class SpectralConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.lin = nn.Linear(in_channels, out_channels)
        self.a = nn.Parameter(torch.tensor(-1.0))
        self.b = nn.Parameter(torch.tensor(2.0))  
        self.c = nn.Parameter(torch.tensor(0.0))


    def forward(self, x, edge_index):
        N = x.size(0)
        edge_index, _ = remove_self_loops(edge_index)
        edge_index, _ = add_self_loops(edge_index, num_nodes=N)
        lap_ei, lap_w = get_laplacian(edge_index, normalization='sym', num_nodes=N)
        L = SparseTensor(row=lap_ei[0], col=lap_ei[1], value=lap_w, sparse_sizes=(N, N))
        I = SparseTensor.eye(N, dtype=L.dtype(),device=L.device())
        Lx = L.matmul(x)
        L2x = L.matmul(Lx)
        Ix = I.matmul(x)
        out = self.a * L2x + self.b * Lx + self.c * Ix

        return self.lin(out)


class DiffPoolLike(nn.Module):
    def __init__(self, in_dim, embed_dim, num_clusters):
        super().__init__()
        self.embed_gcn = GCNConv(in_dim, embed_dim)
        self.assign_gcn = GCNConv(embed_dim, num_clusters)

    def forward(self, x_in, edge_index, batch):

        N = x_in.size(0)
        B = int(batch.max().item()) + 1
        K = self.assign_gcn.out_channels
        d = self.embed_gcn.out_channels

        x = self.embed_gcn(x_in, edge_index)

        S_logits = self.assign_gcn(x, edge_index)

        S = torch.softmax(S_logits, dim=-1)

        cluster_ids = batch.unsqueeze(1) * K + torch.arange(K, device=x.device).unsqueeze(0)
        cluster_idx = cluster_ids.view(-1)

        x_exp = x.unsqueeze(1).expand(-1, K, -1).reshape(-1, d)
        S_flat = S.view(-1)

        pooled_flat = scatter_add(x_exp * S_flat.unsqueeze(1),
                                  cluster_idx,
                                  dim=0,
                                  dim_size=B * K)

        pooled = pooled_flat.view(B, K, d)
        return pooled

class SpectralNet(nn.Module):
    def __init__(self, input_dim, num_conv, hidden_dim, num_classes):
        super().__init__()
        self.num_conv = num_conv
        self.conv = nn.ModuleList()
        self.conv.append(SpectralConv(input_dim, hidden_dim))
        for i in range(1, num_conv):
            self.conv.append(SpectralConv(hidden_dim, hidden_dim))
        num_clusters = 8
        self.pool = DiffPoolLike(hidden_dim, hidden_dim, num_clusters)
        self.lin1 = nn.Linear(hidden_dim*num_clusters, hidden_dim)
        self.lin2 = nn.Linear(hidden_dim, num_classes)


    def forward(self, x, edge_index, batch):
        for i in range(self.num_conv):
            x = self.conv[i](x, edge_index)
            x = F.relu(x)
        x = self.pool(x, edge_index, batch)
        x = x.view(x.size(0), -1)
        x = self.lin1(x)
        x = self.lin2(x)
        return x
