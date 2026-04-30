import torch
import torch.nn as nn
import torch.nn.functional as F

class FTPComplex(nn.Module):
    def __init__(self, num_nodes, embedding_dim=64, num_layers=2):
        super(FTPComplex, self).__init__()
        self.num_nodes = num_nodes
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers

        # Embedding cho tất cả node
        self.embedding = nn.Embedding(num_nodes, embedding_dim)
        nn.init.xavier_uniform_(self.embedding.weight)

    def forward(self, edge_index):
        all_emb = self.embedding.weight
        embs = [all_emb]

        for _ in range(self.num_layers):
            row, col = edge_index
            deg = torch.bincount(row, minlength=self.num_nodes).float()
            deg_inv_sqrt = deg.pow(-0.5)
            deg_inv_sqrt[deg_inv_sqrt == float("inf")] = 0

            norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
            agg = torch.zeros_like(all_emb)
            agg.index_add_(0, row, all_emb[col] * norm.unsqueeze(1))

            all_emb = agg
            embs.append(all_emb)

        final_emb = torch.stack(embs, dim=1).mean(dim=1)  # average
        return final_emb

    def get_embedding(self, edge_index):
        return self.forward(edge_index)
