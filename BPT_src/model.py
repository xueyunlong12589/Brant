import os
import sys

from scipy import signal

from workspace.pretrain_graph.config.gene_cfg import BasicCfg

workspace_dir = os.path.abspath(os.path.join('/home/zdz/54server/BrainPretrain' if os.getcwd().split('/')[2] == 'zdz' else
                                             '/home/yzz/BrainNet'))
sys.path.append(workspace_dir)
import torch
import torch.nn as nn
import numpy as np
# from torchvision import transforms
from pretrain.pre_model import _weights_init

from torch.autograd import Function


class GRL(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha

        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha

        return output, None


class MLP(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(MLP, self).__init__()

        self.mlp = nn.Sequential(
            nn.Linear(in_dim, in_dim//2),
            # nn.BatchNorm1d(in_dim//2),
            nn.ReLU(inplace=True),
            nn.Linear(in_dim//2, in_dim//4),
            # nn.BatchNorm1d(in_dim // 4),
            nn.ReLU(inplace=True),
            nn.Linear(in_dim//4, out_dim),
        )

        self.apply(_weights_init)

    def forward(self, z, alpha=1, reverse_gradient=False):
        if reverse_gradient:
            z = GRL.apply(z, alpha)
        logit = self.mlp(z)

        return logit


# def my_attention(basic_cfg, emb):
#     b, c, _ = emb.shape
#     q = emb
#     k = emb
#     v = emb
#     alpha = torch.matmul(q, k.transpose(1, 2))
#     # alpha = torch.clamp(alpha - torch.quantile(alpha, q=0.9,  dim=-1, keepdim=True), min=1)
#     # alpha[alpha == 1] *= -1e9
#     # print(torch.max(alpha))
#     # print(torch.min(alpha))
#     identity = torch.unsqueeze(torch.eye(c, c), dim=0).repeat(b, 1, 1).to(alpha.device)
#     alpha[identity == 1] = -1e9
#     alpha = torch.nn.functional.softmax(alpha, dim=-1)
#     other = torch.matmul(alpha, v)
#     return torch.cat([emb, other], dim=-1), alpha
#
#     # batch_size, node_num, emb_dim = emb.shape
#     #
#     # inner_prod = torch.einsum('bnf,bmf->bnm', (emb, emb))
#     # t = torch.einsum('bnm->bn', torch.exp(emb))   # 先把i=j也加上了
#     # denominator = torch.Tensor(batch_size, node_num, node_num).to(basic_cfg.d['device'])
#     # for b in range(batch_size):
#     #     for i in range(node_num):
#     #         for j in range(node_num):
#     #             denominator[b][i][j] = t[b][i] - torch.exp(inner_prod)[b][i][j]
#     # a = torch.exp(inner_prod) / denominator
#     # emb_ = torch.einsum('bnm->bn', torch.mul(a, emb))
#     # return emb_
#
#
# from torch_geometric.data import Data
#
# def get_graph(basic_cfg, nodevec, seg_x, epsilon=1e-8, method='cos', del_loop=False):
#     batch_size, node_num, emb_dim = nodevec.shape
#
#     if method == 'cos':
#         # cos similarity to construct graph
#         inner_prod = torch.einsum('bnf,bmf->bnm', (nodevec, nodevec))
#         norm = torch.einsum('bnf,bnf->bn', (nodevec, nodevec))
#         norm = torch.sqrt(norm)
#         denominator = torch.einsum('bn,bm->bnm', (norm, norm))
#         denominator += epsilon
#         adj = inner_prod / denominator
#         # threshold
#         adj[adj<0.5] = 0
#     elif method == 'corr':
#         seg_x = seg_x.reshape(batch_size, node_num, -1)
#         adj = torch.Tensor(batch_size, node_num, node_num)
#         for b in range(batch_size):
#             for i in range(node_num):
#                 for j in range(node_num):
#                     corr = signal.correlate(seg_x[b, i].cpu(), seg_x[b, j].cpu(), mode='same')
#                     corr /= np.max(corr)
#                     adj[b,i,j] = torch.tensor(np.mean(corr))
#     if del_loop:
#         for b in range(batch_size):
#             for i in range(node_num):
#                 adj[b,i,i] = 0
#
#     if basic_cfg.d['gnn_name'] == 'GraphSAGE' or \
#        basic_cfg.d['gnn_name'] == 'GAT':
#         edge_index = [adj[i].nonzero().t().contiguous() for i in range(batch_size)]
#         data = [Data(x=nodevec[i], edge_index=edge_index[i]) for i in range(batch_size)]
#         return data, adj
#
#     elif basic_cfg.d['gnn_name'] == 'GCN':
#         edge_index = torch.tensor([[i, j] for i in range(node_num) for j in range(node_num)])
#         edge_index = [edge_index.t().contiguous() for i in range(batch_size)]
#         edge_weight = [adj[i].reshape(-1) for i in range(batch_size)]
#
#         data = [Data(x=nodevec[i], edge_index=edge_index[i], edge_weight=edge_weight[i]) for i in range(batch_size)]
#         return data, adj
#
#
#
#
#
# import torch.nn.functional as F
# import torch_geometric
# from torch_geometric.nn import GCNConv
# from torch_geometric.nn import SAGEConv
# from torch_geometric.nn import GATConv
#
# class GAT(torch.nn.Module):
#     def __init__(self, in_channels, hidden_channels, out_channels, heads):
#         super().__init__()
#         self.conv1 = GATConv(in_channels, out_channels, heads, concat=True, dropout=0.2)
#         # use `heads` output heads in `conv2`.
#         self.conv2 = GATConv(hidden_channels * heads, out_channels, heads=1, concat=False, dropout=0.2)
#
#     def forward(self, data):
#         x, edge_index = data.x, data.edge_index
#
#         # x = F.dropout(x, p=0.2, training=self.training)
#         # x = F.relu(x)
#         x = self.conv1(x=x, edge_index=edge_index)
#         x = F.dropout(x, p=0.2, training=self.training)
#         x = self.conv2(x=x, edge_index=edge_index)
#         return x
#
# class GraphSAGE(torch.nn.Module):
#     def __init__(self, in_dim, hidden_dim, out_dim, dropout=0.2):
#         super().__init__()
#         self.dropout = dropout
#         self.conv1 = SAGEConv(in_dim, hidden_dim)
#         # self.conv2 = SAGEConv(hidden_dim, hidden_dim)
#         # self.conv3 = SAGEConv(hidden_dim, out_dim)
#
#     def forward(self, data):
#         x, edge_index = data.x, data.edge_index
#
#         x = self.conv1(x=x, edge_index=edge_index)
#         x = F.relu(x)
#         x = F.dropout(x, p=self.dropout, training=self.training)
#
#         # x = self.conv2(x=x, edge_index=edge_index)
#         # x = F.relu(x)
#         # x = F.dropout(x, p=self.dropout, training=self.training)
#
#         # x = self.conv3(x=x, edge_index=edge_index)
#         # x = F.relu(x)
#         # x = F.dropout(x, p=self.dropout, training=self.training)
#         ### return torch.log_softmax(x, dim=-1)
#         return x
#
# class GCN(torch.nn.Module):
#     def __init__(self, emb_dim):
#         super().__init__()
#         self.conv1 = GCNConv(emb_dim, emb_dim, add_self_loops=False)
#         # self.conv2 = GCNConv(emb_dim, num_classes, add_self_loops=False)
#
#     def forward(self, data):
#         x, edge_index, edge_weight = data.x, data.edge_index, data.edge_weight
#
#         x = self.conv1(x=x, edge_index=edge_index, edge_weight=edge_weight)
#         x = F.relu(x)
#         x = F.dropout(x, p=0.2, training=self.training)
#         # x = self.conv2(x=x, edge_index=edge_index, edge_weight=edge_weight)
#         ### return F.log_softmax(x, dim=2)
#         return x
#
#
# class MagicEncoder(nn.Module):
#     def __init__(self, basic_cfg, cnn_emb_dim, lstm_emb_dim):
#         super(MagicEncoder, self).__init__()
#         self.cnn_ed  = cnn_emb_dim
#         self.lstm_emb_dim = lstm_emb_dim
#         self.device = basic_cfg.d['device']
#         # Multiscale CNN
#         self.cnn_s1 = nn.Conv1d(in_channels=1, out_channels=64, kernel_size=(125,), stride=(10,))
#         self.cnn_s2 = nn.Conv1d(in_channels=1, out_channels=64, kernel_size=(40,), stride=(5,))
#         self.cnn_s3 = nn.Conv1d(in_channels=1, out_channels=64, kernel_size=(5,), stride=(1,))
#         self.maxpool1_s1 = nn.MaxPool1d(kernel_size=(4,), stride=(2,))
#         self.maxpool1_s2 = nn.MaxPool1d(kernel_size=(4,), stride=(2,))
#         self.maxpool1_s3 = nn.MaxPool1d(kernel_size=(4,), stride=(2,))
#         self.dropout = nn.Dropout(0.5)
#         self.conv1_s1 = nn.Conv1d(in_channels=64,  out_channels=128,         kernel_size=(7,), stride=(1,))
#         self.conv1_s2 = nn.Conv1d(in_channels=64,  out_channels=128,         kernel_size=(7,), stride=(1,))
#         self.conv1_s3 = nn.Conv1d(in_channels=64,  out_channels=128,         kernel_size=(8,), stride=(1,))
#         self.conv2_s1 = nn.Conv1d(in_channels=128, out_channels=self.cnn_ed, kernel_size=(7,), stride=(1,))
#         self.conv2_s2 = nn.Conv1d(in_channels=128, out_channels=self.cnn_ed, kernel_size=(7,), stride=(1,))
#         self.conv2_s3 = nn.Conv1d(in_channels=128, out_channels=self.cnn_ed, kernel_size=(8,), stride=(1,))
#         self.maxpool2_s1 = nn.MaxPool1d(kernel_size=(2,), stride=(2,))
#         self.maxpool2_s2 = nn.MaxPool1d(kernel_size=(2,), stride=(2,))
#         self.maxpool2_s3 = nn.MaxPool1d(kernel_size=(4,), stride=(4,))
#         # Aggregation Layer
#         self.resi_conv1 = nn.Conv1d(in_channels=self.cnn_ed, out_channels=self.cnn_ed, kernel_size=(1,), stride=(1,))
#         self.resi_conv2 = nn.Conv1d(in_channels=self.cnn_ed, out_channels=self.cnn_ed, kernel_size=(1,), stride=(1,))
#         self.avgpool = nn.AvgPool1d(1,1)
#         self.fc0 = nn.Linear(in_features=122, out_features=32)
#         self.fc1 = nn.Linear(in_features=32, out_features=1)
#         self.relu = nn.ReLU()
#         self.sigmoid = nn.Sigmoid()
#         # BiLSTM-AM
#         # self.lstm = nn.LSTM(input_size=256, hidden_size=256, num_layers=3, dropout=0.3, bidirectional=True, batch_first=True)
#
#         encoder_layer = nn.TransformerEncoderLayer(d_model=self.cnn_ed, nhead=8, dim_feedforward=1024, batch_first=True)
#         self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=6)
#
#     def forward(self, x, mean=True):
#         # scale 1 CNN
#         x_s1 = self.cnn_s1(x)
#         x_s1 = self.maxpool1_s1(x_s1)
#         x_s1 = self.dropout(x_s1)
#         x_s1 = self.conv1_s1(x_s1)
#         x_s1 = self.conv2_s1(x_s1)
#         x_s1 = self.maxpool2_s1(x_s1)
#         # scale 2 CNN
#         x_s2 = self.cnn_s2(x)
#         x_s2 = self.maxpool1_s2(x_s2)
#         x_s2 = self.dropout(x_s2)
#         x_s2 = self.conv1_s2(x_s2)
#         x_s2 = self.conv2_s2(x_s2)
#         x_s2 = self.maxpool2_s1(x_s2)
#         # scale 3 CNN
#         x_s3 = self.cnn_s1(x)
#         x_s3 = self.maxpool1_s1(x_s3)
#         x_s3 = self.dropout(x_s3)
#         x_s3 = self.conv1_s1(x_s3)
#         x_s3 = self.conv2_s1(x_s3)
#         x_s3 = self.maxpool2_s1(x_s3)
#         xx = torch.concat([x_s1, x_s2, x_s3], dim=-1)
#         xx = self.dropout(xx)
#         # aggregation layer
#         xx_bypass1 = xx.clone().detach().requires_grad_(True)
#         xx = self.resi_conv1(xx)
#         xx_bypass2 = self.relu(xx)
#         xx = self.resi_conv2(xx)
#         xx = self.avgpool(xx)
#         xx = self.fc0(xx)
#         xx = self.relu(xx)
#         xx = self.fc1(xx)
#         # xx = torch.mul(torch.mul(xx, xx_bypass2), xx_bypass1)   # point-wise multiply
#         xx = torch.mul(xx, xx_bypass2)   # point-wise multiply
#         # BiLSTM-AM
#         xx = torch.transpose(xx, 1, 2)
#         # output, (hn, cn) = self.lstm(xx)
#         output = self.transformer_encoder(xx)
#         output = torch.transpose(output, 1, 2)
#
#         if mean:
#             output = torch.mean(output, dim=2).unsqueeze(2)
#         return output
#
#
# class MagicDecoder(nn.Module):
#     def __init__(self, basic_cfg, cnn_emb_dim, lstm_emb_dim):
#         super(MagicDecoder, self).__init__()
#         self.cnn_ed  = cnn_emb_dim
#         self.lstm_ed = lstm_emb_dim
#         self.device = basic_cfg.d['device']
#         # BiLSTM-AM
#         # self.lstm = nn.LSTM(input_size=512, hidden_size=64, num_layers=3, dropout=0.3, bidirectional=True, batch_first=True)
#
#         encoder_layer = nn.TransformerEncoderLayer(d_model=self.cnn_ed, nhead=8, dim_feedforward=1024, batch_first=True)
#         self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=3)
#
#         # Multiscale CNN
#         self.cnn_s1 = nn.ConvTranspose1d(in_channels=128, out_channels=64, kernel_size=(125,), stride=(10,))
#         self.cnn_s2 = nn.ConvTranspose1d(in_channels=128, out_channels=64, kernel_size=(40,), stride=(5,))
#         self.cnn_s3 = nn.ConvTranspose1d(in_channels=128, out_channels=64, kernel_size=(5,), stride=(1,))
#         self.maxpool1_s1 = nn.MaxPool1d(kernel_size=(4,), stride=(2,))
#         self.maxpool1_s2 = nn.MaxPool1d(kernel_size=(4,), stride=(2,))
#         self.maxpool1_s3 = nn.MaxPool1d(kernel_size=(4,), stride=(2,))
#         self.dropout = nn.Dropout(0.5)
#         self.conv1_s1 = nn.ConvTranspose1d(in_channels=64,  out_channels=32, kernel_size=(7,), stride=(1,))
#         self.conv1_s2 = nn.ConvTranspose1d(in_channels=64,  out_channels=32, kernel_size=(7,), stride=(1,))
#         self.conv1_s3 = nn.ConvTranspose1d(in_channels=64,  out_channels=32, kernel_size=(8,), stride=(1,))
#         self.conv2_s1 = nn.ConvTranspose1d(in_channels=32, out_channels=32, kernel_size=(7,), stride=(1,))
#         self.conv2_s2 = nn.ConvTranspose1d(in_channels=32, out_channels=32, kernel_size=(7,), stride=(1,))
#         self.conv2_s3 = nn.ConvTranspose1d(in_channels=32, out_channels=32, kernel_size=(8,), stride=(1,))
#         self.maxpool2_s1 = nn.MaxPool1d(kernel_size=(2,), stride=(2,))
#         self.maxpool2_s2 = nn.MaxPool1d(kernel_size=(2,), stride=(2,))
#         self.maxpool2_s3 = nn.MaxPool1d(kernel_size=(4,), stride=(4,))
#         # Aggregation Layer
#         self.resi_conv1 = nn.ConvTranspose1d(in_channels=32, out_channels=30, kernel_size=(1,), stride=(1,))
#         self.resi_conv2 = nn.ConvTranspose1d(in_channels=30,  out_channels=30, kernel_size=(1,), stride=(1,))
#         self.avgpool = nn.AvgPool1d(1,1)
#         self.fc0 = nn.Linear(in_features=844, out_features=32)
#         self.fc1 = nn.Linear(in_features=32, out_features=1)
#         self.relu = nn.ReLU()
#         self.sigmoid = nn.Sigmoid()
#
#     def forward(self, x):
#         batch_size, _, emb_dim = x.shape
#         # BiLSTM-AM
#         x = torch.transpose(x, 1, 2)
#         # output, (hn, cn) = self.lstm(x)
#         output = self.transformer_encoder(x)
#         output = torch.transpose(output, 1, 2)
#
#         # scale 1 CNN
#         x_s1 = self.cnn_s1(output)
#         x_s1 = self.maxpool1_s1(x_s1)
#         x_s1 = self.dropout(x_s1)
#         x_s1 = self.conv1_s1(x_s1)
#         x_s1 = self.conv2_s1(x_s1)
#         x_s1 = self.maxpool2_s1(x_s1)
#         # scale 2 CNN
#         x_s2 = self.cnn_s2(output)
#         x_s2 = self.maxpool1_s2(x_s2)
#         x_s2 = self.dropout(x_s2)
#         x_s2 = self.conv1_s2(x_s2)
#         x_s2 = self.conv2_s2(x_s2)
#         x_s2 = self.maxpool2_s1(x_s2)
#         # scale 3 CNN
#         x_s3 = self.cnn_s1(output)
#         x_s3 = self.maxpool1_s1(x_s3)
#         x_s3 = self.dropout(x_s3)
#         x_s3 = self.conv1_s1(x_s3)
#         x_s3 = self.conv2_s1(x_s3)
#         x_s3 = self.maxpool2_s1(x_s3)
#         xx = torch.concat([x_s1, x_s2, x_s3], dim=-1)
#         xx = self.dropout(xx)
#         # aggregation layer
#         xx_bypass1 = xx.clone().detach().requires_grad_(True)
#         xx = self.resi_conv1(xx)
#         xx_bypass2 = self.relu(xx)
#         xx = self.resi_conv2(xx)
#         xx = self.avgpool(xx)
#         xx = self.fc0(xx)
#         xx = self.relu(xx)
#         xx = self.fc1(xx)
#         # xx = torch.mul(torch.mul(xx, xx_bypass2), xx_bypass1)   # point-wise multiply
#         xx = torch.mul(xx, xx_bypass2)   # point-wise multiply
#
#         xx = torch.mean(xx.to(self.device), dim=1)
#
#         xx = transforms.Resize([batch_size, 1500])(xx.unsqueeze(0))[0, :, :]
#         return xx
#
#
# class MLPDecoder(nn.Module):
#     def __init__(self, indim, outdim):
#         super(MLPDecoder, self).__init__()
#         self.detector = nn.Sequential(
#             nn.Linear(in_features=indim, out_features=indim * 2),
#             nn.ReLU(inplace=True),
#             nn.Linear(in_features=indim * 2, out_features=indim * 2),
#             nn.ReLU(inplace=True),
#             nn.Linear(in_features=indim * 2, out_features=outdim),
#         )
#         self.softmax = nn.Softmax(dim=2)
#
#     def forward(self, x):
#         x = self.detector(x)
#         x = self.softmax(x)
#         return x


# class MLP(nn.Module):
#     def __init__(self, indim, outdim):
#         super(MLP, self).__init__()
#         self.mlp = nn.Sequential(
#             nn.Linear(in_features=indim, out_features=indim // 2),
#             nn.ReLU(inplace=True),
#             nn.Linear(in_features=indim // 2, out_features=indim // 4),
#             nn.ReLU(inplace=True),
#             nn.Linear(in_features=indim // 4, out_features=outdim),
#         )
#
#         self.apply(_weights_init)
#
#     def forward(self, x):
#         x = self.mlp(x)
#         return x

