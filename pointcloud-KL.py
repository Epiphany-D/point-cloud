from typing import Optional, Callable, Union

import numpy as np
import pandas as pd
import scanpy as sc
import torch
from scipy.sparse.csr import csr_matrix
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances
from sklearn.metrics.cluster import adjusted_rand_score
from torch import Tensor
from torch.nn import Sequential as Seq, Dropout, Linear as Lin, ReLU, LayerNorm as LN
from torch.nn.parameter import Parameter
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GAE
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.inits import reset
from torch_geometric.nn.pool import radius
from torch_geometric.typing import OptTensor, PairOptTensor, PairTensor, Adj
from torch_geometric.utils import remove_self_loops, add_self_loops
from torch_sparse import SparseTensor, set_diag

from pytorchtools import EarlyStopping
# import faiss
from torch_utils import to_numpy, to_torch

EPS = 1e-15
MAX_LOGSTD = 10


def prefilter_genes(adata, min_counts=None, max_counts=None, min_cells=10, max_cells=None):
    if min_cells is None and min_counts is None and max_cells is None and max_counts is None:
        raise ValueError('Provide one of min_counts, min_genes, max_counts or max_genes.')
    id_tmp = np.asarray([True] * adata.shape[1], dtype=bool)
    id_tmp = np.logical_and(id_tmp,
                            sc.pp.filter_genes(adata.X, min_cells=min_cells)[0]) if min_cells is not None else id_tmp
    id_tmp = np.logical_and(id_tmp,
                            sc.pp.filter_genes(adata.X, max_cells=max_cells)[0]) if max_cells is not None else id_tmp
    id_tmp = np.logical_and(id_tmp,
                            sc.pp.filter_genes(adata.X, min_counts=min_counts)[0]) if min_counts is not None else id_tmp
    id_tmp = np.logical_and(id_tmp,
                            sc.pp.filter_genes(adata.X, max_counts=max_counts)[0]) if max_counts is not None else id_tmp
    adata._inplace_subset_var(id_tmp)


def load_data(spaFll_path, metaData_path, pca_dim):
    data_list = []
    a = np.load(f'{spaFll_path}')  # 参数
    b = pd.read_csv(f'{metaData_path}')
    e = csr_matrix((a['data'], a['indices'], a['indptr']), dtype=int)
    e = csr_matrix.todense(e)  # feature [3639,33538]

    # 数据预处理
    adata = sc.AnnData(X=e)
    prefilter_genes(adata, min_cells=3)  # avoiding all genes are zeros  to_dense后的 expression
    # Normalize and take log for UMI-------
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    e = adata.X

    pca = PCA(n_components=pca_dim)  # 值 参数
    e = pca.fit_transform(e)  # expression [3639,50]

    coor = b.loc[:, ['pxl_col_in_fullres', 'pxl_row_in_fullres']].values  # pos [3639,2]
    array = b.loc[:, ['array_row', 'array_col']].values
    RGB = b.loc[:, ['R', 'G', 'B']].values  # RGB [3639,3]

    layer = b['layer'].tolist()
    label = []
    delete_index = []
    for index in range(len(layer)):
        if layer[index] == 'Layer1':
            label.append(1)
        elif layer[index] == 'Layer2':
            label.append(2)
        elif layer[index] == 'Layer3':
            label.append(3)
        elif layer[index] == 'Layer4':
            label.append(4)
        elif layer[index] == 'Layer5':
            label.append(5)
        elif layer[index] == 'Layer6':
            label.append(6)
        elif layer[index] == 'wm':
            label.append(7)
        else:
            label.append(0)

    e = np.delete(e, delete_index, axis=0)
    pos = np.delete(pos, delete_index, axis=0)
    RGB = np.delete(RGB, delete_index, axis=0)

    expression = torch.from_numpy(e.astype(np.float32))
    # print(expression.size())
    pos = torch.from_numpy(pos.astype(np.float32))
    # print(pos.size())
    RGB = torch.from_numpy(RGB.astype(np.float32))
    # print(RGB.size())
    label = torch.tensor(label)
    # print(label.size())

    # fea = torch.cat((expression, RGB), axis=1)      # fea[3639,53]
    fea = expression
    data = Data(pos=array, x=fea, y=label)
    data.expression = expression
    data.coor = coor
    data = data.to(device)
    data_list.append(data)
    print("load data")
    return data_list


def distance_matrix(data):  # 距离矩阵
    dis_matrix = pairwise_distances(data.pos.cpu())
    dis_matrix = torch.from_numpy(dis_matrix)
    return dis_matrix


def local_nn(in_channels, out_channels):
    return Seq(*[Lin(in_channels, out_channels), LN(out_channels)])


def global_nn():
    return Seq(*[ReLU(), Dropout(0.1)])


class PointConv(MessagePassing):

    def __init__(self, local_nn: Optional[Callable] = None,
                 global_nn: Optional[Callable] = None,
                 add_self_loops: bool = True, **kwargs):
        super(PointConv, self).__init__(aggr='mean', **kwargs)

        self.local_nn = local_nn
        self.global_nn = global_nn
        self.add_self_loops = add_self_loops

        self.reset_parameters()

    def reset_parameters(self):
        reset(self.local_nn)
        reset(self.global_nn)

    def forward(self, x: Union[OptTensor, PairOptTensor],
                pos: Union[Tensor, PairTensor], edge_index: Adj) -> Tensor:
        """"""
        if not isinstance(x, tuple):
            x: PairOptTensor = (x, None)

        if isinstance(pos, Tensor):
            pos: PairTensor = (pos, pos)

        if self.add_self_loops:
            if isinstance(edge_index, Tensor):
                edge_index, _ = remove_self_loops(edge_index)
                edge_index, _ = add_self_loops(edge_index,
                                               num_nodes=pos[1].size(0))
            elif isinstance(edge_index, SparseTensor):
                edge_index = set_diag(edge_index)

        # propagate_type: (x: PairOptTensor, pos: PairTensor)
        out = self.propagate(edge_index, x=x, pos=pos, size=None)

        if self.global_nn is not None:
            out = self.global_nn(out)

        return out

    def message(self, x_j: Optional[Tensor], pos_i: Tensor,
                pos_j: Tensor) -> Tensor:
        msg = pos_j - pos_i
        if x_j is not None:
            msg = torch.cat([x_j, msg], dim=1)
        if self.local_nn is not None:
            msg = self.local_nn(msg)
        return msg

    def __repr__(self):
        return '{}(local_nn={}, global_nn={})'.format(self.__class__.__name__,
                                                      self.local_nn,
                                                      self.global_nn)


class PointNet(torch.nn.Module):
    def __init__(self, r, local_nn, global_nn):
        super(PointNet, self).__init__()
        self.r = r
        self.conv = PointConv(local_nn=local_nn, global_nn=global_nn)

    def forward(self, x, pos, batch):
        row, col = radius(x=pos, y=pos, r=self.r, batch_x=batch, batch_y=batch, max_num_neighbors=128)  # 找r半径中的点
        edge_index = torch.stack([col, row], dim=0)
        x = self.conv(x, pos, edge_index)
        return x


class PointEncoder(torch.nn.Module):
    def __init__(self, out_channels=3):
        super().__init__()
        self.conv1 = PointNet(1.5, local_nn(50 + 2, 64), global_nn())
        self.conv2 = PointNet(2.5, local_nn(50 + 2, 64), global_nn())
        # self.lin1 = MLP([64 + 64 + 64, 256])
        self.mlp = Lin(64 + 64, out_channels)

    def forward(self, data):
        x0, pos, batch, = data.x, data.pos, data.batch
        x1 = self.conv1(x0, pos, batch)
        x2 = self.conv2(x0, pos, batch)
        # x4 = self.lin1(torch.cat([x1, x2, x3], dim=1))
        out = self.mlp(torch.cat([x1, x2], dim=1))
        return out


def label_generator_kmeans(features, num_classes=500, cuda=True, **kwargs):
    # assert cfg.TRAIN.PSEUDO_LABELS.cluster == "kmeans"
    # assert num_classes, "num_classes for kmeans is null"
    #
    # # num_classes = cfg.TRAIN.PSEUDO_LABELS.cluster_num
    #
    # if not cfg.TRAIN.PSEUDO_LABELS.use_outliers:
    #     warnings.warn("there exists no outlier point by kmeans clustering")

    # k-means cluster by faiss
    cluster = faiss.Kmeans(
        features.size(-1), num_classes, niter=300, verbose=False, gpu=cuda
    )

    cluster.train(to_numpy(features))

    centers = to_torch(cluster.centroids).float()
    _, labels = cluster.index.search(to_numpy(features), 1)
    labels = labels.reshape(-1)
    labels = to_torch(labels).long()
    # k-means does not have outlier points
    assert not (-1 in labels)

    return labels, centers, num_classes


class InnerProductDecoder(torch.nn.Module):
    def forward(self, z):
        adj = torch.matmul(z, z.t())
        return adj


class Decoder(torch.nn.Module):
    def forward(self, z, init="louvain"):
        adj = torch.matmul(z, z.t())
        self.alpha = 0.2
        if init == "kmeans":
            y_pred, center, _ = label_generator_kmeans(z, 8)

        elif init == "louvain":
            adata = sc.AnnData(z.cpu().detach().numpy())
            sc.pp.neighbors(adata, n_neighbors=10)
            sc.tl.louvain(adata, resolution=0.4)
            y_pred = adata.obs['louvain'].astype(int).to_numpy()
            self.n_clusters = len(np.unique(y_pred))
            self.trajectory = []
            self.mu = Parameter(torch.Tensor(self.n_clusters, 3))
            self.trajectory.append(y_pred)
            features = pd.DataFrame(z.detach().cpu().numpy(), index=np.arange(0, z.shape[0]))  # z
            Group = pd.Series(y_pred, index=np.arange(0, features.shape[0]), name="Group")
            Mergefeature = pd.concat([features, Group], axis=1)
            cluster_centers = np.asarray(Mergefeature.groupby("Group").mean())
            self.mu.data.copy_(torch.Tensor(cluster_centers))
            center = self.mu

        center = center.to(device)
        q = 1.0 / ((1.0 + torch.sum((z.unsqueeze(1) - center) ** 2, dim=2) / self.alpha) + 1e-8)
        q = q ** (self.alpha + 1.0) / 2.0
        q = q / torch.sum(q, dim=1, keepdim=True)
        p = q ** 2 / torch.sum(q, dim=0)
        p = p / torch.sum(p, dim=1, keepdim=True)
        return adj, p, q, y_pred


class GAE(torch.nn.Module):
    def __init__(self, encoder, decoder):
        super(GAE, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        GAE.reset_parameters(self)

    def reset_parameters(self):
        reset(self.encoder)
        reset(self.decoder)

    def encode(self, *args, **kwargs):
        r"""Runs the encoder and computes node-wise latent variables."""
        return self.encoder(*args, **kwargs)

    def decode(self, *args, **kwargs):
        r"""Runs the decoder and computes edge probabilities."""
        return self.decoder(*args, **kwargs)

    def loss(self, z, distance_matrix):
        adj, p, q, y_pred = self.decode(z)
        # dis_loss = F.kl_div(F.log_softmax(adj, dim=1), F.softmax(distance_matrix, dim=1), reduction='batchmean')
        cluster_loss = torch.abs(torch.mean(torch.sum(p * torch.log(p / (q + 1e-6)), dim=1)))
        # loss = cluster_loss*0.8+dis_loss*0.2
        return cluster_loss, y_pred


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
a_path = '/home/19yynenu/singlecell/pointcloud/data/151673_humanBrain_scran_spaFull.npz'
b_path = '/home/19yynenu/singlecell/pointcloud/data/151673_humanBrain_metaData.csv'
pc_num = 50
dataset = load_data(spaFll_path=a_path, metaData_path=b_path, pca_dim=pc_num)

model = GAE(PointEncoder(), Decoder())
model = model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=0.0005)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.8, patience=10, verbose=True)
n_epochs = 500
patience = 20
dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
early_stopping = EarlyStopping(patience=patience, verbose=True, path='result/checkpoint.pt')

for epoch in range(1, n_epochs + 1):
    train_losses = []
    model.train()
    node_embedding = []
    for data in dataloader:
        data = data.to(device)
        dis_matrix = distance_matrix(data).to(device)
        # print(dis_matrix)
        optimizer.zero_grad()
        z = model.encode(data)  # node embedding
        loss, _ = model.loss(z, dis_matrix)
        loss.backward()
        optimizer.step()
        scheduler.step(loss)
        train_losses.append(loss.item())

    train_loss = np.average(train_losses)
    epoch_len = len(str(n_epochs))
    print_msg = (f'[{epoch:>{epoch_len}}/{n_epochs:>{epoch_len}}] ' +
                 f'train_loss: {train_loss:.5f} ')

    print(print_msg)

    early_stopping(train_loss, model)

    if early_stopping.early_stop:
        print("Early stopping")
        break

newmodel = torch.load('result/checkpoint.pt')
newmodel.eval()
for data in dataloader:
    data = data.to(device)
    dis_matrix = distance_matrix(data).to(device)
    z = newmodel.encode(data)
    _, y_pred = newmodel.loss(z, dis_matrix)
    ARI = adjusted_rand_score(data.y.cpu().detach().numpy(), y_pred)
    print('ARI: ' + str(ARI))
    # barcode = b['barcode'].tolist()
    # embedding = z.detach().cpu().numpy()
    # embed = pd.DataFrame(embedding)
    # c = {"barcode":barcode}
    # bar = pd.DataFrame(c)
    # result = pd.concat([bar,embed],axis=1)
    # result.columns=['barcode','embedding0','embedding1','embedding2']
    # result.to_csv("result/node_embeddingwithRGB_louvain.csv",index=False)
    # np.savetxt("result/node_embedding_MSE.csv", z.detach().cpu().numpy(), delimiter=',')
