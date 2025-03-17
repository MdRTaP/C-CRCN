

import copy
import torch
from torch_geometric.utils.dropout import dropout_adj
from torch_geometric.transforms import Compose

agmt_dict = {"coauthor-physics":[0.4, 0.1, 0.1, 0.4],
             "coauthor-cs": [0.3, 0.3, 0.2, 0.4],
             "amazon-computers": [0.5, 0.2, 0.4, 0.1],
             "amazon-photos": [0.4, 0.1, 0.1, 0.2],
             "wiki-cs": [0.2, 0.2, 0.3, 0.1],
             "cora": [0.2, 0.2, 0.3, 0.1],
             "pubmed": [0.5, 0.2, 0.6, 0.1],
             "citeseer": [0.2, 0.2, 0.3, 0.1],
             "lasftm-asia": [0.2, 0.2, 0.3, 0.1],
             }

class DropFeatures:
    def __init__(self, p=None, precomputed_weights=True):
        self.p = p

    def __call__(self, data):
        drop_mask = torch.empty((data.x.size(1),), dtype=torch.float32, device=data.x.device).uniform_(0, 1) < self.p
        data.x[:, drop_mask] = 0
        return data

class DropEdges:
    def __init__(self, p, force_undirected=False):
        self.p = p
        self.force_undirected = force_undirected

    def __call__(self, data):
        edge_index = data.edge_index
        edge_attr = data.edge_attr if 'edge_attr' in data else None
        edge_index, edge_attr = dropout_adj(edge_index, edge_attr, p=self.p, force_undirected=self.force_undirected)
        data.edge_index = edge_index
        if edge_attr is not None:
            data.edge_attr = edge_attr
        return data

def augment_graph(drop_edge_p, drop_feat_p):
    augments = list()
    augments.append(copy.deepcopy)

    if drop_edge_p > 0.:
        augments.append(DropEdges(drop_edge_p))

    if drop_feat_p > 0.:
        augments.append(DropFeatures(drop_feat_p))
    return Compose(augments)