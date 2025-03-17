
import random
from torch_geometric.data import Data
import json
from torch.nn.functional import normalize
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T

import numpy as np
import torch

from torch_geometric import datasets
from torch_geometric.transforms import NormalizeFeatures
from torch_geometric.utils import to_undirected

def get_dataset(root, name, transform=NormalizeFeatures()):
    pyg_dataset_dict = {
        'coauthor-cs': (datasets.Coauthor, 'CS'),
        'coauthor-physics': (datasets.Coauthor, 'Physics'),
        'amazon-computers': (datasets.Amazon, 'Computers'),
        'amazon-photos': (datasets.Amazon, 'Photo'),
    }

    assert name in pyg_dataset_dict, "Dataset must be in {}".format(list(pyg_dataset_dict.keys()))

    dataset_class, name = pyg_dataset_dict[name]
    dataset = dataset_class(root, name=name, transform=transform)

    return dataset

def get_wiki_cs(root, transform=NormalizeFeatures()):
    dataset = datasets.WikiCS(root, is_undirected=True,transform=transform)
    data = dataset[0]
    std, mean = torch.std_mean(data.x, dim=0, unbiased=False)
    data.x = (data.x - mean) / std
    data.edge_index = to_undirected(data.edge_index)
    return [data], np.array(data.train_mask), np.array(data.val_mask), np.array(data.test_mask)

def get_lasftm_asia(dataset_name):
    graph_edges = "./data/lasftm-asia/lastfm_asia_edges.txt"
    graph_node_feature = "./data/lasftm-asia/lastfm_asia_features.json"
    graph_node_label = "./data/lasftm-asia/lastfm_asia_target.txt"

    start = []
    to = []
    for line in open(graph_edges):
        strlist = line.split()
        start.append(int(strlist[0]))
        to.append(int(strlist[1]))
    edge_index = torch.tensor([start, to], dtype=torch.int64)

    label_list = [int(line.split()[1]) for line in open(graph_node_label)]
    y = torch.tensor(label_list)

    x_values = []
    with open(graph_node_feature, 'r') as fp:
        json_data = json.load(fp)
    max_index = 7841 # max([max(v) for v in json_data.values() if len(v)>0])

    for raw_feat in json_data.values():
        mask = torch.tensor(raw_feat)
        x_value = torch.zeros(max_index+1)
        if len(raw_feat)>0:
           x_value[mask] = 1
        x_values.append(x_value.tolist())
    x = torch.tensor(x_values, dtype=torch.float32)
    x = normalize(x, p=2.0, dim=0)
    data = Data(x=x, edge_index=edge_index, y=y)
    return data

def get_planetoid(dataset_name):
    dataset = Planetoid("./data", dataset_name, transform=T.TargetIndegree())
    data = dataset[0]
    return data


def get_wiki(dataset_name):
    dataset, _, _,_, = get_wiki_cs('./data/wiki-cs')
    data = dataset[0]
    return data

def get_common_dataset(dataset_name):
    dataset = get_dataset('./data', dataset_name)
    data = dataset[0]
    return data

def load_dataset(dataset_name):
    loader_dict = {"amazon-computers":get_common_dataset,
                   "amazon-photos": get_common_dataset,
                   "coauthor-cs": get_common_dataset,
                   "coauthor-physics": get_common_dataset,
                   "wiki-cs": get_wiki,
                   "cora": get_planetoid,
                   "citeseer": get_planetoid,
                   "pubmed": get_planetoid,
                   "lasftm-asia": get_lasftm_asia
                   }

    data = loader_dict[dataset_name.lower()](dataset_name.lower())
    data.num_classes = torch.max(data.y, dim=0)[0].item() + 1

    return data

def make_pu_dataset(data, pos_index=[0], fixed_seed=True,
                    sample_seed=5, train_pct=0.07, val_pct=0.03, test_pct=1.0):

    data.y = sum([data.y == idx for idx in pos_index]).long()
    data.train_mask = torch.zeros(data.y.size()).bool().fill_(False)
    data.val_mask = torch.zeros(data.y.size()).bool().fill_(False)


    # return indices of positive nodes, view(-1): as a 1-dim tuple
    pos_idx = data.y.nonzero(as_tuple=False).view(-1)
    pos_num = pos_idx.size(0)

    pos_idx_list = pos_idx.tolist()
    if fixed_seed:
        random.seed(sample_seed)
    random.shuffle(pos_idx_list)

    train_val_list = pos_idx_list[:round(pos_num * (train_pct + val_pct))]
    random.shuffle(train_val_list)

    # sample train-positive and val-positive
    pos_idx = torch.tensor(train_val_list)
    train_idx = pos_idx[:round(pos_num * train_pct)]
    val_idx = pos_idx[round(pos_num * train_pct):round(pos_num * (train_pct+val_pct))]
    data.train_mask[train_idx] = True
    data.val_mask[val_idx] = True
    data.pos_train_mask = data.train_mask  # re-name, only for easy use

    # negative train mask: neg_train_mask, only contains negative nodes
    data.neg_train_mask = torch.zeros(data.y.size()).bool().fill_(False)

    neg_idx = (data.y == 0).nonzero(as_tuple=False).view(-1)
    perm_neg_idx = neg_idx[torch.randperm(neg_idx.size(0))]
    neg_train_idx = perm_neg_idx[:round(pos_num * train_pct)] # The number is the same as that of pos_train_mask
    data.neg_train_mask[neg_train_idx] = True

    # un_train_mask--unlabeled train mask (may contain both P and N nodes), and test_mask
    data.un_train_mask = torch.zeros(data.y.size()).bool().fill_(False)
    data.gx_un_train_mask = torch.zeros(data.y.size()).bool().fill_(False)
    data.un_val_mask = torch.zeros(data.y.size()).bool().fill_(False)
    data.test_mask = torch.zeros(data.y.size()).bool().fill_(False)





    # remaining index and permutate
    remaining_idx = (~(data.train_mask | data.val_mask)).nonzero(as_tuple=False).view(-1)
    perm_remaining_idx = remaining_idx[torch.randperm(remaining_idx.size(0))]

    # sample
    data.un_train_mask[perm_remaining_idx[:round(pos_num * train_pct)]] = True # The number is the same as that of pos_train_mask
    data.gx_un_train_mask[perm_remaining_idx[:]] = True
    data.un_val_mask[perm_remaining_idx[:val_idx.size(0)]] = True
    data.test_mask[perm_remaining_idx[:round(perm_remaining_idx.size(0) * test_pct)]] = True  # may overlap un_train_mask

    data.prior = data.y.sum().item() / data.y.size(0)
    data.num_classes = torch.max(data.y, dim=0)[0].item() + 1
    return data
