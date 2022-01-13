import os
from functools import namedtuple
import dgl
import dgl.function as fn
import numpy as np
import torch
from dgl.data import PPIDataset
# from ogb.nodeproppred import DglNodePropPredDataset, Evaluator
from sklearn.metrics import accuracy_score, f1_score
import scipy.sparse as sp
import json
import pickle
import gc
import random
from networkx.readwrite import json_graph

class ACCEvaluator(object):

    def __init__(self):
        pass

    def __call__(self, y_pred, y_true):

        return accuracy_score(y_true.cpu(), y_pred.cpu())


def load_dataset(device, args, fold, mask,test_mode):
    """
    Load dataset and move graph and features to device
    """
    # base_path = 'Z:\DataScience\DGL'
    base_path = 'E:\ZJL\DGL'
    graph, labels, tr_label_idx, val_label_idx, test_label_idx, node_feat = load_dgl_graph_k_fold(base_path,fold=fold, mask=mask, test_mode=test_mode)
    labels = labels.to(device=device, dtype=torch.int64)  # 自己加的

    # graph = dgl.add_reverse_edges(graph, copy_ndata=True)
    ############# 转换单向图 ###############
    # src_nid = graph.edges()[0].numpy().reshape(-1,)
    # dst_nid = graph.edges()[1].numpy().reshape(-1,)
    # graph = dgl.graph((dst_nid, src_nid))
    ############# 考虑单向图和双向图 ###############
    graph = dgl.to_bidirected(graph, copy_ndata=True)
    graph = dgl.add_self_loop(graph)



    # 标准化
    degs = graph.out_degrees().float().clamp(min=1)
    norm = torch.pow(degs, -0.5).numpy()
    norm = sp.diags(norm)
    DAD = norm.dot(graph.adj(scipy_fmt='coo')).dot(norm)
    # print(DAD)
    graph = dgl.from_scipy(DAD)

    graph.ndata['feat'] = node_feat

    n_classes = 23
    train_nid, val_nid, test_nid = torch.tensor(tr_label_idx).long(), torch.tensor(val_label_idx).long(), torch.tensor(test_label_idx).long()

    graph = graph.to(device)

    return graph, labels, n_classes, train_nid, val_nid, test_nid


from sklearn.model_selection import StratifiedKFold

device = torch.device("cpu")
def add_labels(feat, labels, idx):
    onehot = torch.zeros([feat.shape[0], 23]).to(device)
    # onehot[idx, labels[idx, 0]] = 1
    onehot[idx, labels[idx]] = 1
    return torch.cat([feat, onehot], dim=-1)

def load_dgl_graph_k_fold(base_path, fold=-1, mask =0, k=10, seed=1996, test_mode=False):
    """
    读取预处理的Graph，Feature和Label文件，并构建相应的数据供训练代码使用。

    :param base_path:
    :return:
    """
    graphs, _ = dgl.load_graphs(os.path.join(base_path, 'graph.bin'))
    graph = graphs[0]
    # print('################ Graph info: ###############')
    # print(graph)

    with open(os.path.join(base_path, 'labels.pkl'), 'rb') as f:
        label_data = pickle.load(f)

    labels = torch.from_numpy(label_data['label'])
    test_label_idx = label_data['test_label_idx']

    if fold == -1:
        tr_label_idx = label_data['tr_label_idx']
        val_label_idx = label_data['val_label_idx']
    else:
        train_idx = np.concatenate((label_data['tr_label_idx'], label_data['val_label_idx']))
        folds = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
        for i, (tr, val) in enumerate(folds.split(train_idx, labels[train_idx])):
            tr_label_idx, val_label_idx = train_idx[tr], train_idx[val]
            if i == fold:
                # print('    ###      use      fold: {}'.format(fold))
                break

    # print('################ Label info: ################')
    # print('Total labels (including not labeled): {}'.format(labels.shape[0]))
    # print('               Training label number: {}'.format(tr_label_idx.shape[0]))
    # print('             Validation label number: {}'.format(val_label_idx.shape[0]))
    # print('                   Test label number: {}'.format(test_label_idx.shape[0]))

    # get node features
    features = np.load(os.path.join(base_path, 'features.npy'))
    # pretrain_feature = np.load(os.path.join(base_path, 'pretrain_fea'))
    # features = np.load(os.path.join(base_path, 'pretrain_feature.npy'))
    # features_n2v = np.load(os.path.join(base_path, 'features_n2v.npy'))
    # features = np.hstack([features,features_n2v])
    # del features_n2v
    # node_info = np.load(os.path.join(base_path, 'node_info.npy'))
    # features = np.hstack([features, pretrain_feature])
    node_feat = torch.from_numpy(features).float()
    del features
    gc.collect()

    print("fold", fold)
    print("mask", mask)
    mask_rate = 0.5


    seed = random.randint(0,2021)
    torch.manual_seed(seed)
    print("seed", seed)
    # torch.manual_seed(fold)
    if mask == 0:
        mask = torch.rand(tr_label_idx.shape) < mask_rate
    else:
        mask = torch.rand(tr_label_idx.shape) >= mask_rate
    train_labels_idx = tr_label_idx[mask]
    train_pred_idx = tr_label_idx[~mask]
    labels = labels.to(device=device, dtype=torch.int64)
    if test_mode:
        print("################## TEST MODE ###################")
        train_labels_idx = np.hstack([label_data['tr_label_idx'],label_data['val_label_idx']])

    node_feat = add_labels(node_feat, labels, train_labels_idx)

    print('################ Feature info: ###############')

    print('Node\'s feature shape:{}'.format(node_feat.shape))

    print(val_label_idx[:10])
    # return graph, labels, tr_label_idx, val_label_idx, test_label_idx, node_feat
    return graph, labels, train_pred_idx, val_label_idx, test_label_idx, node_feat

def load_dgl_graph(base_path):
    """
    读取预处理的Graph，Feature和Label文件，并构建相应的数据供训练代码使用。

    :param base_path:
    :return:
    """
    graphs, _ = dgl.load_graphs(os.path.join(base_path, 'graph.bin'))
    graph = graphs[0]
    print('################ Graph info: ###############')
    print(graph)

    with open(os.path.join(base_path, 'labels.pkl'), 'rb') as f:
        label_data = pickle.load(f)

    labels = torch.from_numpy(label_data['label'])
    tr_label_idx = label_data['tr_label_idx']
    val_label_idx = label_data['val_label_idx']
    test_label_idx = label_data['test_label_idx']
    print('################ Label info: ################')
    print('Total labels (including not labeled): {}'.format(labels.shape[0]))
    print('               Training label number: {}'.format(tr_label_idx.shape[0]))
    print('             Validation label number: {}'.format(val_label_idx.shape[0]))
    print('                   Test label number: {}'.format(test_label_idx.shape[0]))

    # get node features
    features = np.load(os.path.join(base_path, 'features.npy'))
    node_feat = torch.from_numpy(features).float()
    print('################ Feature info: ###############')
    print('Node\'s feature shape:{}'.format(node_feat.shape))

    return graph, labels, tr_label_idx, val_label_idx, test_label_idx, node_feat

