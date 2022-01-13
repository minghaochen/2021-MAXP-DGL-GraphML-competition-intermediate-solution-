import torch
import torch.nn.functional as F
import dgl
# from utils import load_dgl_graph, time_diff
# from models import GraphSageModel, GraphConvModel, GraphAttnModel
# from dgl.dataloading.neighbor import MultiLayerNeighborSampler
# from my_models import GAT
import os
import numpy as np
from dgl.dataloading.pytorch import NodeDataLoader
import scipy.sparse as sp
# import tqdm as tqdm
import pickle
import pandas as pd
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

BASE_PATH = ''
device_id = torch.device("cpu")

graph, labels, train_nid, val_nid, test_nid, node_feat = load_dgl_graph(BASE_PATH)
labels = labels.to(device=device_id, dtype=torch.int64) # 自己加的

graph = dgl.to_bidirected(graph, copy_ndata=True)
graph = dgl.add_self_loop(graph)

split_idx = {}
split_idx["train"] = torch.tensor(train_nid)
split_idx["valid"] = torch.tensor(val_nid)
split_idx["test"] = torch.tensor(test_nid)


degs = graph.out_degrees().float().clamp(min=1)
norm = torch.pow(degs, -0.5).numpy()
norm = sp.diags(norm)
DAD = norm.dot(graph.adj(scipy_fmt='coo')).dot(norm)
print(DAD.shape)
print(DAD)
DA = norm.dot(norm).dot(graph.adj(scipy_fmt='coo'))
AD = graph.adj(scipy_fmt='coo').dot(norm).dot(norm)
print(DA)
print(AD)
def get_labels_from_name(labels_set, split_idx):
    if isinstance(labels_set, list):
        labels_set = list(labels_set)
        if len(labels_set) == 0:
            return torch.tensor([])
        for idx, i in enumerate(list(labels_set)):
            labels_set[idx] = split_idx[i]
        residual_idx = torch.cat(labels_set)
    else:
        residual_idx = split_idx[labels_set]
    return residual_idx


def only_outcome_correlation(data, model_out, split_idx, A, alpha, num_propagations, labels, device='cpu', display=True):
    # res_result = model_out.clone()
    res_result = model_out.copy()
    label_idxs = get_labels_from_name(labels, split_idx)
    y = pre_outcome_correlation(labels=data, model_out=model_out, label_idx=label_idxs)
    result = general_outcome_correlation(adj=A, y=y, alpha=alpha, num_propagations=num_propagations, post_step=lambda x: np.clip(x, 0, 1), alpha_term=True, display=display, device=device)
    return res_result, result


def pre_residual_correlation(labels, model_out, label_idx):
    """Generates the initial labels used for residual correlation"""
    labels = labels.cpu()
    labels[labels.isnan()] = 0
    labels = labels.long()
    # model_out = model_out.cpu()
    label_idx = label_idx.cpu()
    c = labels.max() + 1
    n = labels.shape[0]
    # y = torch.zeros((n, c))
    y = np.zeros((n,c))
    y[label_idx] = F.one_hot(labels[label_idx], c).float().squeeze(1).numpy() - model_out[label_idx]
    return y


def pre_outcome_correlation(labels, model_out, label_idx):
    """Generates the initial labels used for outcome correlation"""

    labels = labels.cpu()
    # model_out = model_out.cpu()
    label_idx = label_idx.cpu()
    c = labels.max() + 1
    n = labels.shape[0]
    # y = model_out.clone()
    y = model_out.copy()
    if len(label_idx) > 0:
        y[label_idx] = F.one_hot(labels[label_idx] ,c).float().squeeze(1)

    return y


def general_outcome_correlation(adj, y, alpha, num_propagations, post_step, alpha_term, device='cuda', display=True):
    """general outcome correlation. alpha_term = True for outcome correlation, alpha_term = False for residual correlation"""
    # adj = adj.to(device)
    # orig_device = y.device
    # y = y.to(device)
    # result = y.clone()
    result = y.copy()
    for _ in range(num_propagations):
        if _ % 2 == 0:
            print("num_propagations:", _)
        result = alpha * (adj @ result)
        if alpha_term:
            result += (1-alpha)*y
        else:
            result += y
        result = post_step(result)
    # return result.to(orig_device)
    return result

def double_correlation_autoscale(data, model_out, split_idx, A1, alpha1, num_propagations1, A2, alpha2,
                                 num_propagations2, scale=1.0, train_only=False, device='cuda', display=True):
    train_idx, valid_idx, test_idx = split_idx
    if train_only:
        label_idx = torch.cat([split_idx['train']])
        residual_idx = split_idx['train']
    else:
        label_idx = torch.cat([split_idx['train'], split_idx['valid']])
        residual_idx = label_idx

    y = pre_residual_correlation(labels=data, model_out=model_out, label_idx=residual_idx)
    resid = general_outcome_correlation(adj=A1, y=y, alpha=alpha1, num_propagations=num_propagations1,
                                        post_step=lambda x: np.clip(x, -1, 1), alpha_term=True, display=display,
                                        device=device)

    # orig_diff = y[residual_idx].abs().sum() / residual_idx.shape[0]
    orig_diff = np.sum(np.abs(y[residual_idx])) / residual_idx.shape[0]
    # resid_scale = (orig_diff / resid.abs().sum(dim=1, keepdim=True))
    resid_scale = (orig_diff /np.sum(np.abs(resid), axis=1).reshape(-1, 1) )
    resid_scale[torch.tensor(resid_scale).isinf()] = 1.0
    cur_idxs = (resid_scale > 1000)
    resid_scale[cur_idxs] = 1.0
    res_result = model_out + resid_scale * resid
    res_result[torch.tensor(res_result).isnan()] = model_out[torch.tensor(res_result).isnan()]
    y = pre_outcome_correlation(labels=data, model_out=res_result, label_idx=label_idx)
    result = general_outcome_correlation(adj=A2, y=y, alpha=alpha2, num_propagations=num_propagations2,
                                         post_step=lambda x: np.clip(x, 0, 1), alpha_term=True, display=display,
                                         device=device)

    return res_result, result

gat_dict = {
            'labels': ['train'],
            'alpha': 0.8,
            'A': DAD,
            'num_propagations': 50,
            'display': False,
        }

mlp_dict = {
            'train_only': False,
            'alpha1': 0.9791632871592579,
            'alpha2': 0.7564990804200602,
            # 'alpha1': 0.99,
            # 'alpha2': 0.65,
            'A1': DA,
            'A2': AD,
            'num_propagations1': 50,
            'num_propagations2': 50,
            'display': True,
        }
        # mlp_fn = double_correlation_autoscale
# res_result, result = only_outcome_correlation(labels.reshape(-1,1), val_batch_list, split_idx, **gat_dict)
print("Correct and Smooth")
def evaluator(preds, labels):
    return torch.sum(preds == labels) / torch.tensor(labels.shape[0])


eval_inx = np.concatenate((train_nid,val_nid))

# 先cs再融合
# Z:\DataScience\DGL\ SAGN_with_SLE\intermediate_outputs\ogbn-papers100M\sagn
# for fold in range(7):
#     val_batch_list = torch.load(f'fold_{fold}.pt').cpu().numpy()
#     res_result, result = double_correlation_autoscale(labels.reshape(-1,1), val_batch_list, split_idx, **mlp_dict)
#     acc = evaluator(torch.tensor(np.argmax(result[eval_inx], axis=1)), labels[eval_inx])
#     print("after acc", acc)
#     if fold == 0:
#         res1 = res_result/7
#     else:
#         res1 += res_result/7
#
# res1 = res1[test_nid]

for fold in range(7):
    val_batch_list = torch.load(
        f'{fold}.pt').cpu().numpy()
    if fold == 0:
        res2 = val_batch_list / 7
    else:
        res2 += val_batch_list / 7

res_origin = res2.copy()
res_origin = res_origin[test_nid]

res_result, result = double_correlation_autoscale(labels.reshape(-1, 1), res2, split_idx, **mlp_dict)
acc = evaluator(torch.tensor(np.argmax(res_result[eval_inx], axis=1)), labels[eval_inx])
print("after res_result acc", acc)
acc = evaluator(torch.tensor(np.argmax(result[eval_inx], axis=1)), labels[eval_inx])
print("after result acc", acc)
res1 = res_result
res1 = res1[test_nid]
res2 = result # sagnk折比较高，和GAT融合不一定
res2 = res2[test_nid]

fname = "sagn_logits_1.pt"
torch.save(torch.tensor(res1).cpu(), fname) # 直接叠加再cs，然后选择 res_result 更高
fname = "sagn_logits_2.pt"
# torch.save(torch.tensor(result).cpu(), fname)
torch.save(torch.tensor(res2).cpu(), fname)


prediction = ["A","B","C","D","E","F","G",
              "H","I","J","K","L","M","N",
              "O","P","Q","R","S","T","U",
              "V","W"]
score_map = {}
for i in range(23):
    score_map[i] = prediction[i]



res_origin = np.argmax(res_origin, axis=1)
submit = pd.read_csv("sample_submission_for_validation.csv")
submit["label"] = res_origin[0:submit.shape[0]]
submit["label"] = submit["label"].map(score_map)
print(submit.head())
submit.to_csv("submission_sagn.csv", index=None)

res2 = np.argmax(res2, axis=1)
submit = pd.read_csv("sample_submission_for_validation.csv")
submit["label"] = res2[0:submit.shape[0]]
submit["label"] = submit["label"].map(score_map)
print(submit.head())
submit.to_csv("submission_sagn_cs.csv", index=None)