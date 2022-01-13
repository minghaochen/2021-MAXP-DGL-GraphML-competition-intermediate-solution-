import torch
import numpy as np
import pandas as pd
import torch.nn.functional as F
import dgl
import os
import pickle

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

BASE_PATH = 'E:\ZJL\DGL'
device_id = torch.device("cpu")

graph, labels, train_nid, val_nid, test_nid, node_feat = load_dgl_graph(BASE_PATH)


submit = pd.read_csv("E:/ZJL/DGL/sample_submission_for_validation.csv")

def props_to_onehot(props):
    if isinstance(props, list):
        props = np.array(props)
    a = np.argmax(props, axis=1)
    b = np.zeros((len(a), props.shape[1]))
    b[np.arange(len(a)), a] = 1
    return b

sub1 = np.load('my_gat_logits.npy')
print(sub1[0:submit.shape[0]].shape)
sub4 = np.load('tian_test5998_10folds.npy')
print(sub4.shape)
sub5 = np.load('ma_56_31.npy')
print(sub5.shape)
sub6 = np.load('ma_56_18.npy')
print(sub6.shape)

res1 = 0.4*sub1[0:submit.shape[0]] + 0.1*sub4 + 0.5*sub5

vote1 = np.load('my_gat_logits_vote.npy')
vote2 = np.load('56.18_vote.npy')
vote3 = np.load('tian_gat_logits_vote.npy')
res2 = vote1[0:submit.shape[0]] + vote2 + 0.5*vote3 # + props_to_onehot(res1)
# res2 = vote1[0:submit.shape[0]] + vote2 + vote3 + props_to_onehot(res1)
       # props_to_onehot(sub1)[0:submit.shape[0]] + props_to_onehot(sub4) + props_to_onehot(sub5)


count = []
for i in range(res2.shape[0]):
    temp = max(res2[i])
    num = sum(res2[i] == temp)
    if num > 1:
        count.append(i)
print(len(count))

feature_graph_edge = np.load('feature_graph_edge_finals.npy')
# res2 = 0.3*sub1[0:submit.shape[0]] + 0.3*sub2[0:submit.shape[0]] + 0.3*sub4 + 0.4*sub5
feature_graph_edge = feature_graph_edge[test_nid]
feature_graph_edge = labels[feature_graph_edge].numpy()
for id in count:
    discount = 10
    for neigh in feature_graph_edge[id]:
        if neigh > -1:
            res2[id][neigh] += 1/discount

count = []
for i in range(res2.shape[0]):
    temp = max(res2[i])
    num = sum(res2[i] == temp)
    if num > 1:
        count.append(i)
print(len(count))

for id in count:
    res2[id] = res1[id]
count = []
for i in range(res2.shape[0]):
    temp = max(res2[i])
    num = sum(res2[i] == temp)
    if num > 1:
        count.append(i)
print(len(count))

# node info
node_info = np.load('node_info.npy')
node_info = node_info[test_nid]
walk_label_features = np.load('walk_label_features.npy')
walk_label_features = walk_label_features[test_nid]


prediction = ["A","B","C","D","E","F","G",
              "H","I","J","K","L","M","N",
              "O","P","Q","R","S","T","U",
              "V","W"]
score_map = {}
for i in range(23):
    score_map[i] = prediction[i]


res2 = np.argmax(res2, axis=1)

submit["label"] = res2[0:submit.shape[0]]
submit["label"] = submit["label"].map(score_map)
print(submit.head())
submit.to_csv("E:/ZJL/DGL/submission_ronghe.csv", index=None)