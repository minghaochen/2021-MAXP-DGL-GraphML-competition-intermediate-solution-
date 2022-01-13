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


# sub1 = torch.load('gat_logits.pt').cpu().numpy()
# print(sum(sub1[0,:]))


sub1 = np.load('my_gat_logits_finals.npy')
print(sub1[0:submit.shape[0]].shape)

# for fold in range(1,57):
#     val_batch_list = torch.load(
#         f'C:/Users/11732031/Desktop/maxp_baseline_model-main/ SAGN_with_SLE/intermediate_outputs/ogbn-papers100M/sagn/{fold}.pt').cpu().numpy()
#     if fold == 1:
#         res2 = val_batch_list
#     else:
#         res2 += val_batch_list
#
# res_origin = res2.copy()
# sub2 = res_origin[test_nid]
#
# sub2 = torch.load('sagn_logits_1.pt').cpu().numpy()
# sub2 = F.softmax(torch.tensor(sub2), dim=1).numpy()
# print(sum(sub2[1,:]))
# print(sub2[0:submit.shape[0]].shape)
# sub3 = np.load('gat_iso.npy')
# print(sub3[0:submit.shape[0]].shape)

sub4 = np.load('tian_test5998_10folds.npy')
print(sub4.shape)
sub5 = np.load('ma_56_31.npy')
print(sub5.shape)

sub6 = np.load('ma_56_18.npy')
print(sub6.shape)



# res2 = 0.3*sub1 + 0.3*sub2[-592391:] + 0.4*sub3[-592391:]

# res2 = 0.3*sub1[0:submit.shape[0]] + 0.3*sub2[0:submit.shape[0]] + 0.3*sub4 + 0.4*sub5

res2 = 0.4*sub1[0:submit.shape[0]] + 0.0*sub4 + 0.6*sub5

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