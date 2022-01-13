import torch

import numpy as np
import pandas as pd
from utils_dgl import load_dgl_graph, time_diff



BASE_PATH = 'Z:/DataScience/DGL'
device_id = torch.device("cuda")
graph, labels, train_nid, val_nid, test_nid, node_feat = load_dgl_graph(BASE_PATH)

start = 939963 + 104454

res1 = torch.load('Z:/DataScience/DGL/ SAGN_with_SLE/intermediate_outputs/ogbn-papers100M/sagn/fold_0.pt').cpu().cuda()
print(res1.shape)
res1 = res1[-592391:]
print(sum(res1[0]))
res1 = res1.cpu().numpy()

res2 = torch.load('Z:/DataScience/DGL/ SAGN_with_SLE/intermediate_outputs/ogbn-papers100M/sagn/fold_1.pt').cpu().cuda()
print(res2.shape)
res2 = res2[-592391:]
print(sum(res2[0]))
res2 = res2.cpu().numpy()

res3 = torch.load('Z:/DataScience/DGL/ SAGN_with_SLE/intermediate_outputs/ogbn-papers100M/sagn/fold_2.pt').cpu().cuda()
print(res3.shape)
res3 = res3[-592391:]
print(sum(res3[0]))
res3 = res3.cpu().numpy()

res4 = torch.load('Z:/DataScience/DGL/ SAGN_with_SLE/intermediate_outputs/ogbn-papers100M/sagn/fold_3.pt').cpu().cuda()
print(res4.shape)
res4 = res4[-592391:]
print(sum(res4[0]))
res4 = res4.cpu().numpy()

res5 = torch.load('Z:/DataScience/DGL/ SAGN_with_SLE/intermediate_outputs/ogbn-papers100M/sagn/fold_4.pt').cpu().cuda()
print(res5.shape)
res5 = res5[-592391:]
print(sum(res5[0]))
res5 = res5.cpu().numpy()

res6 = torch.load('Z:/DataScience/DGL/ SAGN_with_SLE/intermediate_outputs/ogbn-papers100M/sagn/fold_5.pt').cpu().cuda()
print(res6.shape)
res6 = res6[-592391:]
print(sum(res6[0]))
res6 = res6.cpu().numpy()

res7 = torch.load('Z:/DataScience/DGL/ SAGN_with_SLE/intermediate_outputs/ogbn-papers100M/sagn/fold_6.pt').cpu().cuda()
print(res7.shape)
res7 = res7[-592391:]
print(sum(res7[0]))
res7 = res7.cpu().numpy()

res8 = torch.load('Z:/DataScience/DGL/ SAGN_with_SLE/intermediate_outputs/ogbn-papers100M/sagn/fold_7.pt').cpu().cuda()
print(res7.shape)
res8 = res8[-592391:]
print(sum(res8[0]))
res8 = res8.cpu().numpy()

res9 = torch.load('Z:/DataScience/DGL/ SAGN_with_SLE/intermediate_outputs/ogbn-papers100M/sagn/fold_8.pt').cpu().cuda()
print(res9.shape)
res9 = res9[-592391:]
print(sum(res9[0]))
res9 = res9.cpu().numpy()

res0 = torch.load('Z:/DataScience/DGL/ SAGN_with_SLE/intermediate_outputs/ogbn-papers100M/sagn/fold_9.pt').cpu().cuda()
print(res0.shape)
res0 = res0[-592391:]
print(sum(res0[0]))
res0 = res0.cpu().numpy()

res = res1 + res2 + res3 + res4 + res5 + res6 + res7 + res8 +res9 +res0
# res = res1

result = np.argmax(res, axis=1)


prediction = ["A","B","C","D","E","F","G",
              "H","I","J","K","L","M","N",
              "O","P","Q","R","S","T","U",
              "V","W"]
score_map = {}
for i in range(23):
    score_map[i] = prediction[i]


submit = pd.read_csv("Z:/DataScience/DGL/sample_submission_for_validation.csv")
submit["label"] = result[0:submit.shape[0]]
submit["label"] = submit["label"].map(score_map)
print(submit.head())
submit.to_csv("Z:/DataScience/DGL/submission_sagn_1106.csv", index=None)
