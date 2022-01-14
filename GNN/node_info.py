import pandas as pd
import numpy as np
from sklearn.decomposition import TruncatedSVD, PCA
from sklearn.model_selection import train_test_split
import os
import dgl
import pickle
import numpy as np
import torch as th
pd.set_option('display.max_columns', None)

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

    labels = th.from_numpy(label_data['label'])
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
    node_feat = th.from_numpy(features).float()
    print('################ Feature info: ###############')
    print('Node\'s feature shape:{}'.format(node_feat.shape))

    return graph, labels, tr_label_idx, val_label_idx, test_label_idx, node_feat


graph, labels, tr_label_idx, val_label_idx, test_label_idx, node_feat = load_dgl_graph('Z:/DataScience/DGL')

df_labels = pd.DataFrame()
df_graph = pd.DataFrame()
df_graph['src'] = graph.edges()[0].numpy()
df_graph['dst'] = graph.edges()[1].numpy()
df_labels['labels'] = labels
df_labels = df_labels.reset_index().rename(columns={'index':'dst'})


df_graph = df_graph.merge(df_labels, on='dst')
df_graph.rename(columns={'labels': 'dst_labels'}, inplace=True)
df_graph = df_graph.merge(df_labels, left_on='src', right_on='dst').drop(columns = ['dst_y'])
df_graph.rename(columns={'dst_x': 'dst', 'labels': 'src_labels'}, inplace=True)
# src dst dst_labels src_labels


df_labels_src = df_graph.groupby(['src'])['dst_labels'].value_counts().unstack()
df_labels_dst = df_graph.groupby(['dst']).src_labels.value_counts().unstack()
df_labels_src = df_labels_src.fillna(0)
df_labels_src['sum_in_neibor'] = df_labels_src.sum(axis=1)
df_labels_dst = df_labels_dst.fillna(0)
df_labels_dst['sum_in_neibor'] = df_labels_dst.sum(axis=1)


df_labels_dst = df_labels.merge(df_labels_dst, on='dst', how='outer').fillna(0)
df_labels_src = df_labels.rename(columns={'dst':'src'}).merge(df_labels_src, on='src', how='outer').fillna(0)
df_labels_src_dst = df_labels_src.join(df_labels_dst, lsuffix='_src', rsuffix='_dst')
df_labels_src_dst = df_labels_src_dst.drop(columns=['labels_dst', 'dst'])


node_info = df_labels_src_dst.iloc[:, 2:].values
np.save('node_info', node_info)
