# **2021 MAXP** **基于DGL的图机器学习任务**

### 赛题背景

近几年来，针对图结构化数据的机器学习算法发展得如火如荼，其中图神经网络作为最新的图机器学习研究方向获得了广泛的关注，相关的论文也成为主流人工智能会议的热点方向。在现实场景里，图神经网络在计算机视觉、自然语言处理、生物制药、知识图谱、推荐系统等多个领域得到了应用，并取得了良好的表现。

### 任务说明

- 赛题的任务是进行点性质预测，即预测节点(论文)所属的类别。
- 本次比赛使用的图数据是基于微软学术文献生成的论文关系图，其中的节点是论文，边是论文间的引用关系。整个图包括约150万个节点，2000万条边。节点包含300维的特征，来自论文的标题和摘要等内容。节点属于约50个类别。

# 参考文献

Scalable and Adaptive Graph Neural Networks with Self-Label-Enhanced training

# 特征工程

1. 节点特征统计特征 node info
2. 随机游走统计特征 walk_label_features
3. node2vec特征

特征工程1,2的想法来源于队友Arthur Morgan，感谢！

# 运行指令

python sagn.py

python CorrectAndSmooth.py

注：没有采用SLE的方式进行训练

# 线上成绩

55.9~

**PS：时间原因代码未整理，不过代码无bug可直接运行，有帮助记得给个star~**

Remark：除了特征工程带来上分，一个重要的上分点在于label use的方式，通过切换masked节点带来上分