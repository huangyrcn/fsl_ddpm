import json
import math
import random
import torch
import numpy as np
import pickle as pkl
import torch.nn as nn
import networkx as nx
import torch.nn.functional as F
from numpy.random import RandomState
from collections import defaultdict
from torch_geometric.utils import to_dense_adj, dense_to_sparse

class Dataset:
    def __init__(self, name, args):
        """
        数据集类，负责加载数据、划分训练/测试集、生成混合图等
        :param name: 数据集名称
        :param args: 参数对象，包含各种超参数
        """
        self.dataset_name = name
        self.args = args
        self.train_graphs = []  # 训练集图列表
        self.test_graphs = []   # 测试集图列表

        # 加载所有图、标签字典、节点标签集合
        all_graphs, label_dict, tagset = load_data(self.dataset_name, True)
        self.tagset = tagset

        # 从json文件读取训练/测试类别划分
        with open("datasets/{}/train_test_classes.json".format(args.dataset_name), "r") as f:
            all_class_splits = json.load(f)
            self.train_classes = all_class_splits["train"]  # 训练类别
            self.test_classes = all_class_splits["test"]    # 测试类别

        # 构建训练类别到索引的映射
        train_classes_mapping = {}
        for cl in self.train_classes:
            train_classes_mapping[cl] = len(train_classes_mapping)
        self.train_classes_num = len(train_classes_mapping)

        # 构建测试类别到索引的映射
        test_classes_mapping = {}
        for cl in self.test_classes:
            test_classes_mapping[cl] = len(test_classes_mapping)
        self.test_classes_num = len(test_classes_mapping)

        # 按类别划分训练/测试图
        for i in range(len(all_graphs)):
            if all_graphs[i].label in self.train_classes:
                self.train_graphs.append(all_graphs[i])
            if all_graphs[i].label in self.test_classes:
                self.test_graphs.append(all_graphs[i])

        # 将训练/测试图的标签映射为索引
        for graph in self.train_graphs:
            graph.label = train_classes_mapping[int(graph.label)]
        for i, graph in enumerate(self.test_graphs):
            graph.label = test_classes_mapping[int(graph.label)]

        # 打乱训练图顺序
        np.random.shuffle(self.train_graphs)

        # 构建训练任务字典，key为类别，value为该类别下所有图
        self.train_tasks = defaultdict(list)
        for graph in self.train_graphs:
            self.train_tasks[graph.label].append(graph)

        # 对训练图进行mixup生成新图
        generate_train_graphs = []
        fir, sec = np.random.randint(low=0, high=len(self.train_graphs), size=(2, self.args.gen_train_num))
        for i in range(self.args.gen_train_num):
            # 每64个样本重新采样一次mixup系数lam
            if i % 64 == 0:
                lam = np.random.beta(0.5, 0.5)
            lam = max(lam, 1 - lam)  # 保证lam>=0.5
            # 计算节点特征匹配矩阵
            match = self.train_graphs[fir[i]].node_features @ self.train_graphs[sec[i]].node_features.T
            normalized_match = F.softmax(match, dim=0)
            # 混合邻接矩阵
            mixed_adj = lam * to_dense_adj(self.train_graphs[fir[i]].edge_mat)[0].double() + \
                        (1 - lam) * normalized_match.double() @ \
                        to_dense_adj(self.train_graphs[sec[i]].edge_mat)[0].double() @ normalized_match.double().T
            mixed_adj[mixed_adj < 0.1] = 0  # 小于0.1的置零
            # 混合节点特征
            mixed_x = lam * self.train_graphs[fir[i]].node_features + \
                      (1 - lam) * normalized_match.float() @ self.train_graphs[sec[i]].node_features
            # 邻接矩阵转稀疏边
            edge_index, _ = dense_to_sparse(mixed_adj)
            edges = [(x, y) for x, y in zip(edge_index[0].tolist(), edge_index[1].tolist())]
            g = nx.Graph()
            g.add_edges_from(edges)
            g.add_nodes_from(list(range(edge_index.max() + 1)))
            G = Graph(g, -1)  # -1表示mixup生成的图
            G.edge_mat = edge_index
            G.node_features = mixed_x
            generate_train_graphs.append(G)
        print("generate yes")
        print("generate len is ", len(generate_train_graphs))
        print("before the number of train graphs is ", len(self.train_graphs))
        self.train_graphs.extend(generate_train_graphs)
        print("after the number of train graphs is ", len(self.train_graphs))

        # 打乱测试图顺序
        np.random.shuffle(self.test_graphs)

        # 构建测试任务字典，key为类别，value为该类别下所有图
        self.test_tasks = defaultdict(list)
        for graph in self.test_graphs:
            self.test_tasks[graph.label].append(graph)

        # 构建测试集微调用的支持集和查询集
        self.test_fine_tune_list = []  # 每个类别的K_shot支持集
        self.total_test_g_list = []    # 所有类别剩余的查询集
        for index in range(self.test_classes_num):
            self.test_fine_tune_list.append(list(self.test_tasks[index])[:self.args.K_shot])
            self.total_test_g_list.extend(list(self.test_tasks[index])[self.args.K_shot:])

        # 对测试集支持集做mixup生成新图
        self.generate_test_graphs = defaultdict(list)
        for index in range(self.test_classes_num):
            fir, sec = np.random.randint(low=0, high=len(self.test_fine_tune_list), size=(2, self.args.gen_test_num))
            for i in range(self.args.gen_test_num):
                lam = np.random.beta(0.5, 0.5)
                lam = max(lam, 1 - lam)
                # 计算节点特征匹配矩阵
                match = self.test_fine_tune_list[index][fir[i]].node_features @ \
                        self.test_fine_tune_list[index][sec[i]].node_features.T
                normalized_match = F.softmax(match, dim=0)
                # 混合邻接矩阵
                mixed_adj = lam * to_dense_adj(self.test_fine_tune_list[index][fir[i]].edge_mat)[0].double() + \
                            (1 - lam) * normalized_match.double() @ \
                            to_dense_adj(self.test_fine_tune_list[index][sec[i]].edge_mat)[0].double() @ \
                            normalized_match.double().T
                mixed_adj[mixed_adj < 0.1] = 0
                # 混合节点特征
                mixed_x = lam * self.test_fine_tune_list[index][fir[i]].node_features + \
                          (1 - lam) * normalized_match.float() @ self.test_fine_tune_list[index][sec[i]].node_features
                # 邻接矩阵转稀疏边
                edge_index, _ = dense_to_sparse(mixed_adj)
                edges = [(x, y) for x, y in zip(edge_index[0].tolist(), edge_index[1].tolist())]
                g = nx.Graph()
                g.add_edges_from(edges)
                g.add_nodes_from(list(range(edge_index.max() + 1)))
                G = Graph(g, -2)  # -2表示测试mixup生成的图
                G.edge_mat = edge_index
                G.node_features = mixed_x
                G.y_a = self.test_fine_tune_list[index][fir[i]].label  # mixup的第一个标签
                G.y_b = self.test_fine_tune_list[index][sec[i]].label  # mixup的第二个标签
                G.lam = lam
                self.generate_test_graphs[index].append(G)

        print("generate yes")
        # 打乱所有测试查询集
        rd = RandomState(0)
        rd.shuffle(self.total_test_g_list)

    def sample_one_task(self, task_source, class_index, K_shot, query_size, test_start_idx=None):
        """
        从任务源中采样一个N-way K-shot任务
        :param task_source: 任务源（如self.train_tasks或self.test_tasks）
        :param class_index: 选中的类别索引列表
        :param K_shot: 每类支持集样本数
        :param query_size: 每类查询集样本数
        :param test_start_idx: 测试时查询集起始索引
        :return: 支持集、查询集、补齐数
        """
        support_set = []
        query_set = []
        for index in class_index:
            g_list = list(task_source[index])
            if self.args.test_mixup:
                # 测试时支持集包含原始K_shot和mixup生成的图
                mid = g_list[:K_shot] + list(self.generate_test_graphs[index])
                support_set.append(mid)
            else:
                support_set.append(g_list[:K_shot])

        # 测试时从所有测试样本采样查询集
        append_count = 0
        if task_source == self.test_tasks and test_start_idx is not None:
            for i in range(len(class_index)):
                # 采样query_size个查询样本
                query_set.append(self.total_test_g_list[
                                 min(test_start_idx + i * query_size, len(self.total_test_g_list)):
                                 min(test_start_idx + (i + 1) * query_size, len(self.total_test_g_list))])
                # 若不足query_size则补齐
                while len(query_set[-1]) < query_size:
                    query_set[-1].append(query_set[0][-1])
                    append_count += 1

        return {'support_set': support_set, 'query_set': query_set, 'append_count': append_count}

class Graph(object):
    def __init__(self, g, label, node_tags=None, node_features=None):
        '''
        图对象，包含图结构、标签、节点标签、节点特征等
        :param g: networkx图对象
        :param label: 图标签（整数）
        :param node_tags: 节点标签列表
        :param node_features: 节点特征（one-hot或属性）
        '''
        self.label = label
        self.g = g
        self.node_tags = node_tags
        self.neighbors = []      # 邻居列表
        self.node_features = 0   # 节点特征
        self.edge_mat = 0        # 边矩阵
        self.y_a = -1            # mixup标签a
        self.y_b = -1            # mixup标签b
        self.lam = 0             # mixup系数
        self.max_neighbor = 0    # 最大邻居数

def load_data(dataset, degree_as_tag):
    '''
    加载数据集，支持多种格式
    :param dataset: 数据集名称
    :param degree_as_tag: 是否用度作为节点标签
    :return: 图列表、标签字典、节点标签集合
    '''
    print('loading data')
    
    if dataset in ['Letter_high', 'ENZYMES', 'Reddit', 'TRIANGLES']:
        g_list = []      # 图对象列表
        label_dict = {}  # 标签到索引的映射
        feat_dict = {}   # 节点标签到索引的映射

        # 读取txt格式数据集
        with open('datasets/%s/%s.txt' % (dataset, dataset), 'r') as f:
            n_g = int(f.readline().strip())  # 图数量
            for i in range(n_g):
                row = f.readline().strip().split()
                n, l = [int(w) for w in row]  # 节点数、标签
                if l not in label_dict:
                    mapped = len(label_dict)
                    label_dict[l] = mapped
                g = nx.Graph()
                node_tags = []
                node_features = []
                n_edges = 0
                for j in range(n):
                    g.add_node(j)
                    row = f.readline().strip().split()
                    tmp = int(row[1]) + 2
                    if tmp == len(row):
                        # 无节点属性
                        row = [int(w) for w in row]
                        attr = None
                    else:
                        row, attr = [int(w) for w in row[:tmp]], np.array([float(w) for w in row[tmp:]])
                    if row[0] not in feat_dict:
                        mapped = len(feat_dict)
                        feat_dict[row[0]] = mapped
                    node_tags.append(feat_dict[row[0]])
                    if tmp > len(row):
                        node_features.append(attr)
                    n_edges += row[1]
                    for k in range(2, len(row)):
                        g.add_edge(j, row[k])
                if node_features != []:
                    node_features = np.stack(node_features)
                    node_feature_flag = True
                else:
                    node_features = None
                    node_feature_flag = False
                assert len(g) == n
                g_list.append(Graph(g, l, node_tags))

        # 添加邻居、边矩阵等属性
        for g in g_list:
            g.neighbors = [[] for i in range(len(g.g))]
            for i, j in g.g.edges():
                g.neighbors[i].append(j)
                g.neighbors[j].append(i)
            degree_list = []
            for i in range(len(g.g)):
                g.neighbors[i] = g.neighbors[i]
                degree_list.append(len(g.neighbors[i]))
            g.max_neighbor = max(degree_list)
            edges = [list(pair) for pair in g.g.edges()]
            edges.extend([[i, j] for j, i in edges])
            deg_list = list(dict(g.g.degree(range(len(g.g)))).values())
            g.edge_mat = torch.LongTensor(edges).transpose(0, 1)

        # 若用度作为节点标签
        if degree_as_tag:
            for g in g_list:
                g.node_tags = list(dict(g.g.degree).values())
                if np.sum(np.array(g.node_tags) == 0): print(g.node_tags)

        # 提取所有唯一节点标签
        tagset = set([])
        for g in g_list:
            tagset = tagset.union(set(g.node_tags))
        tagset = list(tagset)
        tag2index = {tagset[i]: i for i in range(len(tagset))}
        # 构建one-hot节点特征
        for g in g_list:
            g.node_features = torch.zeros(len(g.node_tags), len(tagset))
            g.node_features[range(len(g.node_tags)), [tag2index[tag] for tag in g.node_tags]] = 1

        print('# classes: %d' % len(label_dict))
        print('# maximum node tag: %d' % len(tagset))
        print("# data: %d" % len(g_list))

        return g_list, label_dict, tagset

    elif dataset in ['R52', 'COIL']:
        # 读取pickle格式数据集
        print(dataset)
        node_attribures = pkl.load(open('datasets/{}/{}_node_attributes.pickle'.format(dataset, dataset), 'rb'))
        train_set = pkl.load(open('datasets/{}/{}_train_set.pickle'.format(dataset, dataset), 'rb'))
        val_set = pkl.load(open('datasets/{}/{}_val_set.pickle'.format(dataset, dataset), 'rb'))
        test_set = pkl.load(open('datasets/{}/{}_test_set.pickle'.format(dataset, dataset), 'rb'))

        g_list = []
        for sets in [train_set, val_set, test_set]:
            graph2nodes = sets["graph2nodes"]
            graph2edges = sets['graph2edges']
            label2graphs = sets['label2graphs']
            for label, graphs in label2graphs.items():
                for graph_id in graphs:
                    g = nx.Graph()
                    node_mapping = {}
                    for node in graph2nodes[graph_id]:
                        node_mapping[node] = len(node_mapping)
                        g.add_node(node_mapping[node])
                    for edge in graph2edges[graph_id]:
                        g.add_edge(node_mapping[edge[0]], node_mapping[edge[1]])
                    g = Graph(g, label)
                    g.neighbors = [[] for i in range(len(g.g))]
                    for i, j in g.g.edges():
                        g.neighbors[i].append(j)
                        g.neighbors[j].append(i)
                    degree_list = []
                    for i in range(len(g.g)):
                        g.neighbors[i] = g.neighbors[i]
                        degree_list.append(len(g.neighbors[i]))
                    g.max_neighbor = max(degree_list)
                    edges = [list(pair) for pair in g.g.edges()]
                    edges.extend([[i, j] for j, i in edges])
                    deg_list = list(dict(g.g.degree(range(len(g.g)))).values())
                    g.edge_mat = torch.LongTensor(edges).transpose(0, 1)
                    g.node_features = torch.FloatTensor(node_attribures[graph2nodes[graph_id]])
                    if dataset == 'R52':
                        g.node_features = g.node_features.unsqueeze(-1)
                    g_list.append(g)

        print("# data: %d" % len(g_list))
        return g_list, None, None

    elif dataset == 'ogbg-ppa':
        # 读取OGB格式数据集
        dataset = GraphPropPredDataset(name=dataset)
        split_idx = dataset.get_idx_split()
        train_idx, valid_idx, test_idx = split_idx["train"], split_idx["valid"], split_idx["test"]
        g_list = []
        # 遍历所有图
        for i in range(len(dataset)):
            graph, label = dataset[i]  # 获取图和标签
            nx_graph = nx.Graph()
            for j in range(graph['num_nodes']):
                nx_graph.add_node(j)
            for j in range(graph['edge_index'].shape[1]):
                nx_graph.add_edge(graph['edge_index'][j, 0], graph['edge_index'][j, 1])
            g = Graph(nx_graph, label)
            g.edge_mat = torch.LongTensor(graph['edge_index'])
            g.node_features = torch.FloatTensor(graph['node_feat'])
            g_list.append(g)
            tagset = [i for i in range(37)]
        return g_list, {i: i for i in range(37)}, tagset
