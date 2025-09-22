import torch
import numpy as np
import networkx as nx
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch.utils.data import Dataset
from typing import List, Optional, Tuple
import json
from collections import defaultdict
import pickle as pkl



def load_data(dataset: str, degree_as_tag: bool = True) -> Tuple[List[Data], dict, List]:
    """
    完全按照原始实现的方式加载数据，确保完全一致
    """
    print(f"Loading custom data: {dataset}")
    
    # R52/COIL 使用与 dataset.py 一致的 pickle 数据格式，这里直接转为 PyG Data 返回
    if dataset in ['R52', 'COIL']:
        print(dataset)
        node_attribures = pkl.load(open(f'datasets/{dataset}/{dataset}_node_attributes.pickle', 'rb'))
        train_set = pkl.load(open(f'datasets/{dataset}/{dataset}_train_set.pickle', 'rb'))
        val_set = pkl.load(open(f'datasets/{dataset}/{dataset}_val_set.pickle', 'rb'))
        test_set = pkl.load(open(f'datasets/{dataset}/{dataset}_test_set.pickle', 'rb'))

        data_list = []
        for sets in [train_set, val_set, test_set]:
            graph2nodes = sets["graph2nodes"]
            graph2edges = sets['graph2edges']
            label2graphs = sets['label2graphs']
            for label, graphs in label2graphs.items():
                for graph_id in graphs:
                    # 重建节点映射
                    node_mapping = {}
                    for node in graph2nodes[graph_id]:

                        node_mapping[node] = len(node_mapping)

                    # 构造边并进行无向规范化、去重、再对称化
                    edges = []
                    for u, v in graph2edges[graph_id]:
                        edges.append([node_mapping[u], node_mapping[v]])
                    edge_index = torch.tensor(edges, dtype=torch.long).t() if edges else torch.empty((2, 0), dtype=torch.long)
                    if edge_index.numel() > 0:
                        u, v = edge_index
                        lo = torch.minimum(u, v)
                        hi = torch.maximum(u, v)
                        undirected = torch.stack([lo, hi], dim=0)
                        undirected = torch.unique(undirected.t(), dim=0).t()
                        edge_index = torch.cat([undirected, undirected.flip(0)], dim=1)

                    # 节点特征
                    x = torch.FloatTensor(node_attribures[graph2nodes[graph_id]])
                    if dataset == 'R52':
                        x = x.unsqueeze(-1)

                    y = torch.tensor(label, dtype=torch.long)

                    data = Data(x=x, edge_index=edge_index, y=y)
                    data.label = int(label)
                    data_list.append(data)

        print(f"# data: {len(data_list)}\n")
        return data_list, None, None
    
    data_list = []
    label_dict = {}
    
    with open(f"./datasets/{dataset}/{dataset}.txt", "r") as f:
        n_g = int(f.readline().strip())
        
        for i in range(n_g):
            # 读取图信息
            row = f.readline().strip().split()
            n, graph_label = int(row[0]), int(row[1])
            
            # 更新标签字典（与原始实现一致）
            if graph_label not in label_dict:
                mapped = len(label_dict)
                label_dict[graph_label] = mapped
            
            # 完全按照原始实现构建NetworkX图
            g = nx.Graph()
            node_tags = []
            feat_dict = {}
            
            for j in range(n):
                g.add_node(j)
                row = f.readline().strip().split()
                tmp = int(row[1]) + 2
                if tmp == len(row):
                    row = [int(w) for w in row]
                else:
                    row = [int(w) for w in row[:tmp]]
                
                # 节点标签处理（与原始实现一致）
                if not row[0] in feat_dict:
                    mapped = len(feat_dict)
                    feat_dict[row[0]] = mapped
                node_tags.append(feat_dict[row[0]])
                
                # 按原始顺序添加边
                for k in range(2, len(row)):
                    g.add_edge(j, row[k])
            
            # 如果使用度作为标签，覆盖节点标签（与原始实现一致）
            if degree_as_tag:
                node_tags = list(dict(g.degree).values())
            
            # 构建PyG格式的边索引
            edges = [list(pair) for pair in g.edges()]
            edges.extend([[i, j] for j, i in edges])
            if edges:
                edge_index = torch.tensor(edges, dtype=torch.long).t()
            else:
                edge_index = torch.empty((2, 0), dtype=torch.long)
            
            # 存储数据
            raw_data = {
                'edge_index': edge_index,
                'node_tags': node_tags,
                'graph_label': graph_label,
                'n_nodes': n
            }
            data_list.append(raw_data)
    
    # 收集所有标签构建tagset（与原始实现一致）
    global_tag_set = set()
    for raw_data in data_list:
        global_tag_set.update(raw_data['node_tags'])
    
    tagset = sorted(global_tag_set)
    tag2index = {tag: i for i, tag in enumerate(tagset)}
    
    # 最终转换为PyG Data对象
    final_data_list = []
    for raw_data in data_list:
        # 构建one-hot特征
        x = torch.zeros(raw_data['n_nodes'], len(tagset))
        for i, tag in enumerate(raw_data['node_tags']):
            x[i, tag2index[tag]] = 1
        
        # 创建最终的Data对象
        data = Data(
            x=x,
            edge_index=raw_data['edge_index'],
            y=torch.tensor(raw_data['graph_label'], dtype=torch.long)
        )
        data.label = raw_data['graph_label']
        final_data_list.append(data)
    
    print(f"# classes: {len(label_dict)}")
    print(f"# maximum node tag: {len(tagset)}")
    print(f"# data: {len(final_data_list)}\n")
    
    return final_data_list, label_dict, tagset
    

class CustomDataset:
    """
    完全基于PyG Data的数据集类：
    - 从加载开始就使用PyG Data格式
    - 在init中预先划分支持集和查询集池
    - sample_one_task只需要任务ID即可获取数据
    """

    def __init__(self, name, args):
        self.dataset_name = name
        self.args = args
        self.train_graphs = []
        self.test_graphs = []

        # 直接加载为PyG Data对象
        all_graphs, label_dict, tagset = load_data(self.dataset_name, True)
        self.tagset = tagset
        self.label_dict = label_dict
        # 设置特征维度（节点特征的维度），兼容 R52/COIL（tagset 为 None）
        self.feature_dim = (len(tagset) if tagset is not None else (all_graphs[0].x.size(1) if all_graphs else 0))

        # 加载类别划分
        with open("datasets/{}/train_test_classes.json".format(args.dataset_name), "r") as f:
            all_class_splits = json.load(f)

        # 获取训练和测试类别
        train_classes = all_class_splits['train']
        test_classes = all_class_splits['test']

        # 图按类别划分为训练图和测试图（已经是PyG Data格式）
        # 同时重新映射标签，与dataset.py保持一致
        train_classes_mapping = {}
        for cl in train_classes:
            train_classes_mapping[cl] = len(train_classes_mapping)
        
        test_classes_mapping = {}
        for cl in test_classes:
            test_classes_mapping[cl] = len(test_classes_mapping)
            
        for data in all_graphs:
            if data.label in train_classes:
                # 重新映射训练集标签
                data.label = train_classes_mapping[data.label]
                data.y = torch.tensor(data.label, dtype=torch.long)  # 同步更新y
                self.train_graphs.append(data)
            elif data.label in test_classes:
                # 重新映射测试集标签  
                data.label = test_classes_mapping[data.label]
                data.y = torch.tensor(data.label, dtype=torch.long)  # 同步更新y
                self.test_graphs.append(data)

        print(f"# train graphs: {len(self.train_graphs)}, # test graphs: {len(self.test_graphs)}")

        # 添加兼容性属性
        self.train_classes_num = len(train_classes_mapping)
        self.test_classes_num = len(test_classes_mapping)
        self.train_classes = train_classes
        self.test_classes = test_classes
        
        # === 添加与dataset.py一致的shuffle逻辑 ===
        np.random.seed(args.seed)
        np.random.shuffle(self.train_graphs)
        
        np.random.seed(args.seed)
        np.random.shuffle(self.test_graphs)

        # 构建task列表（兼容性接口）
        self.train_tasks = defaultdict(list)
        for data in self.train_graphs:
            self.train_tasks[data.label].append(data)
            
        # 测试类别构建
        self.test_tasks = defaultdict(list)
        for data in self.test_graphs:
            self.test_tasks[data.label].append(data)

        # === 构建nway的全局查询集池 ===
        self.test_fine_tune_list = []
        self.total_test_g_list = []
        for index in range(self.args.N_way):
            # 每个类别的前K_shot个作为支持集候选
            self.test_fine_tune_list.append(list(self.test_tasks[index])[:args.K_shot])
            # 剩余的加入全局查询集池
            self.total_test_g_list.extend(list(self.test_tasks[index])[args.K_shot:])
        
        # nway的 shuffle
        np.random.seed(args.seed)
        np.random.shuffle(self.total_test_g_list)
        self.test_task_Num=len(self.total_test_g_list) // (self.args.N_way * self.args.query_size)

    def aug(self,
            aug1_type: str,
            aug2_type: str,
            batch_size: int = 32,
            shuffle: bool = True,
            num_workers: int = 0,
            pin_memory: bool = False,
            seed: int = None,
            aug_ratio: float = 0.1):
        """
        返回两个对齐的 DataLoader，用于对比学习的 1:1 增强视图。
        - 视图1使用 aug1_type
        - 视图2使用 aug2_type
        两个 DataLoader 使用相同的打乱顺序，保证样本一一对应。
        """
        # 延迟导入，避免循环依赖
        from aug import get_optimized_augmentation

        # 在 CPU 上进行增强，避免与原始 Data 的设备不一致
        aug1 = get_optimized_augmentation(aug_type=aug1_type, aug_ratio=aug_ratio, device='cpu', seed=(seed if seed is not None else self.args.seed))
        aug2 = get_optimized_augmentation(aug_type=aug2_type, aug_ratio=aug_ratio, device='cpu', seed=(seed if seed is not None else self.args.seed))

        # 先对 train_graphs 进行一次全局打乱，随后基于该固定顺序做两种增强
        if shuffle:
            s = int(seed if seed is not None else self.args.seed)
            gperm = torch.Generator(); gperm.manual_seed(s)
            perm = torch.randperm(len(self.train_graphs), generator=gperm).tolist()
        else:
            perm = list(range(len(self.train_graphs)))

        # 基于 list 的增强：先生成两个对齐的增强列表，再构建 DataLoader
        aug_list1 = []
        aug_list2 = []
        for i in perm:
            d = self.train_graphs[i]
            aug_list1.append(aug1(d))
            aug_list2.append(aug2(d))

        # DataLoader 不再额外 shuffle，保障两个视图严格对齐
        loader_kwargs = dict(batch_size=batch_size,
                              shuffle=False,
                              num_workers=num_workers,
                              pin_memory=pin_memory)

        loader1 = DataLoader(aug_list1, **loader_kwargs)
        loader2 = DataLoader(aug_list2, **loader_kwargs)

        return loader1, loader2

    def sample_one_task(self, task_id):
        # 构建支持集：使用sum展平
        support_graphs = sum(self.test_fine_tune_list, [])
        
        # 从全局查询集池中按task_id 顺序取样本
        total_query_needed = self.args.N_way * self.args.query_size
        query_start_idx = task_id * total_query_needed
        
        query_graphs = []
        for i in range(total_query_needed):
            query_idx = (query_start_idx + i) % len(self.total_test_g_list)
            query_graphs.append(self.total_test_g_list[query_idx])
        
        return support_graphs, query_graphs

    def get_encoder_trainloader(self, batch_size: int = 64, shuffle: bool = True, num_workers: int = 0, pin_memory: bool = False, seed: int = None) -> DataLoader:
        """获取编码器训练的PyG DataLoader"""
        # 🔧 修复可重现性：为 DataLoader 设置固定的随机数生成器
        generator = None
        if shuffle and seed is not None:
            generator = torch.Generator()
            generator.manual_seed(seed)
        
        return DataLoader(
            dataset=self.train_graphs,
            batch_size=batch_size, 
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            generator=generator  # 🎯 关键修复：固定 shuffle 的随机性
        )

    def get_ldm_inference_batches(self, batch_size: int = 64, num_workers: int = 0) -> DataLoader:
        """获取LDM推理用的PyG DataLoader"""
        return DataLoader(
            dataset=self.train_graphs,
            batch_size=batch_size,
            shuffle=False,  # 推理不需要shuffle
            num_workers=num_workers,
            pin_memory=False
        )


if __name__ == '__main__':
    # 最小自检：加载 R52 与 COIL，打印训练/测试图数量与类别数
    class _Args:
        def __init__(self, dataset_name: str):
            self.dataset_name = dataset_name
            self.seed = 42
            self.N_way = 3
            self.K_shot = 5
            self.query_size = 10

    for ds in ['ENZYMES', 'Letter_high', 'TRIANGLES','Reddit', 'R52', 'COIL']:
        try:
            print(f"\n=== Sanity Check: {ds} ===")
            args = _Args(ds)
            dset = CustomDataset(ds, args)
            print(f"train_graphs: {len(dset.train_graphs)} | test_graphs: {len(dset.test_graphs)}")
            print(f"train_classes_num: {dset.train_classes_num} | test_classes_num: {dset.test_classes_num}")
            # 采样一个测试任务
            support, query = dset.sample_one_task(task_id=0)
            print(f"sampled support: {len(support)} | query: {len(query)}")
        except Exception as e:
            print(f"[ERROR] {ds}: {e}")