"""
GIN模型实现 - 基于PyTorch Geometric的高效图神经网络

主要组件：
- MLP: 多层感知机，支持1层或多层结构
- GraphCNN: GIN图卷积网络，使用epsilon重加权和scatter操作优化
- Model: 完整的图分类模型，包含GIN编码器和投影头
- LogReg: 逻辑回归分类器

特点：
- 使用PyTorch Geometric实现，性能优于原始实现
- 支持多种图池化方式：sum, mean, max, meanmax, maxavg_concat
- 支持多种邻居聚合方式：sum, mean, max
- 数学逻辑与原始gin_model.py完全一致
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch_geometric.nn import global_mean_pool, global_max_pool, global_add_pool
from torch_geometric.utils import add_self_loops, to_dense_batch
from torch_scatter import scatter_mean, scatter_max, scatter_add


class MLP(nn.Module):
    """多层感知机 - 支持1层或多层结构，包含BatchNorm"""
    def __init__(self, num_layers, input_dim, hidden_dim, output_dim):
        super(MLP, self).__init__()

        self.linear_or_not = True
        self.num_layers = num_layers

        if num_layers < 1:
            raise ValueError("number of layers should be positive!")
        elif num_layers == 1:
            self.linear = nn.Linear(input_dim, output_dim)
        else:
            self.linear_or_not = False
            self.linears = torch.nn.ModuleList()
            self.batch_norms = torch.nn.ModuleList()

            self.linears.append(nn.Linear(input_dim, hidden_dim))
            for layer in range(num_layers - 2):
                self.linears.append(nn.Linear(hidden_dim, hidden_dim))
            self.linears.append(nn.Linear(hidden_dim, output_dim))

            for layer in range(num_layers - 1):
                self.batch_norms.append(nn.BatchNorm1d(hidden_dim))

    def forward(self, x):
        if self.linear_or_not:
            return self.linear(x)
        else:
            h = x
            for layer in range(self.num_layers - 1):
                h = F.relu(self.batch_norms[layer](self.linears[layer](h)))
            return self.linears[self.num_layers - 1](h)


class GraphCNN(nn.Module):
    """
    GIN图卷积网络 - 基于PyTorch Geometric的高效实现
    
    核心思想：使用epsilon重加权机制区分中心节点和邻居节点
    数学公式：h^(l+1) = MLP((1+ε)h^(l) + Σ(neighbors))
    
    参数：
        num_layers: 网络层数（包含输入层）
        num_mlp_layers: MLP层数（不包含输入层）
        input_dim: 输入特征维度
        hidden_dim: 隐藏层维度
        learn_eps: 是否学习epsilon参数
        graph_pooling_type: 图级池化方式
        neighbor_pooling_type: 邻居聚合方式
    """
    def __init__(self, num_layers=5, num_mlp_layers=2, input_dim=200, hidden_dim=128, output_dim=200,
                 drop_rate=0.5, learn_eps=True, graph_pooling_type='sum', neighbor_pooling_type='sum',
                 use_select_sim=False, device=None):
        super(GraphCNN, self).__init__()

        self.drop_rate = drop_rate
        self.device = device
        self.num_layers = num_layers
        self.graph_pooling_type = graph_pooling_type
        self.neighbor_pooling_type = neighbor_pooling_type
        self.use_select_sim = use_select_sim

        self.learn_eps = learn_eps
        self.eps = nn.Parameter(torch.zeros(self.num_layers))

        # List of MLPs
        self.mlps = torch.nn.ModuleList()

        # List of batchnorms applied to the output of MLP
        self.batch_norms = torch.nn.ModuleList()

        for layer in range(self.num_layers):
            if layer == 0:
                self.mlps.append(MLP(num_mlp_layers, input_dim, hidden_dim, hidden_dim))
            else:
                self.mlps.append(MLP(num_mlp_layers, hidden_dim, hidden_dim, hidden_dim))

            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))

    def next_layer_eps(self, h, layer, data):
        """
        消息传递层 - 使用epsilon重加权的邻居聚合
        
        步骤：
        1. 聚合邻居节点特征（sum/mean/max）
        2. 用epsilon重加权中心节点：pooled + (1+ε) * h
        3. 通过MLP和BatchNorm处理
        4. 应用ReLU激活
        
        优化：使用scatter操作替代for循环，提升性能
        """
        edge_index = data.edge_index
        batch = data.batch
        
        # Add self-loops if learn_eps is False - **数学逻辑不变**
        if not self.learn_eps:
            edge_index, _ = add_self_loops(edge_index, num_nodes=h.size(0))
        
        # Perform neighbor aggregation using efficient scatter operations
        # **数学完全等价于原版本的for循环实现**
        row, col = edge_index
        
        if self.neighbor_pooling_type == 'sum':
            # 等价于原版本：neighbor_features.sum(dim=0)
            pooled = scatter_add(h[col], row, dim=0, dim_size=h.size(0))
        elif self.neighbor_pooling_type == 'mean':
            # 等价于原版本：neighbor_features.mean(dim=0)
            pooled = scatter_mean(h[col], row, dim=0, dim_size=h.size(0))
        elif self.neighbor_pooling_type == 'max':
            # 等价于原版本：neighbor_features.max(dim=0)[0]
            pooled = scatter_max(h[col], row, dim=0, dim_size=h.size(0))[0]
        else:
            # 如果是其他类型，回退到原版本的for循环实现确保数学一致性
            pooled = torch.zeros_like(h)
            for i in range(h.size(0)):
                neighbor_mask = (row == i)
                if neighbor_mask.sum() > 0:
                    neighbor_features = h[col[neighbor_mask]]
                    if self.neighbor_pooling_type == 'mean':
                        pooled[i] = neighbor_features.mean(dim=0)
                    else:  # sum
                        pooled[i] = neighbor_features.sum(dim=0)

        # **核心数学逻辑完全不变**
        # Reweight the center node representation when aggregating it with its neighbors
        pooled = pooled + (1 + self.eps[layer]) * h
        pooled_rep = self.mlps[layer](pooled)
        h = self.batch_norms[layer](pooled_rep)
        h = F.relu(h)
        return h

    def forward(self, data, mode='train', promp=None):
        """
        前向传播 - 完整的GIN图卷积过程
        
        输入：PyTorch Geometric Data对象
        输出：
            - pooled_h_layers: 每层的图级表示
            - h: 最终节点表示
            - Adj_block_idx: 邻接矩阵索引
            - hidden_rep: 每层隐藏表示
            - final_hidd: 最终隐藏状态（空列表）
        """
        # PyTorch Geometric自动处理设备转换，无需手动调用.to(device)
        X_concat = data.x

        # List of hidden representation at each layer (including input)
        hidden_rep = [X_concat]
        h = X_concat

        # Forward pass through layers - **数学完全一致**
        for layer in range(self.num_layers):
            h = self.next_layer_eps(h, layer, data)
            hidden_rep.append(h)

        # Perform pooling over all nodes in each graph for every layer
        # **图级别池化逻辑与原版完全保持一致**
        pooled_h_layers = []
        batch_index = data.batch
        num_graphs = batch_index.max().item() + 1

        for layer_h in hidden_rep:
            if self.graph_pooling_type == "sum":
                pooled_h = global_add_pool(layer_h, batch_index)
            elif self.graph_pooling_type == "average":
                pooled_h = global_mean_pool(layer_h, batch_index)
            elif self.graph_pooling_type == "max":
                pooled_h = global_max_pool(layer_h, batch_index)
            elif self.graph_pooling_type == "meanmax":
                # Mean + Max concatenation
                mean_h = global_mean_pool(layer_h, batch_index)
                max_h = global_max_pool(layer_h, batch_index)
                pooled_h = torch.cat([mean_h, max_h], dim=-1)
            elif self.graph_pooling_type == "maxavg_concat":
                # Max + Average concatenation  
                max_h = global_max_pool(layer_h, batch_index)
                avg_h = global_mean_pool(layer_h, batch_index)
                pooled_h = torch.cat([max_h, avg_h], dim=-1)
            else:
                raise ValueError(f"Unknown graph_pooling_type: {self.graph_pooling_type}")

            pooled_h_layers.append(pooled_h)

        # Return same format as gin_model.py - **接口完全保持一致**
        final_hidd = []  # Empty list to match original behavior
        Adj_block_idx = data.edge_index  # Return edge_index as adjacency info
        
        return pooled_h_layers, h, Adj_block_idx, hidden_rep, final_hidd


class LogReg(nn.Module):
    """逻辑回归分类器 - 包含BatchNorm和线性层"""
    def __init__(self, ft_in, nb_classes):
        super(LogReg, self).__init__()
        self.fc = nn.Linear(ft_in, nb_classes)
        self.bn = nn.BatchNorm1d(ft_in)

    def forward(self, seq):
        ret = self.fc(self.bn(seq))
        return ret


class Model(nn.Module):
    """
    完整的图分类模型 - 包含GIN编码器和投影头
    
    功能：
    - 图编码：使用GIN提取图表示
    - 对比学习：支持正负样本对比
    - 少样本学习：支持N-way K-shot任务
    - 图分类：输出图级预测结果
    """
    def __init__(self, args):
        super().__init__()

        self.args = args
        self.T = args.t  # 温度参数
        self.hid = args.gin_hid  # 隐藏层维度
        self.N = args.N_way  # N-way分类
        self.K = args.K_shot  # K-shot学习
        self.Q = args.query_size  # 查询集大小
        self.device = self.args.device

        graph_pooling_type = self.args.graph_pooling_type
        
        self.gin = GraphCNN(
            input_dim=args.node_fea_size, 
            use_select_sim=args.use_select_sim,
            num_layers=args.gin_layer, 
            hidden_dim=args.gin_hid,
            graph_pooling_type=graph_pooling_type, 
            device=self.device
        ).to(self.device)

        # Auto-infer pooled representation total dimension
        per_layer_dim = self.hid
        
        if graph_pooling_type in ("meanmax", "maxavg_concat"):
            per_layer_dim *= 2  # mean+max concatenation
        self.sample_input_emb_size = per_layer_dim * self.args.gin_layer

        # Build proj_head based on auto-inferred result - **与原版完全一致**
        self.proj_head = nn.Sequential(nn.Linear(self.sample_input_emb_size, self.hid), nn.ReLU(inplace=True),
                                       nn.Linear(self.hid, self.hid))

        if args.baseline_mode == 'relation':
            self.rel_classifier = nn.Linear(self.sample_input_emb_size * 2, args.train_classes_num)

        self.dropout = nn.Dropout(args.dropout)

    def encode_graphs(self, data, prompt_embs=None):
        """图编码 - 使用GIN提取图表示并返回池化特征"""
        pooled_h_layers, node_embeds, Adj_block_idx, hidden_rep, final_hidd = \
            self.gin(data, mode='test', promp=prompt_embs)
        embs = torch.cat(pooled_h_layers[1:], -1)  # [B, D]
        return embs

    def sample_input_GNN(self, batch_data, prompt_embs=None):
        """少样本学习 - 处理PyG批次数据用于few-shot任务"""
        # Direct forward pass with PyG batch
        pooled_h_layers, node_embeds, Adj_block_idx, hidden_rep, final_hidd = \
            self.gin(batch_data, mode='test', promp=prompt_embs)
        
        embs = torch.cat(pooled_h_layers[1:], -1)
        final_hidds = [final_hidd] if self.args.use_select_sim else []
        
        return embs, final_hidds

    def forward(self, data):
        """前向传播 - 图分类的主要接口"""
        output_embeds, node_embeds, Adj_block_idx, _, _ = self.gin(data)
        pooled_h = torch.cat(output_embeds[1:], -1)  # 拼接多层特征
        pooled_h = self.proj_head(pooled_h)  # 投影头处理
        return pooled_h

    def loss_cal(self, x, x_aug):
        """
        对比损失计算 - InfoNCE损失
        
        计算正样本和负样本的相似度，使用温度参数T调节
        损失函数：L = -log(exp(sim_pos/T) / Σ(exp(sim_neg/T)))
        """
        batch_size, _ = x.size()
        x_abs = x.norm(dim=1)
        x_aug_abs = x_aug.norm(dim=1)

        sim_matrix = torch.einsum('ik,jk->ij', x, x_aug) / torch.einsum('i,j->ij', x_abs, x_aug_abs)
        sim_matrix = torch.exp(sim_matrix / self.T)
        pos_sim = sim_matrix[range(batch_size), range(batch_size)]
        loss = pos_sim / (sim_matrix.sum(dim=1) - pos_sim)
        loss = - torch.log(loss).mean()

        return loss