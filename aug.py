import torch
import torch.nn.functional as F
from torch_geometric.data import Batch, Data

class GraphAugmentationOptimized:
    """
    高性能 PyG 图增强（保持原版语义）：
      - feature_mask: 节点级加噪（把选中节点整行特征替换为 0.5 + 0.5*N(0,1)）
      - feature_drop: 按维度丢弃（将选中列置零）
      - feature_dropout: F.dropout(p=0.1)
      - node_drop: 不删节点，只删与被选节点相连的边（节点成为孤点）
    实现要点（更快）：
      - 仅用 Torch GPU RNG（无 NumPy、无 CPU↔GPU 往返）
      - 布尔掩码替代 torch.isin
      - 固定比例 p=0.1（对齐原版），忽略外部 aug_ratio 值
    """

    def __init__(self, aug_type, aug_ratio, device, seed=None, **_):
        self.aug_type = self._normalize_aug_name(aug_type)
        self.device = device
        self.p = 0.1  # 固定 10%，对齐原版
        self.gen = torch.Generator(device=device if device is not None else "cpu")
        if seed is not None:
            self.gen.manual_seed(int(seed))

    def __call__(self, data):
        if isinstance(data, Batch):
            # 对 Batch 中的每个图分别应用增强，然后重新组合
            data_list = data.to_data_list()
            aug_data_list = [self._one(g) for g in data_list]
            
            # 使用 PyG 的 Batch.from_data_list，因为 node_drop 不再删除节点
            return Batch.from_data_list(aug_data_list)
        return self._one(data)

    def _one(self, data: Data) -> Data:
        fn = getattr(self, self.aug_type, None)
        return data if fn is None else fn(data)

    # -------- ops（语义与原版一致） --------
    def node_drop(self, data: Data) -> Data:
        """
        不删节点，只删与被选节点相连的边（节点成为孤点）：
        1. 选择要"删除"的节点（实际不删除，只是删除其相关的边）
        2. 删除所有与选中节点相连的边
        3. 保持节点数量不变，但某些节点变成孤点
        """
        ei = getattr(data, "edge_index", None)
        x = getattr(data, "x", None)
        if ei is None or ei.numel() == 0: return data
        if x is None: return data
        
        N = int(x.size(0))
        if N <= 1: return data
        
        # 选择要"删除"的节点（实际不删除节点，只删除其相关的边）
        drop_flag = (torch.rand(N, generator=self.gen, device=x.device) < self.p)
        if not drop_flag.any(): return data
        
        # 创建新的 Data 对象
        out = data.clone()
        
        # 1. x 保持不变（不删除节点）
        out.x = x
        
        # 2. 更新 edge_index：删除与选中节点相连的所有边
        if ei.numel() > 0:
            # 过滤边：删除任何一端是选中节点的边
            valid_edges = (~drop_flag[ei[0]]) & (~drop_flag[ei[1]])
            
            if valid_edges.any():
                out.edge_index = ei[:, valid_edges]
            else:
                # 所有边都被删除
                out.edge_index = torch.empty((2, 0), dtype=ei.dtype, device=ei.device)
        else:
            out.edge_index = ei
        
        # 3. 确保 edge_index 覆盖所有节点（为没有边的节点添加自环边）
        if out.edge_index.numel() > 0:
            max_idx = out.edge_index.max().item()
            expected_max = N - 1
            if max_idx < expected_max:
                # 为没有边的节点添加自环边
                missing_nodes = torch.arange(max_idx + 1, expected_max + 1, device=x.device)
                self_loops = torch.stack([missing_nodes, missing_nodes], dim=0)
                out.edge_index = torch.cat([out.edge_index, self_loops], dim=1)
        elif N > 0:
            # 如果完全没有边，为所有节点添加自环边
            all_nodes = torch.arange(N, device=x.device)
            self_loops = torch.stack([all_nodes, all_nodes], dim=0)
            out.edge_index = self_loops
        
        # 3. 其他属性保持不变（因为节点数量没变）
        # 不需要处理其他节点级属性，因为节点数量保持不变
        
        return out

    def feature_mask(self, data: Data) -> Data:
        x = getattr(data, "x", None)
        if x is None: return data
        N, Fdim = x.size(0), x.size(1)
        if N == 0 or Fdim == 0: return data
        sel = (torch.rand(N, generator=self.gen, device=x.device) < self.p)
        if not sel.any(): return data
        idx = sel.nonzero(as_tuple=False).squeeze(1)
        noise = 0.5 + 0.5 * torch.randn((idx.numel(), Fdim), generator=self.gen,
                                        device=x.device, dtype=x.dtype)
        out = data.clone()
        out.x = x.clone()
        out.x[idx] = noise
        return out

    def feature_drop(self, data: Data) -> Data:
        x = getattr(data, "x", None)
        if x is None: return data
        Fdim = x.size(1)
        if Fdim == 0: return data
        dims_flag = (torch.rand(Fdim, generator=self.gen, device=x.device) < self.p)
        if not dims_flag.any(): return data
        dims = dims_flag.nonzero(as_tuple=False).squeeze(1)
        out = data.clone()
        out.x = x.clone()
        out.x[:, dims] = 0
        return out

    def feature_dropout(self, data: Data) -> Data:
        x = getattr(data, "x", None)
        if x is None: return data
        out = data.clone()
        out.x = F.dropout(x, p=0.1, training=True)  # 原版固定 0.1
        return out

    # 选配：保留 edge_drop（一般不用）
    def edge_drop(self, data: Data) -> Data:
        ei = getattr(data, "edge_index", None)
        if ei is None or ei.numel() == 0: return data
        E = ei.size(1)
        keep = (torch.rand(E, generator=self.gen, device=ei.device) >= self.p)
        if keep.all(): return data
        out = data.clone()
        out.edge_index = ei[:, keep]
        return out

    @staticmethod
    def _normalize_aug_name(name: str) -> str:
        n = (name or "").strip().lower()
        mapping = {
            "feature_mask": "feature_mask", "mask_node": "feature_mask",
            "feature_drop": "feature_drop", "drop_feature": "feature_drop",
            "feature_dropout": "feature_dropout",
            "node_drop": "node_drop", "drop_node": "node_drop",
            "edge_drop": "edge_drop", "drop_edge": "edge_drop",
        }
        return mapping.get(n, n)

def get_optimized_augmentation(aug_type, aug_ratio, device, seed=None, **kwargs):
    # 接口兼容，内部固定 p=0.1，忽略 aug_ratio
    return GraphAugmentationOptimized(aug_type=aug_type, aug_ratio=aug_ratio, device=device, seed=seed)