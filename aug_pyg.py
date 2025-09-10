import torch
import copy
import random
import numpy as np
import torch.nn.functional as F
from torch_geometric.data import Data, Batch
from torch_geometric.utils import to_dense_adj, dense_to_sparse, subgraph, add_self_loops, remove_self_loops
from torch_geometric.transforms import Compose


class GraphAugmentation:
    """
    PyTorch Geometric data augmentation class for graph few-shot learning.
    Based on the original aug.py but adapted for PyG Data format.
    """
    
    def __init__(self, aug_type='random', aug_ratio=0.1, seed=None):
        """
        Initialize the graph augmentation module.
        
        Args:
            aug_type (str): Type of augmentation ('drop_node', 'mask_node', 'drop_edge', 
                          'mask_feature', 'drop_feature', 'add_edge', 'random')
            aug_ratio (float): Ratio of nodes/edges/features to augment
            seed (int): Random seed for reproducibility
        """
        self.aug_type = aug_type
        self.aug_ratio = aug_ratio
        self.seed = seed
        
        # Mapping of augmentation types to methods
        self.aug_methods = {
            'drop_node': self.drop_node,
            'mask_node': self.mask_node, 
            'drop_edge': self.drop_edge,
            'mask_feature': self.mask_feature,
            'drop_feature': self.drop_feature,
            'add_edge': self.add_edge,
            'subgraph': self.random_subgraph,
            'random': self.random_augment
        }
    
    def __call__(self, data):
        """Apply augmentation to PyG Data object or Batch."""
        if self.seed is not None:
            torch.manual_seed(self.seed)
            np.random.seed(self.seed)
            random.seed(self.seed)
            
        if isinstance(data, Batch):
            return self._augment_batch(data)
        elif isinstance(data, Data):
            return self._augment_single(data)
        else:
            raise ValueError("Input must be torch_geometric.data.Data or Batch")
    
    def _augment_batch(self, batch):
        """Augment a batch of graphs."""
        # Convert batch to list of individual graphs
        graphs = batch.to_data_list()
        
        # Apply augmentation to each graph
        augmented_graphs = []
        for graph in graphs:
            aug_graph = self._augment_single(graph)
            augmented_graphs.append(aug_graph)
        
        # Convert back to batch
        return Batch.from_data_list(augmented_graphs)
    
    def _augment_single(self, data):
        """Augment a single graph."""
        if self.aug_type not in self.aug_methods:
            raise ValueError(f"Unknown augmentation type: {self.aug_type}")
        
        # Create a copy to avoid modifying original data
        data_aug = data.clone()
        
        # Apply the specified augmentation
        return self.aug_methods[self.aug_type](data_aug)
    
    def drop_node(self, data):
        """
        Drop nodes from the graph (similar to aug_drop_node in original aug.py).
        Note: Original implementation only removes edges, keeps node features unchanged.
        """
        num_nodes = data.num_nodes
        if num_nodes <= 1:
            return data
            
        # Use int(num_nodes / 10) to match original implementation exactly
        drop_num = int(num_nodes / 10)
        if drop_num == 0:
            drop_num = 1  # At least drop one node for small graphs
        
        # Use numpy random choice to match original implementation
        if hasattr(np, 'random'):
            idx_drop = np.random.choice(num_nodes, min(drop_num, num_nodes-1), replace=False)
            idx_drop = torch.from_numpy(idx_drop)
        else:
            idx_drop = torch.randperm(num_nodes)[:drop_num]
        
        # Convert edge_index to adjacency matrix (matching original approach)
        if hasattr(data, 'edge_index') and data.edge_index is not None:
            # Create adjacency matrix
            adj = torch.zeros((num_nodes, num_nodes), dtype=torch.float)
            if data.edge_index.size(1) > 0:
                adj[data.edge_index[0], data.edge_index[1]] = 1
            
            # Remove edges connected to dropped nodes (set rows and columns to 0)
            adj[idx_drop, :] = 0
            adj[:, idx_drop] = 0
            
            # Convert back to edge_index
            edge_index = adj.nonzero().t()
            data.edge_index = edge_index.to(data.edge_index.dtype)
            
            # Update edge attributes if present - remove attributes for dropped edges
            if hasattr(data, 'edge_attr') and data.edge_attr is not None:
                # This is more complex as we need to match the new edge_index
                # For simplicity, we'll remove edge_attr for now as original doesn't handle it
                data.edge_attr = None
        
        # Note: Original implementation does NOT remove node features
        # Node features remain unchanged, only edges are removed
        
        return data
    
    def mask_node(self, data):
        """
        Mask node features with noise (similar to aug_fea_mask in original aug.py).
        """
        if not hasattr(data, 'x') or data.x is None:
            return data
            
        num_nodes = data.num_nodes
        feat_dim = data.x.size(1)
        # Use int(num_nodes / 10) to match original implementation exactly
        mask_num = int(num_nodes / 10)
        if mask_num == 0 and num_nodes > 0:
            mask_num = 1  # At least mask one node for small graphs
        
        if mask_num >= num_nodes:
            mask_num = num_nodes - 1 if num_nodes > 1 else 0
        
        if mask_num > 0:
            # Use numpy random choice to match original implementation
            idx_mask = np.random.choice(num_nodes, mask_num, replace=False)
            idx_mask = torch.from_numpy(idx_mask)
            
            # Generate random noise exactly like original
            noise = np.random.normal(loc=0.5, scale=0.5, size=(mask_num, feat_dim))
            noise = torch.tensor(noise, dtype=torch.float32)
            
            data.x[idx_mask] = noise.to(data.x.dtype).to(data.x.device)
        
        return data
    
    def drop_edge(self, data):
        """Drop edges randomly from the graph."""
        if not hasattr(data, 'edge_index') or data.edge_index is None:
            return data
            
        num_edges = data.edge_index.size(1)
        if num_edges == 0:
            return data
            
        drop_num = max(1, int(num_edges * self.aug_ratio))
        
        # Randomly select edges to keep
        idx_keep = torch.randperm(num_edges)[drop_num:]
        
        if len(idx_keep) == 0:
            # Keep at least one edge if possible
            if num_edges > 0:
                idx_keep = torch.tensor([0])
            else:
                return data
        
        data.edge_index = data.edge_index[:, idx_keep]
        
        # Update edge attributes if present
        if hasattr(data, 'edge_attr') and data.edge_attr is not None:
            data.edge_attr = data.edge_attr[idx_keep]
            
        return data
    
    def add_edge(self, data):
        """Add random edges to the graph."""
        if not hasattr(data, 'edge_index') or data.num_nodes <= 1:
            return data
            
        num_nodes = data.num_nodes
        max_edges = num_nodes * (num_nodes - 1) // 2  # Maximum edges in undirected graph
        current_edges = data.edge_index.size(1) // 2 if data.edge_index.size(1) > 0 else 0
        
        add_num = min(max(1, int(current_edges * self.aug_ratio)), 
                     max_edges - current_edges)
        
        if add_num <= 0:
            return data
        
        # Get current edges as set for efficient lookup
        current_edge_set = set()
        if data.edge_index.size(1) > 0:
            edge_list = data.edge_index.t().tolist()
            for edge in edge_list:
                current_edge_set.add(tuple(sorted(edge)))
        
        # Generate new random edges
        new_edges = []
        attempts = 0
        max_attempts = add_num * 10
        
        while len(new_edges) < add_num and attempts < max_attempts:
            src = torch.randint(0, num_nodes, (1,)).item()
            dst = torch.randint(0, num_nodes, (1,)).item()
            
            if src != dst:  # No self-loops
                edge = tuple(sorted([src, dst]))
                if edge not in current_edge_set:
                    new_edges.append([src, dst])
                    new_edges.append([dst, src])  # Add both directions for undirected
                    current_edge_set.add(edge)
            attempts += 1
        
        if new_edges:
            new_edge_index = torch.tensor(new_edges, dtype=data.edge_index.dtype).t()
            data.edge_index = torch.cat([data.edge_index, new_edge_index], dim=1)
            
            # Add zero edge attributes if edge_attr exists
            if hasattr(data, 'edge_attr') and data.edge_attr is not None:
                new_attr = torch.zeros((len(new_edges), data.edge_attr.size(1)),
                                      dtype=data.edge_attr.dtype, device=data.edge_attr.device)
                data.edge_attr = torch.cat([data.edge_attr, new_attr], dim=0)
        
        return data
    
    def mask_feature(self, data):
        """
        Mask features by setting them to zero (similar to aug_fea_drop in original aug.py).
        """
        if not hasattr(data, 'x') or data.x is None:
            return data
            
        feat_dim = data.x.size(1)
        # Use exact same logic as original: uniform random < 0.1
        drop_mask = torch.empty((feat_dim,), dtype=torch.float32).uniform_(0, 1) < 0.1
        
        # Clone to avoid modifying original
        data.x = data.x.clone()
        data.x[:, drop_mask] = 0
        
        return data
    
    def drop_feature(self, data):
        """
        Apply dropout to node features (similar to aug_fea_dropout in original aug.py).
        """
        if not hasattr(data, 'x') or data.x is None:
            return data
            
        data.x = F.dropout(data.x, p=self.aug_ratio, training=True)
        return data
    
    def random_subgraph(self, data):
        """Extract a random subgraph by keeping a subset of nodes."""
        if not hasattr(data, 'edge_index') or data.num_nodes <= 1:
            return data
            
        num_nodes = data.num_nodes
        keep_num = max(1, int(num_nodes * (1 - self.aug_ratio)))
        
        # Randomly select nodes to keep
        idx_keep = torch.randperm(num_nodes)[:keep_num]
        
        # Extract subgraph
        edge_index, edge_attr = subgraph(idx_keep, data.edge_index, 
                                       edge_attr=data.edge_attr if hasattr(data, 'edge_attr') else None,
                                       relabel_nodes=True, num_nodes=num_nodes)
        
        # Update data
        data.edge_index = edge_index
        if hasattr(data, 'edge_attr') and edge_attr is not None:
            data.edge_attr = edge_attr
            
        if hasattr(data, 'x') and data.x is not None:
            data.x = data.x[idx_keep]
            
        if hasattr(data, 'y') and data.y is not None and data.y.dim() > 0:
            if len(data.y) == num_nodes:  # Node-level labels
                data.y = data.y[idx_keep]
                
        data.num_nodes = len(idx_keep)
        return data
    
    def random_augment(self, data):
        """Randomly choose one augmentation method."""
        methods = ['drop_node', 'mask_node', 'drop_edge', 'mask_feature']
        chosen_method = random.choice(methods)
        return self.aug_methods[chosen_method](data)




def main():
    """示例用法和测试"""
    # 创建一个简单的测试图
    edge_index = torch.tensor([[0, 1, 1, 2, 2, 3],
                              [1, 0, 2, 1, 3, 2]], dtype=torch.long)
    x = torch.randn(4, 8)  # 4个节点，8个特征
    y = torch.tensor([0])  # 图标签
    
    data = Data(x=x, edge_index=edge_index, y=y)
    
    print("原始图:")
    print(f"节点数: {data.num_nodes}, 边数: {data.num_edges}")
    print(f"节点特征形状: {data.x.shape}")
    
    # 测试不同的增强方法
    augmentations = ['drop_node', 'mask_node', 'drop_edge', 'mask_feature', 'add_edge']
    
    for aug_type in augmentations:
        aug = GraphAugmentation(aug_type=aug_type, aug_ratio=0.2, seed=42)
        aug_data = aug(data)
        print(f"\n{aug_type}:")
        print(f"节点数: {aug_data.num_nodes}, 边数: {aug_data.num_edges}")
        if hasattr(aug_data, 'x') and aug_data.x is not None:
            print(f"节点特征形状: {aug_data.x.shape}")


if __name__ == "__main__":
    main()
