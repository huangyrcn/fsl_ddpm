import math
import torch
import torch.nn as nn

from functools import reduce
from operator import mul
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, num_layers, input_dim, hidden_dim, output_dim):
        '''
            num_layers: number of layers in the neural networks (EXCLUDING the input layer). If num_layers=1, this reduces to linear model.
            input_dim: dimensionality of input features
            hidden_dim: dimensionality of hidden units at ALL layers
            output_dim: number of classes for prediction
            device: which device to use
        '''
        super(MLP, self).__init__()

        self.linear_or_not = True  # default is linear model
        self.num_layers = num_layers

        if num_layers < 1:
            raise ValueError("number of layers should be positive!")
        elif num_layers == 1:
            # Linear model
            self.linear = nn.Linear(input_dim, output_dim)
        else:
            # Multi-layer model
            self.linear_or_not = False
            self.linears = torch.nn.ModuleList()
            self.batch_norms = torch.nn.ModuleList()

            self.linears.append(nn.Linear(input_dim, hidden_dim))
            for layer in range(num_layers - 2):
                self.linears.append(nn.Linear(hidden_dim, hidden_dim))
            self.linears.append(nn.Linear(hidden_dim, output_dim))

            for layer in range(num_layers - 1):
                self.batch_norms.append(nn.BatchNorm1d((hidden_dim)))

    def forward(self, x):
        if self.linear_or_not:
            # If linear model
            return self.linear(x)
        else:
            # If MLP
            h = x
            for layer in range(self.num_layers - 1):
                h = F.relu(self.batch_norms[layer](self.linears[layer](h)))
            return self.linears[self.num_layers - 1](h)


class GraphCNN(nn.Module):
    def __init__(self, num_layers=5, num_mlp_layers=2, input_dim=200, hidden_dim=128, output_dim=200,
                 final_dropout=0.5, learn_eps=True, graph_pooling_type='sum', neighbor_pooling_type='sum',
                 use_select_sim=False):
        '''
            num_layers: number of layers in the neural networks (INCLUDING the input layer)
            num_mlp_layers: number of layers in mlps (EXCLUDING the input layer)
            input_dim: dimensionality of input features
            hidden_dim: dimensionality of hidden units at ALL layers
            output_dim: number of classes for prediction
            final_dropout: dropout ratio on the final linear layer
            learn_eps: If True, learn epsilon to distinguish center nodes from neighboring nodes. If False, aggregate neighbors and center nodes altogether.
            neighbor_pooling_type: how to aggregate neighbors (mean, average, or max)
            graph_pooling_type: how to aggregate entire nodes in a graph (mean, average)
            device: which device to use
        '''

        super(GraphCNN, self).__init__()

        self.final_dropout = final_dropout
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        self.num_layers = num_layers
        self.graph_pooling_type = graph_pooling_type
        self.neighbor_pooling_type = neighbor_pooling_type
        self.use_select_sim = use_select_sim

        self.learn_eps = learn_eps
        self.eps = nn.Parameter(torch.zeros(self.num_layers - 1))

        ###List of MLPs
        self.mlps = torch.nn.ModuleList()

        ###List of batchnorms applied to the output of MLP (input of the final prediction linear layer)
        self.batch_norms = torch.nn.ModuleList()

        for layer in range(self.num_layers - 1):
            if layer == 0:
                self.mlps.append(MLP(num_mlp_layers, input_dim, hidden_dim, hidden_dim))
            else:
                self.mlps.append(MLP(num_mlp_layers, hidden_dim, hidden_dim, hidden_dim))

            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))

    def __preprocess_neighbors_sumavepool(self, batch_graph):
        ###create block diagonal sparse matrix

        edge_mat_list = []
        start_idx = [0]
        for i, graph in enumerate(batch_graph):
            start_idx.append(start_idx[i] + len(graph.g))
            edge_mat_list.append(graph.edge_mat + start_idx[i])  # convert to a huge graph
        Adj_block_idx = torch.cat(edge_mat_list, 1)
        Adj_block_elem = torch.ones(Adj_block_idx.shape[1])

        # Add self-loops in the adjacency matrix if learn_eps is False, i.e., aggregate center nodes and neighbor nodes altogether.

        if not self.learn_eps:
            num_node = start_idx[-1]
            self_loop_edge = torch.LongTensor([range(num_node), range(num_node)])
            elem = torch.ones(num_node)
            Adj_block_idx = torch.cat([Adj_block_idx, self_loop_edge], 1)
            Adj_block_elem = torch.cat([Adj_block_elem, elem], 0)

        Adj_block = torch.sparse_coo_tensor(Adj_block_idx, Adj_block_elem, torch.Size([start_idx[-1], start_idx[-1]]))

        return Adj_block.to(self.device), Adj_block_idx

    def __preprocess_neighbors_sumavepool_test(self, batch_graph, num_token):
        ###create block diagonal sparse matrix

        edge_mat_list = []
        start_idx = [0]
        for i, graph in enumerate(batch_graph):
            start_idx.append(start_idx[i] + len(graph.g) + num_token)

            # add original edge to the prompt node
            #             edge = [[i, j] for j in range(len(graph.g)) for i in range(len(graph.g), len(graph.g)+num_token)]
            #             edge.extend([[j, i] for i, j in edge])

            # only add unidirection edge to the original graph
            #             edge = [[j, i] for j in range(len(graph.g)) for i in range(len(graph.g), len(graph.g)+num_token)]

            #             edge.extend([[j,i] for i, j in edge])

            #             edge = torch.LongTensor(edge).transpose(0, 1)
            #             graph.edge_mat = torch.cat([graph.edge_mat, edge], dim=1)

            edge_mat_list.append(graph.edge_mat + start_idx[i])  # convert to a huge graph
        Adj_block_idx = torch.cat(edge_mat_list, 1)
        Adj_block_elem = torch.ones(Adj_block_idx.shape[1])

        # Add self-loops in the adjacency matrix if learn_eps is False, i.e., aggregate center nodes and neighbor nodes altogether.

        if not self.learn_eps:
            num_node = start_idx[-1]
            self_loop_edge = torch.LongTensor([range(num_node), range(num_node)])
            elem = torch.ones(num_node)
            Adj_block_idx = torch.cat([Adj_block_idx, self_loop_edge], 1)
            Adj_block_elem = torch.cat([Adj_block_elem, elem], 0)

        Adj_block = torch.sparse_coo_tensor(Adj_block_idx, Adj_block_elem, torch.Size([start_idx[-1], start_idx[-1]]))

        return Adj_block.to(self.device), Adj_block_idx

    def __preprocess_graphpool(self, batch_graph):
        ###create sum or average pooling sparse matrix over entire nodes in each graph (num graphs x num nodes)

        start_idx = [0]

        # compute the padded neighbor list
        for i, graph in enumerate(batch_graph):
            start_idx.append(start_idx[i] + len(graph.g))

        idx = []
        elem = []
        for i, graph in enumerate(batch_graph):
            ###average pooling
            if self.graph_pooling_type == "average":
                elem.extend([1. / len(graph.g)] * len(graph.g))

            else:
                ###sum pooling
                elem.extend([1] * len(graph.g))

            idx.extend([[i, j] for j in range(start_idx[i], start_idx[i + 1], 1)])
        elem = torch.FloatTensor(elem)
        idx = torch.LongTensor(idx).transpose(0, 1)
        graph_pool = torch.sparse_coo_tensor(idx, elem, torch.Size([len(batch_graph), start_idx[-1]]))

        return graph_pool.to(self.device)

    def __preprocess_graphpool_test(self, batch_graph, num_token):
        ###create sum or average pooling sparse matrix over entire nodes in each graph (num graphs x num nodes)

        start_idx = [0]

        # compute the padded neighbor list
        for i, graph in enumerate(batch_graph):
            start_idx.append(start_idx[i] + len(graph.g) + num_token)

        idx = []
        elem = []
        for i, graph in enumerate(batch_graph):
            ###sum pooling
            elem.extend([1] * (len(graph.g) + num_token))
            idx.extend([[i, j] for j in range(start_idx[i], start_idx[i + 1], 1)])
            ###sum pooling only prompt token
        #             elem.extend([1] * num_token)
        #             idx.extend([[i, j] for j in range(start_idx[i], start_idx[i] + num_token, 1)])

        # sum pooling but remove the prompt
        #             elem.extend([1] * (len(graph.g)))
        #             idx.extend([[i, j] for j in range(start_idx[i]+5, start_idx[i + 1], 1)])

        elem = torch.FloatTensor(elem)
        idx = torch.LongTensor(idx).transpose(0, 1)
        graph_pool = torch.sparse_coo_tensor(idx, elem, torch.Size([len(batch_graph), start_idx[-1]]))

        return graph_pool.to(self.device)

    def next_layer_eps(self, h, layer, padded_neighbor_list=None, Adj_block=None):
        ###pooling neighboring nodes and center nodes separately by epsilon reweighting.

        # If sum or average pooling
        #             print(Adj_block.shape)
        #             print(h.shape)
        pooled = torch.spmm(Adj_block, h)  # [num_node, num_node] x [num_node+token x batch, dim]->[num_node, dim]

        # Reweights the center node representation when aggregating it with its neighbors
        pooled = pooled + (1 + self.eps[layer]) * h  # [num_node, dim] + [num_node+token x batch, dim]
        pooled_rep = self.mlps[layer](pooled)
        h = self.batch_norms[layer](pooled_rep)  # can we insert the prompt vector here?

        # non-linearity
        h = F.relu(h)
        return h

    def forward(self, batch_graph, mode='train', promp=None):
        if mode == 'train':

            X_concat = torch.cat([graph.node_features for graph in batch_graph], 0).to(self.device)  # [num_node, dim]

            graph_pool = self.__preprocess_graphpool(batch_graph)  # [batch, num_node]

            Adj_block, Adj_block_idx = self.__preprocess_neighbors_sumavepool(batch_graph)  # [num_node, num_node]
        else:

            X_concat = torch.cat([promp + graph.node_features.to(self.device) for graph in batch_graph], 0)
            graph_pool = self.__preprocess_graphpool(batch_graph)  # [batch, num_node]

            Adj_block, Adj_block_idx = self.__preprocess_neighbors_sumavepool(batch_graph)  # [num_node, num_node]

        # list of hidden representation at each layer (including input)
        hidden_rep = [X_concat]
        h = X_concat

        for layer in range(self.num_layers - 1):
            h = self.next_layer_eps(h, layer, Adj_block=Adj_block)

            hidden_rep.append(h)

        final_hidd = []

        pooled_h_layers = []

        # perform pooling over all nodes in each graph in every layer
        for layer, h in enumerate(hidden_rep):
            pooled_h = torch.spmm(graph_pool, h)
            pooled_h_layers.append(pooled_h)

        return pooled_h_layers, h, Adj_block_idx, hidden_rep, final_hidd
class LogReg(nn.Module):
    def __init__(self, ft_in, nb_classes):
        super(LogReg, self).__init__()
        self.fc = nn.Linear(ft_in, nb_classes)
        self.bn = nn.BatchNorm1d(ft_in)

    def forward(self, seq):
        ret = self.fc(self.bn(seq))
        return ret

class Model(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.args = args
        self.T = args.t
        self.hid = args.gin_hid
        self.N = args.N_way
        self.K = args.K_shot
        self.Q = args.query_size
        self.device = self.args.device

        self.sample_input_emb_size = args.sample_input_size

        self.gin = GraphCNN(input_dim=args.node_fea_size, use_select_sim=args.use_select_sim, num_layers=args.gin_layer,
                            hidden_dim=args.gin_hid).to(self.device)  # .cuda()

        self.proj_head = nn.Sequential(nn.Linear(self.sample_input_emb_size, self.hid), nn.ReLU(inplace=True),
                                       nn.Linear(self.hid, self.hid))  # whether to use the batchnorm1d?

        if args.baseline_mode == 'relation':
            self.rel_classifier = nn.Linear(self.sample_input_emb_size * 2, args.train_classes_num)

        self.dropout = nn.Dropout(args.dropout)

    def sample_input_GNN(self, tasks, prompt_embs, is_support):
        embs = []
        final_hidds = []
        for task in tasks:
            graphs = []

            if is_support:
                for i in range(len(task['support_set'])):
                    graphs.extend(task['support_set'][i])
            else:
                for i in range(len(task['query_set'])):
                    graphs.extend(task['query_set'][i])

            pooled_h_layers, node_embeds, Adj_block_idx, hidden_rep, final_hidd = self.gin(graphs, mode='test',
                                                                                           promp=prompt_embs)  # [N(K+Q), emb_size]
            embs.append(torch.cat(pooled_h_layers[1:], -1))
            final_hidds.append(final_hidd)
        return torch.cat(embs, 0), [torch.cat([one[layer] for one in final_hidds], 0) for layer in
                                    range(self.gin.num_layers)] if self.args.use_select_sim else []

    def forward(self, batch_graph):
        output_embeds, node_embeds, Adj_block_idx, _, _ = self.gin(batch_graph)
        pooled_h = torch.cat(output_embeds[1:], -1)
        pooled_h = self.proj_head(pooled_h)
        return pooled_h

    def loss_cal(self, x, x_aug):
        batch_size, _ = x.size()
        x_abs = x.norm(dim=1)
        x_aug_abs = x_aug.norm(dim=1)

        sim_matrix = torch.einsum('ik,jk->ij', x, x_aug) / torch.einsum('i,j->ij', x_abs, x_aug_abs)
        sim_matrix = torch.exp(sim_matrix / self.T)
        pos_sim = sim_matrix[range(batch_size), range(batch_size)]
        loss = pos_sim / (sim_matrix.sum(dim=1) - pos_sim)
        loss = - torch.log(loss).mean()

        return loss


class Prompt(nn.Module):
    def __init__(self, args):
        super(Prompt, self).__init__()
        self.token = args.num_token
        self.prompt_dropout = nn.Dropout(args.dropout)
        #         self.prompt_proj = nn.Linear(
        #             args.node_fea_size, args.node_fea_size) #args.gin_hid)
        #         nn.init.kaiming_normal_(
        #             self.prompt_proj.weight, a=0, mode='fan_out')

        val = 0.001  # 0.5  # noqa

        self.prompt_embeddings = nn.Parameter(torch.zeros(
            self.token, args.node_fea_size))
        # xavier_uniform initialization
        nn.init.uniform_(self.prompt_embeddings.data, -val, val)

    def forward(self):
        return self.prompt_dropout(
            self.prompt_embeddings)  # self.prompt_dropout(self.prompt_proj(self.prompt_embeddings))
