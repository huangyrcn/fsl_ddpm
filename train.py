import torch
import numpy as np

import torch.nn as nn
import torch.optim as optim
from copy import deepcopy
from tqdm import tqdm
from gnn_model import Model, Prompt, LogReg
from aug import aug_fea_mask, aug_drop_node, aug_fea_drop, aug_fea_dropout
from dataset import Dataset


class Trainer:
    def __init__(self, args, logf=None):
        self.args = args
        self.logf = logf

        self.dataset = Dataset(args.dataset_name, args)
        args.train_classes_num = self.dataset.train_classes_num
        args.test_classes_num = self.dataset.test_classes_num
        args.node_fea_size = self.dataset.train_graphs[0].node_features.shape[1]

        args.N_way = self.dataset.test_classes_num

        self.model = Model(args).to(args.device)  # .cuda()

        self.prompt = Prompt(self.args).to(args.device)

        self.optimizer = optim.Adam(self.model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        # use torch to implement the linear reg
        self.log = LogReg(self.model.sample_input_emb_size, self.args.N_way).to(args.device)
        
        # 根据use_prompt决定优化器参数
        if getattr(args, 'use_prompt', True):
            self.opt = optim.SGD([{'params': self.log.parameters()}, {'params': self.prompt.parameters()}], lr=0.01)
        else:
            self.opt = optim.SGD([{'params': self.log.parameters()}], lr=0.01)
        self.xent = nn.CrossEntropyLoss()

    def train(self):
        # best_test_acc = 0
        # best_valid_acc = 0
        best = 1e9
        best_t = 0
        cnt_wait = 0

        train_accs = []
        # graph_copy_1 = deepcopy(self.dataset.train_graphs)
        graph_copy_2 = deepcopy(self.dataset.train_graphs)
        if self.args.aug1 == 'identity':
            graph_aug1 = self.dataset.train_graphs
        elif self.args.aug1 == 'node_drop':
            graph_aug1 = aug_drop_node(self.dataset.train_graphs)
        elif self.args.aug1 == 'feature_mask':
            graph_aug1 = aug_fea_mask(self.dataset.train_graphs)
        elif self.args.aug1 == 'feature_drop':
            graph_aug1 = aug_fea_drop(self.dataset.train_graphs)
        elif self.args.aug1 == 'feature_dropout':
            graph_aug1 = aug_fea_dropout(self.dataset.train_graphs)

        if self.args.aug2 == 'node_drop':
            graph_aug2 = aug_drop_node(graph_copy_2)
        elif self.args.aug2 == 'feature_mask':
            graph_aug2 = aug_fea_mask(graph_copy_2)
        elif self.args.aug2 == 'feature_drop':
            graph_aug2 = aug_fea_drop(self.dataset.train_graphs)
        elif self.args.aug2 == 'feature_dropout':
            graph_aug2 = aug_fea_dropout(self.dataset.train_graphs)

        print("graph augmentation complete!")
        
        # 新增1行：包装range为tqdm
        for i in tqdm(range(self.args.epoch_num), desc="Training"):
            loss = self._pretrain_step(graph_aug1, graph_aug2)

            if loss == None: continue

            if i % 50 == 0:
                tqdm.write('Epoch {} Loss {:.4f}'.format(i, loss))
                if self.logf is not None:
                    self.logf.write('Epoch {} Loss {:.4f}'.format(i, loss) + '\n')
                if loss < best:
                    best = loss
                    best_t = i
                    cnt_wait = 0
                    torch.save(self.model.state_dict(), './savepoint/' + self.args.dataset_name + '_model.pkl')
                else:
                    cnt_wait += 1
            if cnt_wait > self.args.patience:
                tqdm.write("Early Stopping!")
                break

    def test(self):
        best_test_acc = 0
        self.model.load_state_dict(torch.load('./savepoint/' + self.args.dataset_name + '_model.pkl', weights_only=True))
        print("model load success!")
        self.model.eval()

        test_accs = []
        start_test_idx = 0
        while start_test_idx < len(self.dataset.test_graphs) - self.args.K_shot * self.dataset.test_classes_num:
            test_acc = self._evaluate_one_task(start_test_idx)
            test_accs.append(test_acc)
            start_test_idx += self.args.N_way * self.args.query_size

        # print('test task num', len(test_accs))
        mean_acc = sum(test_accs) / len(test_accs)
        std = np.array(test_accs).std()
        #         if mean_acc > best_test_acc:
        #             best_test_acc = mean_acc

        print('Mean Test Acc {:.4f}  Std {:.4f}'.format(mean_acc, std))
        if self.logf is not None:
            self.logf.write('Mean Test Acc {:.4f}  Std {:.4f}'.format(mean_acc, std) + '\n')

        return best_test_acc

    def _pretrain_step(self, graph_aug1, graph_aug2):
        """执行一步对比学习的预训练"""
        self.model.train()
        train_embs = self.model(graph_aug1)
        train_embs_aug = self.model(graph_aug2)

        loss = self.model.loss_cal(train_embs, train_embs_aug)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss

    def _evaluate_one_task(self, test_idx):
        """完整评估一个 few-shot 任务"""
        self.model.eval()
        
        if self.args.use_prompt:
            self.prompt.train()
            prompt_embeds = self.prompt()
        else:
            prompt_embeds = torch.zeros(self.args.num_token, self.args.node_fea_size).to(self.args.device)

        first_N_class_sample = np.array(list(range(self.dataset.test_classes_num)))
        current_task = self.dataset.sample_one_task(self.dataset.test_tasks, first_N_class_sample,
                                                    K_shot=self.args.K_shot, query_size=self.args.query_size,
                                                    test_start_idx=test_idx)
        support_current_sample_input_embs, support_current_sample_input_embs_selected = self.model.sample_input_GNN(
            [current_task], prompt_embeds, True)  # [N(K+Q), emb_size]

        if self.args.gen_test_num == 0:
            # 没有 mixup 数据：直接使用原始嵌入
            support_data = support_current_sample_input_embs.detach()
            support_data_mixup = None
        else:
            # 有 mixup 数据：需要 reshape 分离
            data = support_current_sample_input_embs.reshape(self.args.N_way, self.args.K_shot + self.args.gen_test_num,
                                                       self.model.sample_input_emb_size)
            support_data = data[:, :self.args.K_shot, :].reshape(self.args.N_way * self.args.K_shot,
                                                                              self.model.sample_input_emb_size).detach()
            support_data_mixup = data[:, self.args.K_shot:self.args.K_shot + self.args.gen_test_num,
                                                                              :].reshape(
                                                                              self.args.N_way * self.args.gen_test_num, self.model.sample_input_emb_size).detach()  # .cpu().numpy()

        support_label, support_label_mix_a, weight, support_label_mix_b = [], [], [], []
        for graphs in current_task['support_set']:
            support_label.append(np.array([graph.label for graph in graphs[:self.args.K_shot]]))
            support_label_mix_a.append(np.array([graph.y_a for graph in graphs[self.args.K_shot:]]))
            support_label_mix_b.append(np.array([graph.y_b for graph in graphs[self.args.K_shot:]]))
            weight.append(np.array([graph.lam for graph in graphs[self.args.K_shot:]]))

        support_label = torch.LongTensor(np.hstack(support_label)).to(self.args.device)
        support_label_mix_a = torch.LongTensor(np.hstack(support_label_mix_a)).to(self.args.device)
        support_label_mix_b = torch.LongTensor(np.hstack(support_label_mix_b)).to(self.args.device)
        weight = torch.FloatTensor(np.hstack(weight)).to(self.args.device)

        # 局部函数：训练线性分类器
        def _train_classifier():
            self.log.train()
            best_loss = 1e9
            wait = 0
            patience = 10
            for _ in range(500):
                self.opt.zero_grad()
                # original support data
                logits = self.log(support_data)
                loss_ori = self.xent(logits, support_label)

                # 只有当有mixup数据时才计算mixup损失
                if self.args.gen_test_num > 0:
                    # mixup data
                    logits_mix = self.log(support_data_mixup)  # [Nxgen, class]
                    loss_mix = (weight * self.xent(logits_mix, support_label_mix_a) + \
                                (1 - weight) * self.xent(logits_mix, support_label_mix_b)).mean()
                else:
                    loss_mix = torch.tensor(0.).to(self.args.device)

                l2_reg = torch.tensor(0.).to(self.args.device)
                for param in self.log.parameters():
                    l2_reg += torch.norm(param)
                loss_leg = loss_ori + loss_mix + 0.1 * l2_reg

                loss_leg.backward()
                self.opt.step()

                if loss_leg < best_loss:
                    best_loss = loss_leg
                    wait = 0
                    torch.save(self.log.state_dict(), './savepoint/' + self.args.dataset_name + '_lr.pkl')
                else:
                    wait += 1
                if wait > patience:
                    print("Early Stopping!")
                    break

            self.log.load_state_dict(torch.load('./savepoint/' + self.args.dataset_name + '_lr.pkl', weights_only=True))
            self.log.eval()

        # 调用局部函数训练分类器
        _train_classifier()
        
        if self.args.use_prompt:
            self.prompt.eval()
            prompt_embeds = self.prompt()
        else:
            prompt_embeds = torch.zeros(self.args.num_token, self.args.node_fea_size).to(self.args.device)

        query_current_sample_input_embs, _ = self.model.sample_input_GNN(
            [current_task], prompt_embeds, False)  # [N(K+Q), emb_size]
        query_label = []

        query_data = query_current_sample_input_embs.detach()  # 统一处理，保持为tensor

        for graphs in current_task['query_set']:
            query_label.append(np.array([graph.label for graph in graphs]))

        query_label = torch.LongTensor(np.hstack(query_label)).to(self.args.device)

        query_len = query_label.shape[0]
        if current_task['append_count'] != 0:
            query_data = query_data[: query_len - current_task['append_count'], :]
            query_label = query_label[: query_len - current_task['append_count']]

        logits = self.log(query_data)
        preds = torch.argmax(logits, dim=1)
        acc = torch.sum(preds == query_label).float() / query_label.shape[0]

        test_acc = acc.cpu().numpy()

        return test_acc
__all__ = ["Trainer"]
