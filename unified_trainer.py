import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
from copy import deepcopy
from tqdm import tqdm

from gnn_model import Model, Prompt, LogReg
from train_ldm import LDM  
from dataset import Dataset
from aug import aug_fea_mask, aug_drop_node, aug_fea_drop, aug_fea_dropout


class UnifiedTrainer:
    """统一的训练器，支持encoder预训练 -> LDM训练 -> 增强评估的完整流程"""
    
    def __init__(self, args, logf=None):
        self.args = args
        self.logf = logf
        
        # 设置保存路径
        self.save_dir = './savepoint'
        os.makedirs(self.save_dir, exist_ok=True)
        
        # 初始化数据集
        self.dataset = Dataset(args.dataset_name, args)
        args.train_classes_num = self.dataset.train_classes_num
        args.test_classes_num = self.dataset.test_classes_num
        args.node_fea_size = self.dataset.train_graphs[0].node_features.shape[1]
        args.N_way = self.dataset.test_classes_num
        
        # 阶段1：初始化encoder相关组件
        self.model = Model(args).to(args.device)
        self.prompt = Prompt(self.args).to(args.device)
        self.encoder_optimizer = optim.Adam(
            self.model.parameters(), 
            lr=args.lr, 
            weight_decay=args.weight_decay
        )
        
        # 阶段2：LDM相关组件（延迟初始化）
        self.ldm = None
        self.ldm_optimizer = None
        
        # 评估相关
        self.log = LogReg(self.model.sample_input_emb_size, self.args.N_way).to(args.device)
        if getattr(args, 'use_prompt', True):
            self.opt = optim.SGD([
                {'params': self.log.parameters()}, 
                {'params': self.prompt.parameters()}
            ], lr=0.01)
        else:
            self.opt = optim.SGD([{'params': self.log.parameters()}], lr=0.01)
        self.xent = nn.CrossEntropyLoss()

    def train_encoder(self):
        """阶段1：训练encoder（对比学习）"""
        print("=== 阶段1：开始训练Encoder ===")
        
        best = 1e9
        best_t = 0
        cnt_wait = 0
        
        # 准备数据增强
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

        print("图增强完成!")
        
        for i in tqdm(range(self.args.epoch_num), desc="训练Encoder"):
            loss = self._pretrain_step(graph_aug1, graph_aug2)
            
            if loss is None:
                continue
                
            if i % 50 == 0:
                tqdm.write('Epoch {} Loss {:.4f}'.format(i, loss))
                if self.logf is not None:
                    self.logf.write('Epoch {} Loss {:.4f}'.format(i, loss) + '\n')
                    
                if loss < best:
                    best = loss
                    best_t = i
                    cnt_wait = 0
                    # 保存encoder状态
                    torch.save(
                        self.model.state_dict(), 
                        os.path.join(self.save_dir, f'{self.args.dataset_name}_encoder.pkl')
                    )
                else:
                    cnt_wait += 1
                    
            if cnt_wait > self.args.patience:
                tqdm.write("提前停止!")
                break
        
        print(f"Encoder训练完成，最佳epoch: {best_t}")
        
    def _pretrain_step(self, graph_aug1, graph_aug2):
        """执行一步对比学习的预训练"""
        self.model.train()
        train_embs = self.model(graph_aug1)
        train_embs_aug = self.model(graph_aug2)
        
        loss = self.model.loss_cal(train_embs, train_embs_aug)
        
        self.encoder_optimizer.zero_grad()
        loss.backward()
        self.encoder_optimizer.step()
        return loss
        
    def init_ldm_components(self):
        """初始化LDM相关组件"""
        print("=== 初始化LDM组件 ===")
        
        # 基于训练好的encoder获取embedding维度
        # 使用一个样本来推断embedding维度
        self.model.eval()
        with torch.no_grad():
            if self.args.use_prompt:
                self.prompt.eval()
                prompt_embeds = self.prompt()
            else:
                prompt_embeds = torch.zeros(self.args.num_token, self.args.node_fea_size).to(self.args.device)
            
            # 使用第一个训练任务来推断embedding维度
            first_N_class_sample = np.array(list(range(min(self.dataset.train_classes_num, 2))))
            sample_task = self.dataset.sample_one_task(
                self.dataset.train_tasks if hasattr(self.dataset, 'train_tasks') else None,
                first_N_class_sample,
                K_shot=self.args.K_shot,
                query_size=1,  # 只需要一个样本来推断维度
                test_start_idx=0
            )
            
            sample_embs, _ = self.model.sample_input_GNN([sample_task], prompt_embeds, True)
            embedding_dim = sample_embs.shape[1]
            
        print(f"检测到embedding维度: {embedding_dim}")
        
        # 初始化LDM，基于encoder的输出维度
        self.ldm = LDM(
            self.args.device,
            embedding_dim,  # 使用encoder的输出维度
            getattr(self.args, 'time_steps', 100),
            getattr(self.args, 'beta_start', 0.0001),
            getattr(self.args, 'beta_end', 0.02)
        ).to(self.args.device)
        
        self.ldm_optimizer = torch.optim.AdamW(
            self.ldm.parameters(),
            lr=getattr(self.args, 'learning_rate_ldm', 0.001),
            weight_decay=getattr(self.args, 'weight_decay_ldm', 1e-4)
        )
        
        print("LDM组件初始化完成!")
        
    def collect_training_embeddings(self):
        """收集训练集的embeddings用于LDM训练"""
        print("=== 收集训练集embeddings ===")
        
        self.model.eval()
        all_embeddings = []
        all_labels = []
        
        if self.args.use_prompt:
            self.prompt.eval()
            prompt_embeds = self.prompt()
        else:
            prompt_embeds = torch.zeros(self.args.num_token, self.args.node_fea_size).to(self.args.device)
        
        with torch.no_grad():
            all_graphs = self.dataset.train_graphs
            self.training_embeddings = self.model.encode_graphs(all_graphs, prompt_embeds)  # [N, D]
            self.training_labels = torch.tensor([graph.label for graph in all_graphs], device=self.args.device)
    
        print(f"收集了 {self.training_embeddings.shape[0]} 个训练embeddings，维度: {self.training_embeddings.shape}")
        from sklearn.cluster import KMeans
        
        print(f"使用K-Means聚类，聚类数量: {self.args.train_classes_num}")
        
        kmeans = KMeans(n_init=10, n_clusters=self.args.train_classes_num, random_state=42)
        cluster_labels = kmeans.fit_predict(self.training_embeddings.cpu().detach().numpy())
        
        # 聚类中心作为prototypes
        prototypes = torch.tensor(kmeans.cluster_centers_, dtype=torch.float, device=self.args.device)
        
        # 每个样本的条件是其聚类中心
        cluster_labels = torch.tensor(cluster_labels, dtype=torch.long, device=self.args.device)
        self.conditions = prototypes[cluster_labels]  # [N, embedding_dim]
        
        print(f"K-Means聚类完成，prototypes形状: {prototypes.shape}")
        print(f"条件数据形状: {self.conditions.shape}")
        
    def train_ldm(self):
        """训练LDM，基于encoder的embeddings"""
        print("=== 开始训练LDM ===")
        
        # 使用K-Means聚类得到的条件
        conditions = self.conditions
        
        print(f"训练数据: {self.training_embeddings.shape[0]} 个embeddings")
        print(f"条件数据: {conditions.shape[0]} 个条件")
        
        best_loss = float('inf')
        patience_count = 0
        patience = self.args.patience_ldm
        
        for epoch in range(1, getattr(self.args, 'num_epochs_ldm', 200) + 1):
            # 随机打乱数据
            perm = torch.randperm(self.training_embeddings.shape[0])
            shuffled_embeddings = self.training_embeddings[perm]
            shuffled_conditions = conditions[perm]
            
            epoch_loss = 0
            batch_size = getattr(self.args, 'ldm_batch_size', 64)
            num_batches = 0
            
            # 批训练
            for i in range(0, len(shuffled_embeddings), batch_size):
                batch_embeddings = shuffled_embeddings[i:i+batch_size]
                batch_conditions = shuffled_conditions[i:i+batch_size]
                
                loss = self._train_ldm_step(batch_embeddings, batch_conditions)
                epoch_loss += loss.item()
                num_batches += 1
            
            avg_loss = epoch_loss / num_batches if num_batches > 0 else 0
            
            if epoch % 20 == 0:
                print(f"LDM Epoch {epoch}, 平均Loss: {avg_loss:.6f}")
                
            # 早停检查
            if avg_loss < best_loss:
                best_loss = avg_loss
                patience_count = 0
                # 保存最佳模型
                torch.save(
                    self.ldm.state_dict(),
                    os.path.join(self.save_dir, f'{self.args.dataset_name}_ldm.pkl')
                )
            else:
                patience_count += 1
                
            if patience_count >= patience:
                print(f"LDM训练提前停止在epoch {epoch}")
                break
        
        print("LDM训练完成!")
                
    def _train_ldm_step(self, z, conditions):
        """LDM训练步骤"""
        self.ldm.train()
        loss_fn = nn.MSELoss()
        self.ldm_optimizer.zero_grad()
        
        # 前向扩散
        t = torch.randint(0, self.ldm.timesteps, (z.size(0),))
        noisy_z, eps_gt = self.ldm.addnoise(z, t)
        t = t.to(z.device)
        eps_pred = self.ldm.denoise(noisy_z, t, conditions)
        
        # 计算损失
        loss = loss_fn(eps_pred, eps_gt)
        loss.backward()
        self.ldm_optimizer.step()
        
        return loss
        
    def test_with_ldm_augmentation(self):
        """使用LDM增强进行最终测试"""
        print("=== 最终测试（使用LDM增强）===")
        
        # 加载最佳LDM模型
        try:
            self.ldm.load_state_dict(torch.load(
                os.path.join(self.save_dir, f'{self.args.dataset_name}_ldm.pkl'),
                weights_only=True
            ))
            print("LDM模型加载成功!")
        except:
            print("LDM模型加载失败，使用当前模型状态")
        
        self.model.eval()
        self.ldm.eval()
        
        test_accs = []
        start_test_idx = 0
        num_augmented_samples = getattr(self.args, 'num_augmented_samples', 10)
        
        while start_test_idx < len(self.dataset.test_graphs) - self.args.K_shot * self.dataset.test_classes_num:
            test_acc = self._evaluate_one_task_with_ldm(start_test_idx, num_augmented_samples)
            test_accs.append(test_acc)
            start_test_idx += self.args.N_way * self.args.query_size
            
        mean_acc = sum(test_accs) / len(test_accs)
        std = np.array(test_accs).std()
        
        print(f'LDM增强测试准确率: {mean_acc:.4f} ± {std:.4f}')
        if self.logf is not None:
            self.logf.write(f'LDM增强测试准确率: {mean_acc:.4f} ± {std:.4f}\n')
            
        return mean_acc, std
        
    def _evaluate_one_task_with_ldm(self, test_idx, num_augmented_samples):
        """评估一个few-shot任务（使用LDM增强）"""
        self.model.eval()
        
        if self.args.use_prompt:
            self.prompt.train()
            prompt_embeds = self.prompt()
        else:
            prompt_embeds = torch.zeros(self.args.num_token, self.args.node_fea_size).to(self.args.device)
            
        first_N_class_sample = np.array(list(range(self.dataset.test_classes_num)))
        current_task = self.dataset.sample_one_task(
            self.dataset.test_tasks, first_N_class_sample,
            K_shot=self.args.K_shot, query_size=self.args.query_size,
            test_start_idx=test_idx
        )
        
        # 获取原始支持集embeddings
        support_current_sample_input_embs, _ = self.model.sample_input_GNN(
            [current_task], prompt_embeds, True
        )
        
        original_support_data = support_current_sample_input_embs.detach()
        
        # 获取支持集标签
        support_label = []
        for graphs in current_task['support_set']:
            support_label.append(np.array([graph.label for graph in graphs[:self.args.K_shot]]))
        support_label = torch.LongTensor(np.hstack(support_label)).to(self.args.device)
        
        # 使用LDM生成增强的embeddings
        augmented_embeddings = []
        augmented_labels = []
        
        # 为每个类别生成增强样本
        unique_labels = torch.unique(support_label)
        for label in unique_labels:
            # 计算当前支持集中该类别的原型作为条件
            label_mask = support_label == label
            if label_mask.sum() > 0:
                class_embeddings = original_support_data[label_mask]
                prototype = class_embeddings.mean(dim=0)  # 当前支持集的类别原型
                condition = prototype.unsqueeze(0).repeat(num_augmented_samples, 1)  # [num_aug, emb_dim]
                
                # 使用LDM生成新的embeddings
                with torch.no_grad():
                    generated_embs = self.ldm.sample(
                        (num_augmented_samples, original_support_data.shape[1]), 
                        condition
                    )
                    
                augmented_embeddings.append(generated_embs)
                augmented_labels.extend([label.item()] * num_augmented_samples)
        
        # 合并原始和增强的embeddings
        if augmented_embeddings:
            all_augmented_embs = torch.cat(augmented_embeddings, dim=0)
            all_augmented_labels = torch.tensor(augmented_labels, device=self.args.device)
            
            # 合并支持集
            enhanced_support_data = torch.cat([original_support_data, all_augmented_embs], dim=0)
            enhanced_support_labels = torch.cat([support_label, all_augmented_labels], dim=0)
        else:
            enhanced_support_data = original_support_data
            enhanced_support_labels = support_label
            
        # 训练分类器（使用增强的支持集）
        self._train_classifier_simple(enhanced_support_data, enhanced_support_labels)
        
        # 评估查询集
        if self.args.use_prompt:
            self.prompt.eval()
            prompt_embeds = self.prompt()
        else:
            prompt_embeds = torch.zeros(self.args.num_token, self.args.node_fea_size).to(self.args.device)
            
        query_current_sample_input_embs, _ = self.model.sample_input_GNN(
            [current_task], prompt_embeds, False
        )
        query_data = query_current_sample_input_embs.detach()
        
        query_label = []
        for graphs in current_task['query_set']:
            query_label.append(np.array([graph.label for graph in graphs]))
        query_label = torch.LongTensor(np.hstack(query_label)).to(self.args.device)
        
        query_len = query_label.shape[0]
        if current_task['append_count'] != 0:
            query_data = query_data[:query_len - current_task['append_count'], :]
            query_label = query_label[:query_len - current_task['append_count']]
            
        logits = self.log(query_data)
        preds = torch.argmax(logits, dim=1)
        acc = torch.sum(preds == query_label).float() / query_label.shape[0]
        
        return acc.cpu().numpy()
        
    def _train_classifier_simple(self, support_data, support_labels):
        """训练简单的线性分类器（用于LDM增强测试）"""
        self.log.train()
        optimizer = torch.optim.SGD(self.log.parameters(), lr=0.01)
        
        best_loss = 1e9
        wait = 0
        patience = 10
        
        for _ in range(200):  # 减少训练步数
            optimizer.zero_grad()
            
            logits = self.log(support_data)
            loss = self.xent(logits, support_labels)
            
            # L2正则化
            l2_reg = torch.tensor(0.).to(self.args.device)
            for param in self.log.parameters():
                l2_reg += torch.norm(param)
            loss_total = loss + 0.01 * l2_reg
            
            loss_total.backward()
            optimizer.step()
            
            if loss_total < best_loss:
                best_loss = loss_total
                wait = 0
            else:
                wait += 1
            if wait > patience:
                break
                
        self.log.eval()
        
    def test_original(self):
        """使用原始encoder进行测试（无LDM增强）"""
        print("=== 原始Encoder测试 ===")
        
        # 加载最佳encoder
        self.model.load_state_dict(torch.load(
            os.path.join(self.save_dir, f'{self.args.dataset_name}_encoder.pkl'),
            weights_only=True
        ))
        print("Encoder模型加载成功!")
        self.model.eval()
        
        test_accs = []
        start_test_idx = 0
        
        while start_test_idx < len(self.dataset.test_graphs) - self.args.K_shot * self.dataset.test_classes_num:
            test_acc = self._evaluate_one_task(start_test_idx)
            test_accs.append(test_acc)
            start_test_idx += self.args.N_way * self.args.query_size
            
        mean_acc = sum(test_accs) / len(test_accs)
        std = np.array(test_accs).std()
        
        print(f'原始Encoder测试准确率: {mean_acc:.4f} ± {std:.4f}')
        if self.logf is not None:
            self.logf.write(f'原始Encoder测试准确率: {mean_acc:.4f} ± {std:.4f}\n')
            
        return mean_acc, std
        
    def _evaluate_one_task(self, test_idx):
        """评估一个few-shot任务（复用train.py中的逻辑）"""
        self.model.eval()
        
        if self.args.use_prompt:
            self.prompt.train()
            prompt_embeds = self.prompt()
        else:
            prompt_embeds = torch.zeros(self.args.num_token, self.args.node_fea_size).to(self.args.device)
            
        first_N_class_sample = np.array(list(range(self.dataset.test_classes_num)))
        current_task = self.dataset.sample_one_task(
            self.dataset.test_tasks, first_N_class_sample,
            K_shot=self.args.K_shot, query_size=self.args.query_size,
            test_start_idx=test_idx
        )
        
        support_current_sample_input_embs, _ = self.model.sample_input_GNN(
            [current_task], prompt_embeds, True
        )
        
        if self.args.gen_test_num == 0:
            support_data = support_current_sample_input_embs.detach()
            support_data_mixup = None
        else:
            data = support_current_sample_input_embs.reshape(
                self.args.N_way, self.args.K_shot + self.args.gen_test_num,
                self.model.sample_input_emb_size
            )
            support_data = data[:, :self.args.K_shot, :].reshape(
                self.args.N_way * self.args.K_shot,
                self.model.sample_input_emb_size
            ).detach()
            support_data_mixup = data[:, self.args.K_shot:self.args.K_shot + self.args.gen_test_num, :].reshape(
                self.args.N_way * self.args.gen_test_num, 
                self.model.sample_input_emb_size
            ).detach()
            
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
        
        # 训练分类器
        self._train_classifier(support_data, support_data_mixup, support_label, 
                              support_label_mix_a, support_label_mix_b, weight)
        
        # 评估
        if self.args.use_prompt:
            self.prompt.eval()
            prompt_embeds = self.prompt()
        else:
            prompt_embeds = torch.zeros(self.args.num_token, self.args.node_fea_size).to(self.args.device)
            
        query_current_sample_input_embs, _ = self.model.sample_input_GNN(
            [current_task], prompt_embeds, False
        )
        query_data = query_current_sample_input_embs.detach()
        
        query_label = []
        for graphs in current_task['query_set']:
            query_label.append(np.array([graph.label for graph in graphs]))
        query_label = torch.LongTensor(np.hstack(query_label)).to(self.args.device)
        
        query_len = query_label.shape[0]
        if current_task['append_count'] != 0:
            query_data = query_data[:query_len - current_task['append_count'], :]
            query_label = query_label[:query_len - current_task['append_count']]
            
        logits = self.log(query_data)
        preds = torch.argmax(logits, dim=1)
        acc = torch.sum(preds == query_label).float() / query_label.shape[0]
        
        return acc.cpu().numpy()
        
    def _train_classifier(self, support_data, support_data_mixup, support_label,
                         support_label_mix_a, support_label_mix_b, weight):
        """训练线性分类器"""
        self.log.train()
        best_loss = 1e9
        wait = 0
        patience = 10
        
        for _ in range(500):
            self.opt.zero_grad()
            
            # 原始支持数据
            logits = self.log(support_data)
            loss_ori = self.xent(logits, support_label)
            
            # Mixup数据损失
            if self.args.gen_test_num > 0 and support_data_mixup is not None:
                logits_mix = self.log(support_data_mixup)
                loss_mix = (weight * self.xent(logits_mix, support_label_mix_a) + \
                           (1 - weight) * self.xent(logits_mix, support_label_mix_b)).mean()
            else:
                loss_mix = torch.tensor(0.).to(self.args.device)
                
            # L2正则化
            l2_reg = torch.tensor(0.).to(self.args.device)
            for param in self.log.parameters():
                l2_reg += torch.norm(param)
            loss_leg = loss_ori + loss_mix + 0.1 * l2_reg
            
            loss_leg.backward()
            self.opt.step()
            
            if loss_leg < best_loss:
                best_loss = loss_leg
                wait = 0
                torch.save(
                    self.log.state_dict(), 
                    os.path.join(self.save_dir, f'{self.args.dataset_name}_lr.pkl')
                )
            else:
                wait += 1
            if wait > patience:
                break
                
        self.log.load_state_dict(torch.load(
            os.path.join(self.save_dir, f'{self.args.dataset_name}_lr.pkl'),
            weights_only=True
        ))
        self.log.eval()
