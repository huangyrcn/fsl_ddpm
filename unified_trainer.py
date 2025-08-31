import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import os
from copy import deepcopy
from tqdm import tqdm
import json
import hashlib

# 添加 wandb 导入
import wandb

# 添加 sklearn 导入
from sklearn.cluster import KMeans


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
        
        # 评估相关：仅使用线性分类器
        in_dim = self.model.sample_input_emb_size
        num_cls = self.args.N_way
        self.log = LogReg(in_dim, num_cls).to(self.args.device)
        # 注意：删除了self.opt，改为各训练函数内部自己建优化器
        self.xent = nn.CrossEntropyLoss()

    

    def load_pretrained_encoder(self, ckpt_path=None):
        """从已保存的pkl加载encoder权重并切到eval模式。
        ckpt_path为空时，默认从 savepoint/{dataset_name}_encoder.pkl 加载。
        """
        if ckpt_path is None:
            ckpt_path = os.path.join(self.save_dir, f'{self.args.dataset_name}_encoder.pkl')
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"找不到encoder权重文件: {ckpt_path}")
        state = torch.load(ckpt_path, map_location=self.args.device, weights_only=True)
        self.model.load_state_dict(state)
        self.model.eval()
        print(f"已从 {os.path.basename(ckpt_path)} 加载Encoder权重，并切换到eval模式")

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
                
            # 每个epoch都检查早停条件
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
                
            # 每50个epoch打印一次（保持原有的打印频率）
            if i % 50 == 0:
                tqdm.write('Epoch {} Loss {:.4f}'.format(i, loss))
                if self.logf is not None:
                    self.logf.write('Epoch {} Loss {:.4f}'.format(i, loss) + '\n')
                    
            # 每个epoch都检查早停
            if cnt_wait > self.args.patience:
                tqdm.write("提前停止!")
                break
        
        print(f"Encoder训练完成，最佳epoch: {best_t}")
        
    def _pretrain_step(self, graph_aug1, graph_aug2):
        """执行一步对比学习的预训练
        
        Args:
            graph_aug1: 第一组增强图数据
            graph_aug2: 第二组增强图数据
            
        Returns:
            loss: 对比学习损失值
        """
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
        
        # 直接使用模型公开的embedding维度配置
        embedding_dim = self.model.sample_input_emb_size
            
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
    
    def load_pretrained_ldm(self, ckpt_path=None):
        """从ckpt加载LDM state_dict，并切到eval模式。"""
        if ckpt_path is None:
            ckpt_path = os.path.join(self.save_dir, f'{self.args.dataset_name}_ldm.pkl')
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"找不到LDM权重文件: {ckpt_path}")
        self.ldm.load_state_dict(torch.load(ckpt_path, map_location=self.args.device, weights_only=True))
        self.ldm.eval()
        print(f"已从 {os.path.basename(ckpt_path)} 加载LDM权重，并切换到eval模式")
        
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
        
        # 根据参数选择条件类型
        condition_type = getattr(self.args, 'condition_type', 'kmeans')  # 默认为kmeans
        
        if condition_type == 'kmeans':
            # 使用 K-Means 进行聚类获取类别 Prototype
            kmeans = KMeans(n_init=10, n_clusters=self.args.train_classes_num, random_state=42)
            cluster_labels = kmeans.fit_predict(self.training_embeddings.cpu().detach().numpy())  # (N,) 或 (num_graphs,)
            prototypes = torch.tensor(
                kmeans.cluster_centers_, dtype=torch.float, device=self.args.device
            )  # num_clusters x latent_dim
            self.prototypes = prototypes

            # 计算每个样本的类别 Prototype 作为条件
            cluster_labels = torch.tensor(cluster_labels, dtype=torch.long, device=self.args.device)
            self.conditions = prototypes[cluster_labels]  # 获取每个样本对应的 Prototype
            
            print(f"K-Means聚类完成，prototypes形状: {prototypes.shape}")
            print(f"Prototype-based条件数据形状: {self.conditions.shape}")
            
        elif condition_type == 'self_conditioning':
            # 使用样本自己作为条件
            self.conditions = self.training_embeddings.clone()
            # 注意：这里不设置self.prototypes，但在可视化时会重新计算kmeans
            
            print(f"使用self-conditioning，条件数据形状: {self.conditions.shape}")
            
        else:
            raise ValueError(f"不支持的condition_type: {condition_type}，支持的类型: kmeans, self_conditioning")
        
        # 对条件进行归一化
        self.conditions = F.normalize(self.conditions, dim=1)
        print(f"最终条件数据形状: {self.conditions.shape}")
        
    def train_ldm(self):
        """训练LDM，基于encoder的embeddings"""
        print("=== 开始训练LDM ===")
        
        # 根据条件类型获取条件数据
        conditions = self.conditions
        condition_type = getattr(self.args, 'condition_type', 'kmeans')
        
        print(f"训练数据: {self.training_embeddings.shape[0]} 个embeddings")
        if condition_type == 'kmeans':
            print(f"条件数据: {conditions.shape[0]} 个条件（基于{self.args.train_classes_num}个聚类中心）")
        elif condition_type == 'self_conditioning':
            print(f"条件数据: {conditions.shape[0]} 个条件（使用样本自己作为条件）")
        else:
            print(f"条件数据: {conditions.shape[0]} 个条件")
        
        decay = float(getattr(self.args, 'ldm_ema_decay', 0.9))
        check_interval = int(getattr(self.args, 'ldm_es_interval', 20))
        ema_loss = None
        best_ema = float('inf')
        patience_count = 0
        patience = self.args.patience_ldm
        
        # 创建进度条
        pbar = tqdm(range(1, getattr(self.args, 'num_epochs_ldm', 200) + 1), desc="LDM Training")
        
        for epoch in pbar:
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
            
            # 更新进度条描述
            pbar.set_postfix({'Loss': f'{avg_loss:.4f}'})
            
            # 记录到 wandb（如果启用）
            if wandb.run is not None:
                wandb.log({
                    "ldm_loss": avg_loss,
                    "ldm_ema_loss": ema_loss if ema_loss is not None else avg_loss
                }, step=epoch)
            
            # EMA更新
            if ema_loss is None:
                ema_loss = avg_loss
            else:
                ema_loss = decay * ema_loss + (1.0 - decay) * avg_loss

            # 每个epoch都进行早停检查（基于EMA）
            if ema_loss < best_ema:
                best_ema = ema_loss
                patience_count = 0
                torch.save(
                    self.ldm.state_dict(),
                    os.path.join(self.save_dir, f'{self.args.dataset_name}_ldm.pkl')
                )
            else:
                patience_count += 1
            if epoch % check_interval == 0:
                # 记录到 wandb（如果启用）
                if wandb.run is not None:
                    wandb.log({
                        "ldm_epoch": epoch,
                        "ldm_avg_loss": avg_loss,
                        "ldm_ema_loss": ema_loss,
                        "ldm_patience_count": patience_count,
                        "ldm_patience": patience
                    }, step=epoch)
            if patience_count >= patience:
                break
        
    def _train_ldm_step(self, z, conditions):
        """LDM训练步骤（使用ldm.loss方法）
        
        Args:
            z: 输入embeddings [B, D]
            conditions: 条件embeddings [B, D]
            
        Returns:
            loss: LDM训练损失值
        """
        self.ldm.train()
        self.ldm_optimizer.zero_grad()
        
        # 使用ldm.loss方法计算损失
        loss = self.ldm.loss(z, conditions, p_uncond=0.1, control=None)  # 训练时不使用control
        
        loss.backward()
        self.ldm_optimizer.step()
        
        return loss
        
    def test_model(self, use_ldm_augmentation=True, test_name=None):
        """统一的测试函数，支持原始Encoder和LDM增强两种模式
        
        Args:
            use_ldm_augmentation: 是否使用LDM增强
            test_name: 测试名称，用于日志记录
        """
        if use_ldm_augmentation:
            test_name = test_name or "LDM增强测试"
            print(f"=== {test_name} ===")
            print(f"任务级微调步数: {getattr(self.args, 'task_finetune_steps', 0)}，学习率: {getattr(self.args, 'task_finetune_lr', 0.0)}")
            
            # 加载最佳LDM模型
            ldm_model_path = None
            standard_path = os.path.join(self.save_dir, f'{self.args.dataset_name}_ldm.pkl')
            if os.path.exists(standard_path):
                ldm_model_path = standard_path
                print("加载标准LDM模型")
            
            if ldm_model_path:
                try:
                    self.ldm.load_state_dict(torch.load(ldm_model_path, map_location=self.args.device, weights_only=True))
                    print(f"LDM模型加载成功: {os.path.basename(ldm_model_path)}")
                except Exception as e:
                    print(f"LDM模型加载失败: {e}，使用当前模型状态")
            else:
                print("未找到LDM模型文件，使用当前模型状态")
            
            self.model.eval()
            self.ldm.eval()
            num_augmented_samples = getattr(self.args, 'num_augmented_samples', 10)
        else:
            test_name = test_name or "原始Encoder测试"
            print(f"=== {test_name} ===")
            self.model.eval()
            num_augmented_samples = 0
        
        test_accs = []
        start_test_idx = 0
        
        # 计算总任务数
        total_tasks = (len(self.dataset.test_graphs) - self.args.K_shot * self.dataset.test_classes_num) // (self.args.N_way * self.args.query_size)
        
        # 创建进度条
        pbar = tqdm(total=total_tasks, desc=test_name)
        
        while start_test_idx < len(self.dataset.test_graphs) - self.args.K_shot * self.dataset.test_classes_num:
            test_acc = self._evaluate_one_task_with_ldm(start_test_idx, num_augmented_samples, start_test_idx=start_test_idx)
            test_accs.append(test_acc)
            
            # 根据测试类型记录不同的日志信息
            if self.logf is not None:
                if use_ldm_augmentation:
                    self.logf.write(f"任务起始索引 {start_test_idx} 微调后准确率: {test_acc:.4f}\n")
                else:
                    self.logf.write(f"任务起始索引 {start_test_idx} 原始准确率: {test_acc:.4f}\n")
            
            start_test_idx += self.args.N_way * self.args.query_size
            
            # 使用tqdm.write显示任务准确率
            task_num = len(test_accs)
            if use_ldm_augmentation:
                pbar.write(f"任务 {task_num}: 微调后准确率 = {test_acc:.4f}")
            else:
                pbar.write(f"任务 {task_num}: 原始准确率 = {test_acc:.4f}")
            
            # 更新进度条
            pbar.update(1)
            pbar.set_postfix({'Acc': f'{test_acc:.4f}'})
        
        pbar.close()
        
        mean_acc = sum(test_accs) / len(test_accs)
        std = np.array(test_accs).std()
        
        print(f'{test_name}准确率: {mean_acc:.4f} ± {std:.4f}')
        if self.logf is not None:
            if use_ldm_augmentation:
                self.logf.write(f'{test_name}准确率: {mean_acc:.4f} ± {std:.4f}\n')
            else:
                self.logf.write(f'{test_name}准确率: {mean_acc:.4f} ± {std:.4f}\n')
                
        return mean_acc, std
    

        
    def _evaluate_one_task_with_ldm(self, test_idx, num_augmented_samples, start_test_idx=None):
        """评估一个few-shot任务（使用LDM增强）
        
        Args:
            test_idx: 测试任务起始索引
            num_augmented_samples: 每个类别生成的增强样本数量
            start_test_idx: 全局测试起始索引，用于计算任务ID
        """
        self.model.eval()
        
        # 获取prompt embeddings
        if self.args.use_prompt:
            self.prompt.train()
            prompt_embeds = self.prompt()
        else:
            prompt_embeds = torch.zeros(self.args.num_token, self.args.node_fea_size).to(self.args.device)
            
        # 采样当前任务
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
        
        # 任务前备份LDM权重，避免任务间串扰
        _ldm_backup = deepcopy(self.ldm.state_dict())

        # 任务级微调：用支持集对LDM做少量步数的适配
        if getattr(self.args, 'task_finetune_steps', 20) > 0 and num_augmented_samples > 0:
            # 使用任务索引生成任务ID用于进度条显示
            task_id = f"task_{start_test_idx // (self.args.N_way * self.args.query_size)}"
            self._task_level_finetune(original_support_data, support_label, task_id=task_id)


        
        # 使用LDM生成增强的embeddings（仅在需要时）
        if num_augmented_samples > 0:
            augmented_embeddings = []
            augmented_labels = []
            
            # 为每个支持集样本生成增强样本
            for i in range(len(original_support_data)):
                # 使用支持集样本自己作为条件
                support_sample = original_support_data[i:i+1]  # [1, D]
                support_sample_label = support_label[i:i+1]    # [1]
                
                # 归一化条件
                condition = F.normalize(support_sample, dim=1)
                condition = condition.repeat(num_augmented_samples, 1)  # [num_aug, D]
                
                # 使用LDM生成新的embeddings
                with torch.no_grad():
                    generated_embeddings = self.ldm.sample(
                        shape=(num_augmented_samples, self.training_embeddings.shape[1]),
                        cond=condition,
                        control=condition  # 使用条件作为control信号
                    )  # [num_samples, D]
                    
                augmented_embeddings.append(generated_embeddings)
                augmented_labels.extend([support_sample_label.item()] * num_augmented_samples)
            
            # 合并原始和增强的embeddings
            if augmented_embeddings:
                all_augmented_embs = torch.cat(augmented_embeddings, dim=0)
                all_augmented_labels = torch.tensor(augmented_labels, device=self.args.device, dtype=torch.long)
                
                # 合并支持集
                enhanced_support_data = torch.cat([original_support_data, all_augmented_embs], dim=0)
                enhanced_support_labels = torch.cat([support_label, all_augmented_labels], dim=0)
            else:
                enhanced_support_data = original_support_data
                enhanced_support_labels = support_label
        else:
            # 无LDM增强，直接使用原始数据
            enhanced_support_data = original_support_data
            enhanced_support_labels = support_label
            
        # 训练分类器（使用增强的支持集）
        in_dim = self.model.sample_input_emb_size
        num_cls = self.args.N_way
        self.log = LogReg(in_dim, num_cls).to(self.args.device)  # 重置分类头
        
        # 使用统一的训练函数；无mixup时传入空参数
        empty_long = torch.empty(0, dtype=torch.long, device=self.args.device)
        empty_float = torch.empty(0, dtype=torch.float32, device=self.args.device)
        self._train_classifier(
            enhanced_support_data,
            None,
            enhanced_support_labels,
            empty_long,
            empty_long,
            empty_float
        )
        
        # 评估查询集（复用前面计算的prompt_embeds）
        query_current_sample_input_embs, _ = self.model.sample_input_GNN(
            [current_task], prompt_embeds, False
        )
        query_data = query_current_sample_input_embs.detach()
        
        # 获取查询集标签
        query_label = []
        for graphs in current_task['query_set']:
            query_label.append(np.array([graph.label for graph in graphs]))
        query_label = torch.LongTensor(np.hstack(query_label)).to(self.args.device)
        
        # 处理查询集长度（去除填充部分）
        query_len = query_label.shape[0]
        if current_task['append_count'] != 0:
            query_data = query_data[:query_len - current_task['append_count'], :]
            query_label = query_label[:query_len - current_task['append_count']]
            
        # 预测和计算准确率
        logits = self.log(query_data)
        preds = torch.argmax(logits, dim=1)
        acc = torch.sum(preds == query_label).float() / query_label.shape[0]
        
        # 恢复LDM权重，避免任务间串扰
        self.ldm.load_state_dict(_ldm_backup)
        
        return acc.cpu().numpy()
    
    def _task_level_finetune(self, support_data, support_labels, task_id=None):
        """任务级 LDM 微调：适配当前任务的简单微调
        
        Args:
            support_data: 支持集embeddings [S, D]
            support_labels: 支持集标签 [S]
            task_id: 任务ID，用于进度条显示
        """
        steps = int(getattr(self.args, 'task_finetune_steps', 20))
        if steps <= 0:
            return

        # 1) 冻结主干，仅训控制分支
        trainable = []
        for n, p in self.ldm.named_parameters():
            p.requires_grad = False
            if 'diffusion.controlnet' in n:
                p.requires_grad = True
                trainable.append(p)

        # 2) 自条件（与训练阶段一致：先归一化）
        with torch.no_grad():
            cond_per_sample = F.normalize(support_data, dim=1)
            
            # 构造类原型作为control信号（更稳定）
            uniq = torch.unique(support_labels)
            ns = F.normalize(support_data, dim=1)
            proto = {int(lbl): F.normalize(ns[support_labels==lbl].mean(0, keepdim=True), dim=1) for lbl in uniq}
            control_in = torch.cat([proto[int(lbl)] for lbl in support_labels.tolist()], dim=0)

        # 3) 优化器（仅本地使用）
        lr = float(getattr(self.args, 'task_finetune_lr', 1e-4))  # 降低学习率
        wd = float(getattr(self.args, 'task_finetune_weight_decay', 0.0))
        opt = torch.optim.AdamW(trainable, lr=lr, weight_decay=wd)

        self.ldm.train()

        drop_p   = float(getattr(self.args, 'task_finetune_cond_dropout', 0.05))  # 降低dropout
        grad_clip = float(getattr(self.args, 'task_finetune_grad_clip', 1.0))
        patience  = int(getattr(self.args, 'task_finetune_patience', 5))

        # 早停跟踪
        best_loss  = float('inf')
        best_state = None
        wait = 0

        pbar = tqdm(range(steps), desc=f"Task Finetune ({task_id})", position=1, leave=False)
        for step in pbar:
            # 单次构造 cond（只做你的一层dropout；loss里p_uncond=0.0避免二次置零）
            cond_in = cond_per_sample.clone()
            if drop_p > 0:
                mask = (torch.rand(cond_in.size(0), device=cond_in.device) < drop_p).float().unsqueeze(-1)
                cond_in = cond_in * (1 - mask)

            loss = self.ldm.loss(support_data, cond_in, p_uncond=0.0, control=control_in)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, grad_clip)
            opt.step()

            cur = loss.item()
            if cur < best_loss:
                best_loss = cur
                wait = 0
                # 记住最佳权重（轻量任务微调，state_dict拷贝成本很低）
                from copy import deepcopy
                best_state = deepcopy(self.ldm.state_dict())
            else:
                wait += 1
                
            # 日志：每一步都提交到wandb
            pbar.set_postfix({'Loss': f'{cur:.4f}'})
            if wandb.run is not None:
                wandb.log({f"task_{task_id}_loss": cur, f"task_{task_id}_step": step})

            if wait >= patience:
                break

        # 用最佳步的参数做后续采样/增强
        if best_state is not None:
            self.ldm.load_state_dict(best_state)
        self.ldm.eval()
        
    def _train_classifier(self, support_data, support_data_mixup, support_label,
                         support_label_mix_a, support_label_mix_b, weight):
        """训练线性分类器（与 baseline 一致：优化器包含 prompt）
        
        Args:
            support_data: 支持集数据
            support_data_mixup: Mixup增强数据（可能为None）
            support_label: 支持集标签
            support_label_mix_a: Mixup数据标签A
            support_label_mix_b: Mixup数据标签B
            weight: Mixup权重
        """
        self.log.train()
        
        # 与 baseline 相同：根据 use_prompt 选择是否一并优化 prompt
        if getattr(self.args, 'use_prompt', True):
            opt = torch.optim.SGD([{'params': self.log.parameters()}, {'params': self.prompt.parameters()}], lr=0.01)
        else:
            opt = torch.optim.SGD([{'params': self.log.parameters()}], lr=0.01)
            
        # 早停相关变量
        best_loss = 1e9
        wait = 0
        patience = 10
        
        for _ in range(500):
            opt.zero_grad()
            
            # 原始支持数据损失
            logits = self.log(support_data)
            loss_ori = self.xent(logits, support_label)
            
            # Mixup数据损失（仅在存在mixup数据时计算）
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
            loss_total = loss_ori + loss_mix + 0.1 * l2_reg
            
            loss_total.backward()
            opt.step()
            
            # 早停检查
            if loss_total < best_loss:
                best_loss = loss_total
                wait = 0
                torch.save(
                    self.log.state_dict(), 
                    os.path.join(self.save_dir, f'{self.args.dataset_name}_lr.pkl')
                )
            else:
                wait += 1
            if wait > patience:
                break
                
        # 加载最佳模型权重
        self.log.load_state_dict(torch.load(
            os.path.join(self.save_dir, f'{self.args.dataset_name}_lr.pkl'),
            weights_only=True
        ))
        self.log.eval()
        
    def visualize_prototype_generation(self, num_samples_per_prototype=50, save_path=None):
        """
        在LDM训练完成后，使用聚类原型作为条件生成样本并进行降维可视化
        
        Args:
            num_samples_per_prototype: 每个原型生成的样本数量
            save_path: 可视化图片保存路径，如果为None则显示图片
        """
        print("=== 开始原型引导生成和可视化 ===")
        
        if not hasattr(self, 'ldm'):
            print("❌ 错误：未找到LDM模型，请先运行 init_ldm_components()")
            return
        
        # 确保LDM处于评估模式
        self.ldm.eval()
        
        # 检查条件类型
        condition_type = getattr(self.args, 'condition_type', 'kmeans')
        
        # 根据条件类型决定生成策略
        if condition_type == 'kmeans':
            # 使用聚类原型生成
            if not hasattr(self, 'prototypes') or self.prototypes is None:
                print("❌ 错误：未找到聚类原型，请先运行 collect_training_embeddings()")
                return
                
            num_prototypes = self.prototypes.shape[0]
            print(f"找到 {num_prototypes} 个聚类原型")
            
            # 存储生成的样本
            generated_samples = []
            prototype_labels = []
            
            # 为每个原型生成样本
            with torch.no_grad():
                for proto_idx in range(num_prototypes):
                    print(f"正在为原型 {proto_idx + 1}/{num_prototypes} 生成 {num_samples_per_prototype} 个样本...")
                    
                    # 获取当前原型作为条件
                    prototype_condition = self.prototypes[proto_idx:proto_idx+1]  # [1, D]
                    
                    # 归一化条件
                    prototype_condition = F.normalize(prototype_condition, dim=1)
                    
                    # 重复条件以生成多个样本
                    repeated_condition = prototype_condition.repeat(num_samples_per_prototype, 1)  # [num_samples, D]
                    
                    # 使用LDM生成样本
                    generated_embeddings = self.ldm.sample(
                        shape=(num_samples_per_prototype, self.training_embeddings.shape[1]),
                        cond=repeated_condition,
                        control=repeated_condition  # 使用原型作为control信号
                    )  # [num_samples, D]
                    
                    # 存储生成的样本和对应的原型标签
                    generated_samples.append(generated_embeddings.cpu())
                    prototype_labels.extend([proto_idx] * num_samples_per_prototype)
            
            # 合并所有生成的样本
            all_generated = torch.cat(generated_samples, dim=0)  # [total_samples, D]
            prototype_labels = np.array(prototype_labels)
            
            # 创建包含三种点的单一可视化图
            self._visualize_three_types(
                training_embeddings=self.training_embeddings.cpu(),
                prototypes=self.prototypes.cpu(),
                generated_embeddings=all_generated,
                prototype_labels=prototype_labels,
                save_path=save_path
            )
            
        elif condition_type == 'self_conditioning':
            # 使用self-conditioning生成
            print("使用self-conditioning模式生成样本")
            
            # 在可视化时重新计算kmeans聚类，用于生成样本
            print("重新计算kmeans聚类用于可视化生成...")
            kmeans = KMeans(n_init=10, n_clusters=self.args.train_classes_num, random_state=42)
            cluster_labels = kmeans.fit_predict(self.training_embeddings.cpu().detach().numpy())
            visualization_prototypes = torch.tensor(
                kmeans.cluster_centers_, dtype=torch.float, device=self.args.device
            )
            
            # 使用聚类原型生成样本
            num_prototypes = visualization_prototypes.shape[0]
            print(f"可视化聚类完成，找到 {num_prototypes} 个原型")
            
            # 存储生成的样本
            generated_samples = []
            prototype_labels = []
            
            # 为每个原型生成样本
            with torch.no_grad():
                for proto_idx in range(num_prototypes):
                    print(f"正在为原型 {proto_idx + 1}/{num_prototypes} 生成 {num_samples_per_prototype} 个样本...")
                    
                    # 获取当前原型作为条件
                    prototype_condition = visualization_prototypes[proto_idx:proto_idx+1]  # [1, D]
                    
                    # 归一化条件
                    prototype_condition = F.normalize(prototype_condition, dim=1)
                    
                    # 重复条件以生成多个样本
                    repeated_condition = prototype_condition.repeat(num_samples_per_prototype, 1)  # [num_samples, D]
                    
                    # 使用LDM生成样本
                    generated_embeddings = self.ldm.sample(
                        shape=(num_samples_per_prototype, self.training_embeddings.shape[1]),
                        cond=repeated_condition,
                        control=repeated_condition  # 使用原型作为control信号
                    )  # [num_samples, D]
                    
                    # 存储生成的样本和对应的原型标签
                    generated_samples.append(generated_embeddings.cpu())
                    prototype_labels.extend([proto_idx] * num_samples_per_prototype)
            
            # 合并所有生成的样本
            all_generated = torch.cat(generated_samples, dim=0)  # [total_samples, D]
            prototype_labels = np.array(prototype_labels)
            
            # 创建包含三种点的单一可视化图
            self._visualize_three_types(
                training_embeddings=self.training_embeddings.cpu(),
                prototypes=visualization_prototypes.cpu(),
                generated_embeddings=all_generated,
                prototype_labels=prototype_labels,
                save_path=save_path
            )
            
        else:
            raise ValueError(f"不支持的condition_type: {condition_type}")
        
        print(f"总共生成了 {all_generated.shape[0]} 个样本")
        print("✅ 原型引导生成和可视化完成！")
        
        return all_generated, prototype_labels
    
    def _visualize_three_types(self, training_embeddings, prototypes, generated_embeddings, prototype_labels, save_path=None):
        """
        创建包含三种点的单一可视化图：训练集点、聚类原型点、生成的样本点
        
        Args:
            training_embeddings: 训练集embeddings [N, D]
            prototypes: 聚类原型 [K, D]
            generated_embeddings: 生成的embeddings [M, D]
            prototype_labels: 生成的样本对应的原型标签 [M]
            save_path: 保存路径，如果为None则显示图片
        """
        try:
            from sklearn.manifold import TSNE
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            print(f"正在进行TSNE降维可视化...")
            
            # 合并所有数据用于TSNE降维
            all_embeddings = torch.cat([training_embeddings, prototypes, generated_embeddings], dim=0)
            
            # 使用TSNE降维到2D
            tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(all_embeddings)-1))
            embeddings_2d = tsne.fit_transform(all_embeddings.numpy())
            
            # 分离不同数据类型的2D坐标
            n_training = len(training_embeddings)
            n_prototypes = len(prototypes)
            n_generated = len(generated_embeddings)
            
            training_2d = embeddings_2d[:n_training]
            prototypes_2d = embeddings_2d[n_training:n_training+n_prototypes]
            generated_2d = embeddings_2d[n_training+n_prototypes:]
            
            # 创建可视化图形
            plt.figure(figsize=(14, 10))
            
            # 定义颜色主题（在绘制之前先定义）
            colors = plt.cm.Set3(np.linspace(0, 1, len(prototypes)))
            
            # 1. 绘制训练集点（实心点，按标签使用不同颜色）
            # 计算每个训练样本对应的原型标签
            training_proto_labels = torch.argmin(torch.cdist(training_embeddings, prototypes), dim=1).numpy()
            
            for i in range(len(prototypes)):
                mask = training_proto_labels == i
                if mask.any():
                    plt.scatter(
                        training_2d[mask, 0], training_2d[mask, 1],
                        c=[colors[i]], alpha=0.6, s=25,
                        label=f'Training (P{i})', edgecolors='none', marker='o'
                    )
            
            # 2. 绘制聚类原型点（大星形，实心，不同颜色）
            for i, (proto_2d, color) in enumerate(zip(prototypes_2d, colors)):
                plt.scatter(
                    proto_2d[0], proto_2d[1],
                    c=[color], s=200, marker='*',
                    label=f'Prototype {i}', edgecolors='black', linewidth=1.5
                )
            
            # 3. 绘制生成的样本点（空心点，与对应原型使用相同颜色）
            for i in range(len(prototypes)):
                mask = prototype_labels == i
                if mask.any():
                    plt.scatter(
                        generated_2d[mask, 0], generated_2d[mask, 1],
                        c=[colors[i]], s=60, alpha=0.7,
                        label=f'Generated (P{i})', edgecolors=colors[i], linewidth=1.5,
                        facecolors='none', marker='o'
                    )
            
            # 设置图形标题和标签
            plt.title('LDM Prototype-Guided Generation: Training Data, Prototypes, and Generated Samples', 
                     fontsize=16, fontweight='bold')
            plt.xlabel('TSNE 1', fontsize=12)
            plt.ylabel('TSNE 2', fontsize=12)
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            
            # 保存或显示图片
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"可视化图片已保存到: {save_path}")
            else:
                plt.show()
            
            plt.close()
            
        except ImportError as e:
            print(f"❌ 可视化依赖缺失: {e}")
            print("请安装: pip install scikit-learn matplotlib seaborn")
        except Exception as e:
            print(f"❌ 可视化失败: {e}")
    
    def visualize_support_query_generation(self, num_samples_per_prototype=50, save_path=None):
        """
        在LDM训练完成后，可视化支持集、剩余测试数据和生成模型扩充的数据
        
        Args:
            num_samples_per_prototype: 每个原型生成的样本数量
            save_path: 可视化图片保存路径，如果为None则显示图片
        """
        print("=== 开始支持集、测试数据和生成数据的可视化 ===")
        
        if not hasattr(self, 'ldm'):
            print("❌ 错误：未找到LDM模型，请先运行 init_ldm_components()")
            return
        
        # 确保LDM处于评估模式
        self.ldm.eval()
        
        # 1. 获取支持集embeddings（所有任务都一样的）
        print("获取支持集embeddings...")
        support_task = self.dataset.sample_one_task(
            self.dataset.test_tasks, 
            np.array(list(range(self.dataset.test_classes_num))),
            K_shot=self.args.K_shot, 
            query_size=self.args.query_size,
            test_start_idx=0  # 从0开始，获取固定的支持集
        )
        
        # 获取prompt embeddings
        if self.args.use_prompt:
            self.prompt.eval()
            prompt_embeds = self.prompt()
        else:
            prompt_embeds = torch.zeros(self.args.num_token, self.args.node_fea_size).to(self.args.device)
        
        # 获取支持集embeddings
        support_embs, _ = self.model.sample_input_GNN([support_task], prompt_embeds, True)
        support_embeddings = support_embs.detach()
        
        # 获取支持集标签
        support_labels = []
        for graphs in support_task['support_set']:
            support_labels.append(np.array([graph.label for graph in graphs[:self.args.K_shot]]))
        support_labels = torch.LongTensor(np.hstack(support_labels)).to(self.args.device)
        
        # 调试信息：打印支持集详细信息
        print(f"支持集详细信息:")
        print(f"  - 支持集embeddings形状: {support_embeddings.shape}")
        print(f"  - 支持集标签形状: {support_labels.shape}")
        print(f"  - 支持集标签内容: {support_labels.tolist()}")
        print(f"  - 唯一标签: {torch.unique(support_labels).tolist()}")
        print(f"  - 每个标签的样本数:")
        for label in torch.unique(support_labels):
            count = (support_labels == label).sum().item()
            print(f"    标签 {label.item()}: {count} 个样本")
        
        # 2. 获取剩余测试数据embeddings
        print("获取剩余测试数据embeddings...")
        remaining_test_embeddings = []
        remaining_test_labels = []
        
        # 计算剩余测试数据的起始位置
        start_idx = self.args.K_shot * self.dataset.test_classes_num
        
        # 分批处理剩余测试数据
        batch_size = getattr(self.args, 'batch_size_for_embedding', 512)
        for i in range(start_idx, len(self.dataset.test_graphs), batch_size):
            end_idx = min(i + batch_size, len(self.dataset.test_graphs))
            batch_graphs = self.dataset.test_graphs[i:end_idx]
            
            # 创建临时任务结构
            temp_task = {'support_set': [], 'query_set': [batch_graphs]}
            temp_embs, _ = self.model.sample_input_GNN([temp_task], prompt_embeds, False)
            
            remaining_test_embeddings.append(temp_embs.detach())
            
            # 获取标签
            batch_labels = [graph.label for graph in batch_graphs]
            remaining_test_labels.extend(batch_labels)
        
        if remaining_test_embeddings:
            remaining_test_embeddings = torch.cat(remaining_test_embeddings, dim=0)
            remaining_test_labels = torch.LongTensor(remaining_test_labels).to(self.args.device)
        else:
            remaining_test_embeddings = torch.empty(0, support_embeddings.shape[1])
            remaining_test_labels = torch.empty(0, dtype=torch.long)
        
        # 3. 使用生成模型扩充数据
        print("使用生成模型扩充数据...")
        augmented_embeddings = []
        augmented_labels = []
        
        # 为每个支持集样本生成增强样本
        for i in range(len(support_embeddings)):
            # 使用支持集样本自己作为条件
            support_sample = support_embeddings[i:i+1]  # [1, D]
            support_sample_label = support_labels[i:i+1]    # [1]
            
            # 归一化条件
            condition = F.normalize(support_sample, dim=1)
            condition = condition.repeat(num_samples_per_prototype, 1)  # [num_aug, D]
            
            # 使用LDM生成新的embeddings
            with torch.no_grad():
                generated_embeddings = self.ldm.sample(
                    shape=(num_samples_per_prototype, support_embeddings.shape[1]),
                    cond=condition,
                    control=condition  # 使用条件作为control信号
                )  # [num_samples, D]
                
            augmented_embeddings.append(generated_embeddings)
            augmented_labels.extend([support_sample_label.item()] * num_samples_per_prototype)
        
        if augmented_embeddings:
            all_augmented_embs = torch.cat(augmented_embeddings, dim=0)
            all_augmented_labels = torch.LongTensor(augmented_labels).to(self.args.device)
        else:
            all_augmented_embs = torch.empty(0, support_embeddings.shape[1])
            all_augmented_labels = torch.empty(0, dtype=torch.long)
        
        # 4. 创建可视化
        print("创建可视化...")
        self._visualize_support_query_generation(
            support_embeddings=support_embeddings.cpu(),
            support_labels=support_labels.cpu(),
            remaining_test_embeddings=remaining_test_embeddings.cpu(),
            remaining_test_labels=remaining_test_labels.cpu(),
            generated_embeddings=all_augmented_embs.cpu(),
            generated_labels=all_augmented_labels.cpu(),
            save_path=save_path
        )
        
        print(f"✅ 支持集、测试数据和生成数据的可视化完成！")
        print(f"支持集样本数: {len(support_embeddings)}")
        print(f"剩余测试样本数: {len(remaining_test_embeddings)}")
        print(f"生成样本数: {len(all_augmented_embs)}")
        
        return support_embeddings, remaining_test_embeddings, all_augmented_embs
    
    def _visualize_support_query_generation(self, support_embeddings, support_labels, 
                                          remaining_test_embeddings, remaining_test_labels,
                                          generated_embeddings, generated_labels, save_path=None):
        """
        创建包含四种数据的可视化图：支持集、剩余测试数据、生成数据
        
        Args:
            support_embeddings: 支持集embeddings [S, D]
            support_labels: 支持集标签 [S]
            remaining_test_embeddings: 剩余测试数据embeddings [R, D]
            remaining_test_labels: 剩余测试数据标签 [R]
            generated_embeddings: 生成的embeddings [G, D]
            generated_labels: 生成的样本标签 [G]
            save_path: 保存路径，如果为None则显示图片
        """
        try:
            from sklearn.manifold import TSNE
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            print(f"正在进行TSNE降维可视化...")
            
            # 合并所有数据用于TSNE降维
            all_embeddings = torch.cat([support_embeddings, remaining_test_embeddings, generated_embeddings], dim=0)
            
            # 使用TSNE降维到2D
            tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(all_embeddings)-1))
            embeddings_2d = tsne.fit_transform(all_embeddings.numpy())
            
            # 分离不同数据类型的2D坐标
            n_support = len(support_embeddings)
            n_remaining = len(remaining_test_embeddings)
            n_generated = len(generated_embeddings)
            
            support_2d = embeddings_2d[:n_support]
            remaining_2d = embeddings_2d[n_support:n_support+n_remaining]
            generated_2d = embeddings_2d[n_support+n_remaining:]
            
            # 创建可视化图形
            plt.figure(figsize=(16, 12))
            
            # 定义颜色主题
            unique_labels = torch.unique(torch.cat([support_labels, remaining_test_labels, generated_labels]))
            colors = plt.cm.Set3(np.linspace(0, 1, len(unique_labels)))
            color_map = {label.item(): colors[i] for i, label in enumerate(unique_labels)}
            
            # 1. 绘制支持集点（实心圆点，较大，按标签使用不同颜色）
            print(f"绘制支持集点: 总共 {len(support_2d)} 个点")
            for i, (emb_2d, label) in enumerate(zip(support_2d, support_labels)):
                color = color_map[label.item()]
                plt.scatter(
                    emb_2d[0], emb_2d[1],
                    c=[color], alpha=0.8, s=100,
                    label=f'Support (L{label.item()})' if i == 0 or label != support_labels[i-1] else "",
                    edgecolors='black', linewidth=1, marker='o'
                )
                # 打印每一个点的详细信息
                print(f"  点 {i:2d}: 位置({emb_2d[0]:8.3f}, {emb_2d[1]:8.3f}), 标签{label.item()}")
            
            # 2. 绘制剩余测试数据点（实心方块，中等大小，按标签使用不同颜色）
            for i, (emb_2d, label) in enumerate(zip(remaining_2d, remaining_test_labels)):
                color = color_map[label.item()]
                plt.scatter(
                    emb_2d[0], emb_2d[1],
                    c=[color], alpha=0.6, s=60,
                    label=f'Test (L{label.item()})' if i == 0 or label != remaining_test_labels[i-1] else "",
                    edgecolors='none', marker='s'
                )
            
            # 3. 绘制生成数据点（空心圆点，较小，按标签使用不同颜色）
            for i, (emb_2d, label) in enumerate(zip(generated_2d, generated_labels)):
                color = color_map[label.item()]
                plt.scatter(
                    emb_2d[0], emb_2d[1],
                    c=[color], s=40, alpha=0.7,
                    label=f'Generated (L{label.item()})' if i == 0 or label != generated_labels[i-1] else "",
                    edgecolors=color, linewidth=1.5,
                    facecolors='none', marker='o'
                )
            
            # 设置图形标题和标签
            plt.title('LDM Data Visualization: Support Set, Test Data, and Generated Samples', 
                     fontsize=16, fontweight='bold')
            plt.xlabel('TSNE 1', fontsize=12)
            plt.ylabel('TSNE 2', fontsize=12)
            
            # 创建图例（去除重复项）
            handles, labels = plt.gca().get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            plt.legend(by_label.values(), by_label.keys(), 
                      bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
            
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            
            # 保存或显示图片
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"可视化图片已保存到: {save_path}")
            else:
                plt.show()
            
            plt.close()
            
        except ImportError as e:
            print(f"❌ 可视化依赖缺失: {e}")
            print("请安装: pip install scikit-learn matplotlib seaborn")
        except Exception as e:
            print(f"❌ 可视化失败: {e}")
            import traceback
            traceback.print_exc()
    

    
   




    

