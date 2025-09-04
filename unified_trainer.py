
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
            self.args.time_steps,
            self.args.beta_start,
            self.args.beta_end
        ).to(self.args.device)
        
        self.ldm_optimizer = torch.optim.AdamW(
            self.ldm.parameters(),
            lr=self.args.learning_rate_ldm,
            weight_decay=self.args.weight_decay_ldm
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
    
        # 根据参数选择条件类型
        condition_type = self.args.condition_type  # 默认为self_labeling
        
        if condition_type == 'self_labeling':
            # 使用样本自己作为条件
            self.conditions = self.training_embeddings.clone()
            # 注意：这里不设置self.prototypes，但在可视化时会重新计算kmeans
            
        elif condition_type == 'class_proto':
            # 训练阶段：使用K-means聚类获取类别原型
            # 先归一化embeddings，再进行聚类
            normalized_embeddings = F.normalize(self.training_embeddings, dim=1)
            kmeans = KMeans(n_init=10, n_clusters=self.args.train_classes_num, random_state=42)
            cluster_labels = kmeans.fit_predict(normalized_embeddings.cpu().detach().numpy())
            prototypes = torch.tensor(
                kmeans.cluster_centers_, dtype=torch.float, device=self.args.device
            )
            # 对聚类中心再次归一化
            prototypes = F.normalize(prototypes, dim=1)
            self.prototypes = prototypes

            # 计算每个样本的类别 Prototype 作为条件
            cluster_labels = torch.tensor(cluster_labels, dtype=torch.long, device=self.args.device)
            self.conditions = prototypes[cluster_labels]
            
            print(f"训练阶段：K-Means聚类完成，prototypes形状: {prototypes.shape}")
            
        else:
            raise ValueError(f"不支持的condition_type: {condition_type}，支持的类型: self_labeling, class_proto")
        
        # 对条件进行归一化
        self.conditions = F.normalize(self.conditions, dim=1)
        
    def train_ldm(self):
        """训练LDM，基于encoder的embeddings"""
        print("=== 开始训练LDM ===")
        
        # 根据条件类型获取条件数据
        conditions = self.conditions
        condition_type = self.args.condition_type
        
        print(f"训练数据: {self.training_embeddings.shape[0]} 个embeddings")
        print(f"条件数据: {conditions.shape[0]} 个条件")
        
        decay = float(self.args.ldm_ema_decay)
        check_interval = int(self.args.ldm_es_interval)
        ema_loss = None
        best_ema = float('inf')
        patience_count = 0
        patience = self.args.patience_ldm
        
        # 创建进度条
        pbar = tqdm(range(1, self.args.num_epochs_ldm + 1), desc="LDM Training")
        
        for epoch in pbar:
            # 随机打乱数据
            perm = torch.randperm(self.training_embeddings.shape[0])
            shuffled_embeddings = self.training_embeddings[perm]
            shuffled_conditions = conditions[perm]
            
            epoch_loss = 0
            batch_size = self.args.ldm_batch_size
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
        


        
    def test_model(self, num_augmented_samples=0, refine_support_before_training=False, ldm_model=None, test_name=None):
        """统一的测试函数"""
        # 统一命名
        test_name = test_name or ("LDM增强测试" if num_augmented_samples > 0 else "原始Encoder测试")
        self.model.eval()

        # 是否需要LDM：当需要refine或需要生成增强时
        need_ldm = bool(refine_support_before_training or (num_augmented_samples > 0))

        # 根据 refine/augmentation 情况，统一打印更清晰的评估模式
        if need_ldm:
            if refine_support_before_training and num_augmented_samples == 0:
                mode_label = "仅Refine评估（无增强采样）"
            elif num_augmented_samples > 0:
                mode_label = f"LDM增强评估（每样本生成{num_augmented_samples}）"
            else:
                mode_label = "LDM评估"
        else:
            mode_label = "原始Encoder评估（无LDM）"

        print(f"=== {mode_label} ===")
        
        test_accs = []
        start_test_idx = 0
        
        # 计算总任务数
        total_tasks = (len(self.dataset.test_graphs) - self.args.K_shot * self.args.test_classes_num) // (self.args.N_way * self.args.query_size)
        
        # 创建进度条
        pbar = tqdm(total=total_tasks, desc=test_name)
        
        # 确保只处理完整的任务，跳过不完整的最后一个任务
        end_test_idx = start_test_idx + total_tasks * (self.args.N_way * self.args.query_size)
        
        while start_test_idx < end_test_idx:
            test_acc = self._evaluate_one_task_with_ldm(start_test_idx, num_augmented_samples, ldm_model, refine_support_before_training, start_test_idx=start_test_idx)
            test_accs.append(test_acc)
            
            # 根据测试类型记录不同的日志信息
            if self.logf is not None:
                self.logf.write(f"任务起始索引 {start_test_idx} {mode_label} 准确率: {test_acc:.4f}\n")
            
            start_test_idx += self.args.N_way * self.args.query_size
            
            # 使用tqdm.write显示任务准确率
            task_num = len(test_accs)
            pbar.write(f"任务 {task_num}: {mode_label} = {test_acc:.4f}")
            
            # 更新进度条
            pbar.update(1)
            pbar.set_postfix({'Acc': f'{test_acc:.4f}'})
        
        pbar.close()
        
        mean_acc = sum(test_accs) / len(test_accs)
        std = np.array(test_accs).std()
        
        print(f'{test_name}准确率: {mean_acc:.4f} ± {std:.4f}')
        if self.logf is not None:
            if num_augmented_samples > 0:
                self.logf.write(f'{test_name}准确率: {mean_acc:.4f} ± {std:.4f}\n')
            else:
                self.logf.write(f'{test_name}准确率: {mean_acc:.4f} ± {std:.4f}\n')
                
        return mean_acc, std
    

        
    def _evaluate_one_task_with_ldm(self, test_idx, num_augmented_samples, ldm_model, refine_support_before_training, start_test_idx=None):
        """评估一个few-shot任务"""
        self.model.eval()
        
        # 获取prompt embeddings（评估期不使用 prompt，统一零向量）
        prompt_embeds = torch.zeros(self.args.num_token, self.args.node_fea_size).to(self.args.device)
            
        # 采样当前任务
        first_N_class_sample = np.array(list(range(self.dataset.test_classes_num)))
        current_task = self.dataset.sample_one_task(
            self.dataset.test_tasks, first_N_class_sample,
            K_shot=self.args.K_shot, query_size=self.args.query_size,
            test_start_idx=test_idx
        )
        
        # 获取支持集embeddings并处理维度
        support_current_sample_input_embs, _ = self.model.sample_input_GNN([current_task], prompt_embeds, True)
        total_support_samples = self.args.K_shot + self.args.gen_test_num
        
        # 重塑支持集数据：只取前K_shot个样本
        support_data = support_current_sample_input_embs.reshape(
            self.args.N_way, total_support_samples, self.model.sample_input_emb_size
        )[:, :self.args.K_shot, :].reshape(
            self.args.N_way * self.args.K_shot, self.model.sample_input_emb_size
        ).detach()
        
        # 获取支持集标签
        support_label = []
        for graphs in current_task['support_set']:
            support_label.append(np.array([graph.label for graph in graphs[:self.args.K_shot]]))
        support_label = torch.LongTensor(np.hstack(support_label)).to(self.args.device)
        
        # 第一步：refine支持集（如果启用）
        if refine_support_before_training:
            support_data_for_enhancement = self.refine_embeddings_with_ldm(
                embeddings=support_data,
                labels=support_label,
                ldm_model=ldm_model,
                alpha=self.args.refine_alpha,
                use_slerp=self.args.refine_use_slerp,
                batch_size=self.args.ldm_batch_size
            )
        else:
            support_data_for_enhancement = support_data

        # 扩充支持集
        enhanced_support_data, enhanced_support_labels = self.generate_augmented_embeddings(
            embeddings=support_data_for_enhancement,
            labels=support_label,
            num_to_generate=num_augmented_samples,
            ldm_model=ldm_model,
            device=self.args.device,
            condition_type=self.args.condition_type
        )

        # 训练分类器
        self.log = LogReg(self.model.sample_input_emb_size, self.args.N_way).to(self.args.device)
        empty_tensor = torch.empty(0, dtype=torch.long, device=self.args.device)
        self._train_classifier(
            enhanced_support_data, None, enhanced_support_labels,
            empty_tensor, empty_tensor, torch.empty(0, dtype=torch.float32, device=self.args.device)
        )
        
        # 评估查询集
        query_current_sample_input_embs, _ = self.model.sample_input_GNN([current_task], prompt_embeds, False)
        query_data = query_current_sample_input_embs.reshape(
            self.args.N_way, self.args.query_size, self.model.sample_input_emb_size
        ).reshape(self.args.N_way * self.args.query_size, self.model.sample_input_emb_size).detach()
        
        # 获取查询集标签
        query_label = []
        for graphs in current_task['query_set']:
            query_label.append(np.array([graph.label for graph in graphs]))
        query_label = torch.LongTensor(np.hstack(query_label)).to(self.args.device)
        
        # 处理填充数据
        if current_task['append_count'] != 0:
            query_len = query_label.shape[0]
            query_data = query_data[:query_len - current_task['append_count'], :]
            query_label = query_label[:query_len - current_task['append_count']]
        
        # 预测和计算准确率
        logits = self.log(query_data)
        preds = torch.argmax(logits, dim=1)
        acc = torch.sum(preds == query_label).float() / query_label.shape[0]
        
        return acc.cpu().numpy()

    def _task_level_finetune(self, support_data, support_labels, task_id=None):
        """任务级 LDM 微调：适配当前任务的简单微调
        
        Args:
            support_data: 支持集embeddings [S, D]
            support_labels: 支持集标签 [S]
            task_id: 任务ID，用于进度条显示
        """
        steps = int(self.args.task_finetune_steps)
        if steps <= 0:
            return

        # 1) 冻结主干，仅训控制分支
        trainable = []
        for n, p in self.ldm.named_parameters():
            p.requires_grad = False
            if 'diffusion.controlnet' in n:
                p.requires_grad = True
                trainable.append(p)

        # 2) 根据条件类型选择条件策略
        condition_type = self.args.condition_type
        
        with torch.no_grad():
            if condition_type == 'class_proto':
                # 测试阶段：使用真实标签聚类生成类别原型
                unique_labels = torch.unique(support_labels)
                label_prototypes = {}
                
                # 为每个类别计算原型
                for label in unique_labels:
                    label_mask = (support_labels == label)
                    if label_mask.sum() > 0:
                        label_embs = support_data[label_mask]
                        # 先归一化，再均值，再归一化
                        normalized_embs = F.normalize(label_embs, dim=1)
                        prototype = F.normalize(normalized_embs.mean(dim=0, keepdim=True), dim=1)
                        label_prototypes[label.item()] = prototype
                
                # 为每个样本分配对应的类别原型
                cond_per_sample = torch.zeros_like(support_data)
                for i, label in enumerate(support_labels):
                    label_key = label.item()
                    if label_key in label_prototypes:
                        cond_per_sample[i] = label_prototypes[label_key]
                    else:
                        # 如果没有原型，使用样本自己
                        cond_per_sample[i] = F.normalize(support_data[i:i+1], dim=1).squeeze(0)
                
                # 控制信号使用类别原型
                control_in = cond_per_sample.clone()
                
            else:
                # 其他条件类型：使用样本自己作为条件
                cond_per_sample = F.normalize(support_data, dim=1)
                control_in = cond_per_sample.clone()

        # 3) 优化器（仅本地使用）
        lr = float(self.args.task_finetune_lr)  # 降低学习率
        wd = float(self.args.task_finetune_weight_decay)
        opt = torch.optim.AdamW(trainable, lr=lr, weight_decay=wd)

        self.ldm.train()

        drop_p   = float(self.args.task_finetune_cond_dropout)  # 降低dropout
        grad_clip = float(self.args.task_finetune_grad_clip)
        patience  = int(self.args.task_finetune_patience)

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
        best_state = None
        wait = 0
        patience = 100
        
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
                # 直接在内存中保存最佳权重状态
                best_state = deepcopy(self.log.state_dict())
            else:
                wait += 1
            if wait > patience:
                break
                
        # 加载内存中的最佳权重
        if best_state is not None:
            self.log.load_state_dict(best_state)
        self.log.eval()
        
 
    def generate_augmented_embeddings(self, embeddings: torch.Tensor, labels: torch.Tensor,
                                      num_to_generate: int, *, ldm_model, device, condition_type: str):
        """使用给定的 LDM 模型对输入嵌入进行扩充。
        - embeddings: [N, D]
        - labels:     [N]
        - num_to_generate: 每类/每样本生成的样本数（由 condition_type 决定粒度）
        - ldm_model:  采样用的 LDM 实例
        - device:     目标设备
        - condition_type: 'class_proto' 或 'self_labeling'
        返回 (enhanced_embeddings, enhanced_labels)。当 num_to_generate<=0 原样返回。
        """
        if num_to_generate <= 0:
            return embeddings, labels

        embedding_dim = embeddings.shape[1]

        gen_batches: list[torch.Tensor] = []
        gen_labels: list[int] = []

        if condition_type == 'class_proto':
            prototypes = {}
            for cls in torch.unique(labels):
                mask = (labels == cls)
                if mask.any():
                    z = embeddings[mask]
                    proto = F.normalize(F.normalize(z, dim=1).mean(dim=0, keepdim=True), dim=1)
                    prototypes[int(cls.item())] = proto
            for cls_id, proto in prototypes.items():
                cond = proto.repeat(num_to_generate, 1)
                with torch.no_grad():
                    z_gen = ldm_model.sample(shape=(num_to_generate, embedding_dim), cond=cond, control=cond)
                gen_batches.append(z_gen)
                gen_labels.extend([cls_id] * num_to_generate)
        else:
            for i in range(len(embeddings)):
                cond = F.normalize(embeddings[i:i+1], dim=1).repeat(num_to_generate, 1)
                with torch.no_grad():
                    z_gen = ldm_model.sample(shape=(num_to_generate, embedding_dim), cond=cond, control=cond)
                gen_batches.append(z_gen)
                gen_labels.extend([int(labels[i].item())] * num_to_generate)

        if not gen_batches:
            return embeddings, labels

        gen_all = torch.cat(gen_batches, dim=0)
        gen_y = torch.tensor(gen_labels, device=device, dtype=torch.long)
        return torch.cat([embeddings, gen_all], dim=0), torch.cat([labels, gen_y], dim=0)
    

    def refine_embeddings_with_ldm(
        self,
        embeddings: torch.Tensor,            # [S, D] 支持集/任意待修饰嵌入
        labels: torch.Tensor,                # [S]    对应标签（>=0）
        ldm_model,                           # LDM模型实例
        alpha: float = 0.7,                 # 修饰强度(保留度)：alpha=0.7 => 70%原始 + 30%生成
        use_slerp: bool = False,             # True: 球面插值；False: 线性插值
        batch_size: int = 64                # 采样mini-batch大小
    ) -> torch.Tensor:
        """
        仅做"支持集去噪/修饰"，不做可视化、不改动任务结构。
        条件使用【类原型的球面均值】（先归一化再均值再归一化），与对比学习的余弦几何一致。
        返回 refined_embeddings（不做归一化，由调用方根据需要决定）。
        
        Args:
            embeddings: 输入嵌入 [S, D]
            labels: 对应标签 [S]
            ldm_model: LDM模型实例
            alpha: 修饰强度，默认0.7
            use_slerp: 是否使用球面插值，默认False
            batch_size: 批处理大小，默认64
        """
        device = embeddings.device
        S, D = embeddings.size()

        # 过滤非法标签（如 <0），保持原样
        valid_mask = labels >= 0
        refined = embeddings.clone()

        # 计算"原始原型"（球面均值：norm→mean→norm）
        classes = torch.unique(labels[valid_mask])
        protos = {}
        for c in classes:
            m = (labels == c)
            if m.any():
                zc = F.normalize(embeddings[m], dim=1)
                mu = F.normalize(zc.mean(dim=0, keepdim=True), dim=1)
                protos[int(c.item())] = mu

        # slerp（可选）
        def _slerp(z0: torch.Tensor, z1: torch.Tensor, t: float, eps: float = 1e-7):
            z0 = F.normalize(z0, dim=1); z1 = F.normalize(z1, dim=1)
            cos = (z0 * z1).sum(dim=1, keepdim=True).clamp(-1 + eps, 1 - eps)
            theta = torch.acos(cos)
            sin = torch.sin(theta).clamp_min(eps)
            return (torch.sin((1 - t) * theta) / sin) * z0 + (torch.sin(t * theta) / sin) * z1

        # 按类别成组采样并修饰
        for c, proto in protos.items():
            idx = torch.nonzero((labels == c), as_tuple=False).squeeze(1)
            if idx.numel() == 0:
                continue

            cond_full = proto.repeat(idx.numel(), 1)  # [Nc, D]
            gens = []
            for i in range(0, idx.numel(), batch_size):
                cond_b = cond_full[i:i+batch_size]
                gen_b = ldm_model.sample(shape=(cond_b.size(0), D), cond=cond_b, control=cond_b)
                gens.append(gen_b)
            Gc = torch.cat(gens, dim=0)               # [Nc, D]
            Zc = embeddings.index_select(0, idx)      # [Nc, D]

            if use_slerp:
                Rc = _slerp(Zc, Gc, t=1.0 - alpha)    # 更贴近余弦几何
            else:
                Rc = alpha * Zc + (1.0 - alpha) * Gc  # 线性插值

            refined.index_copy_(0, idx, Rc)

        # 非法标签样本保持原样（已在 refined.clone()）
        return refined
        
    def visualize_refine_aug(self, save_path=None):
        """Refine → Augment (from refined) → Visualize on one t-SNE figure."""
        import numpy as np
        import matplotlib.pyplot as plt
        from sklearn.manifold import TSNE
        import torch
        import torch.nn.functional as F

        assert self.ldm is not None, "LDM not initialized."
        self.ldm.eval()

        # 固定抽一个支持集任务
        task = self.dataset.sample_one_task(
            self.dataset.test_tasks,
            np.arange(self.dataset.test_classes_num),
            K_shot=self.args.K_shot,
            query_size=self.args.query_size,
            test_start_idx=0
        )

        # 取支持集嵌入与标签（评估期不使用prompt，置零）
        prompt = torch.zeros(self.args.num_token, self.args.node_fea_size, device=self.args.device)
        z_sup, _ = self.model.sample_input_GNN([task], prompt, True)
        z_sup = z_sup.detach()
        y_sup = torch.LongTensor(np.hstack([[g.label for g in gs] for gs in task['support_set']])).to(self.args.device)

        # refine 支持集（一次）
        z_ref = self.refine_embeddings_with_ldm(
            embeddings=z_sup,
            labels=y_sup,
            ldm_model=self.ldm,
            alpha=float(self.args.refine_alpha),
            use_slerp=bool(self.args.refine_use_slerp),
            batch_size=int(self.args.ldm_batch_size)
        )

        # 检查：若 refine 前后有样本嵌入未变化，打印告警
        with torch.no_grad():
            same_mask = torch.isclose(z_ref, z_sup, rtol=1e-5, atol=1e-8).all(dim=1)
            num_same = int(same_mask.sum().item())
            if num_same > 0:
                idx_same = torch.nonzero(same_mask, as_tuple=False).squeeze(1).cpu().tolist()
                print(f"[Warning] Refine后仍未变化的支持集样本数: {num_same} / {z_sup.size(0)}，示例索引: {idx_same[:10]}")

        # 基于 refined 扩充若干（仅可视化）
        num_aug = int(self.args.num_augmented_samples)
        if num_aug > 0:
            z_enh, y_enh = self.generate_augmented_embeddings(
                embeddings=z_ref.detach(),
                labels=y_sup,
                num_to_generate=num_aug,
                ldm_model=self.ldm,
                device=self.args.device,
                condition_type=self.args.condition_type
            )
            z_aug = z_enh[len(z_ref):]
            y_aug = y_enh[len(y_sup):]
        else:
            z_aug = torch.empty(0, z_ref.shape[1], device=self.args.device)
            y_aug = torch.empty(0, dtype=torch.long, device=self.args.device)

        # 原型（norm→mean→norm）
        def _protos(embs: torch.Tensor, labels: torch.Tensor):
            ps, ids = [], []
            for c in torch.unique(labels):
                m = (labels == c)
                if m.any():
                    p = F.normalize(F.normalize(embs[m], dim=1).mean(0, keepdim=True), dim=1)
                    ps.append(p); ids.append(int(c.item()))
            return (torch.cat(ps, 0) if len(ps) else torch.empty(0, embs.size(1), device=embs.device)), ids

        p_orig, ids = _protos(z_sup, y_sup)
        p_ref, _    = _protos(z_ref, y_sup)

        # 仅可视化数据（归一化+CPU）
        z_sup_v = F.normalize(z_sup, dim=1).cpu()
        z_ref_v = F.normalize(z_ref, dim=1).cpu()
        z_aug_v = (F.normalize(z_aug, dim=1).cpu()
                if z_aug.numel() > 0 else torch.empty(0, z_ref.shape[1]))
        y_aug_v = (y_aug.cpu().numpy() if y_aug.numel() > 0 else np.zeros(0, dtype=int))
        protos_v = (torch.cat([p_orig, p_ref], 0).cpu() if p_orig.numel() > 0 else torch.empty(0, z_ref.shape[1]))
        proto_tags = [f"orig_{i}" for i in ids] + [f"refined_{i}" for i in ids]

        # t-SNE
        parts, sizes = [z_sup_v, z_ref_v], [len(z_sup_v), len(z_ref_v)]
        if len(z_aug_v) > 0: parts.append(z_aug_v); sizes.append(len(z_aug_v))
        if len(protos_v) > 0: parts.append(protos_v); sizes.append(len(protos_v))
        X = torch.cat(parts, 0)
        n = len(X)
        perp = max(2, min(30, max(5, n // 3)))
        X2 = TSNE(n_components=2, random_state=42, perplexity=min(perp, n - 1)).fit_transform(X.numpy())

        s0 = sizes[0]; s1 = s0 + sizes[1]; idx = s1
        sup2 = X2[:s0]; ref2 = X2[s0:s1]
        if len(z_aug_v) > 0:
            s2 = idx + sizes[2]; aug2 = X2[idx:s2]; idx = s2
        else:
            aug2 = np.zeros((0, 2))
        if len(protos_v) > 0:
            pr2 = X2[idx:idx + sizes[-1]]
            orig_pr2, ref_pr2 = pr2[:len(ids)], pr2[len(ids):]
        else:
            orig_pr2 = ref_pr2 = np.zeros((0, 2))

        # 颜色（使用对比更强的tab20）
        uniq = torch.unique(y_sup.cpu()).tolist()
        _palette = plt.cm.tab20(np.linspace(0, 1, 20))
        cmap = {int(lbl): _palette[i % len(_palette)] for i, lbl in enumerate(uniq)}

        # 画图
        plt.figure(figsize=(16, 12))
        y_np = y_sup.cpu().numpy()
        for c in uniq:
            m = (y_np == c)
            if m.any():
                plt.scatter(sup2[m,0], sup2[m,1], c=[cmap[c]], s=60, alpha=0.9, edgecolors='black', linewidth=0.8, label=f'Orig C{c}')
                plt.scatter(ref2[m,0], ref2[m,1], c=[cmap[c]], s=80, alpha=0.7, facecolors='none', edgecolors=cmap[c], linewidth=1.6, label=f'Refined C{c}')

        if len(aug2) > 0:
            for c in uniq:
                m_aug = (y_aug_v == c)
                if m_aug.any():
                    plt.scatter(aug2[m_aug,0], aug2[m_aug,1], c=[cmap[c]], s=50, alpha=0.6, facecolors='none', edgecolors=cmap[c], linewidth=1.2, marker='^', label=f'Aug C{c}')

        for i in range(len(sup2)):
            plt.plot([sup2[i,0], ref2[i,0]], [sup2[i,1], ref2[i,1]], 'k-', alpha=0.25, linewidth=0.5)

        for i in range(len(orig_pr2)):
            c = uniq[i] if i < len(uniq) else uniq[-1]
            plt.scatter(orig_pr2[i,0], orig_pr2[i,1], c=[cmap[c]], s=240, marker='*', edgecolors='black', linewidth=1.8, label=f'Proto-Orig {c}')
        for i in range(len(ref_pr2)):
            c = uniq[i] if i < len(uniq) else uniq[-1]
            plt.scatter(ref_pr2[i,0], ref_pr2[i,1], c=[cmap[c]], s=280, marker='*', facecolors='none', edgecolors=cmap[c], linewidth=2.2, label=f'Proto-Ref {c}')

        plt.title('Refine → Augment (Support)', fontsize=16)
        plt.xlabel('t-SNE 1'); plt.ylabel('t-SNE 2')
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys(), bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
        plt.grid(True, alpha=0.3); plt.tight_layout()
        if save_path: plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else: plt.show()
        plt.close()

        # ===== 新增：并排绘制两张图（原始支持集+查询集；refine后支持集+查询集）并保存为一个PNG =====
        # 计算同一任务的查询集嵌入与标签
        prompt_q = torch.zeros(self.args.num_token, self.args.node_fea_size, device=self.args.device)
        q_embs, _ = self.model.sample_input_GNN([task], prompt_q, False)
        q_embs = q_embs.detach()
        q_labels = torch.LongTensor(np.hstack([[g.label for g in gs] for gs in task['query_set']])).to(self.args.device)

        # 若需要，对查询集也做与支持集一致的refine以便对比（右图使用refine后的支持集，但查询集保持原始以体现对比）
        # 这里按需求：右图仅替换支持集为z_ref，查询集保持原始q_embs

        # 归一化
        z_sup_norm = F.normalize(z_sup, dim=1).cpu()
        z_ref_norm = F.normalize(z_ref, dim=1).cpu()
        z_q_norm = F.normalize(q_embs, dim=1).cpu()

        # 统一做一次TSNE，保证两张子图坐标一致性（原/精修 支持 + 原始查询）
        parts_sq = [z_sup_norm, z_ref_norm, z_q_norm]
        sizes_sq = [len(z_sup_norm), len(z_ref_norm), len(z_q_norm)]
        X_sq = torch.cat(parts_sq, 0)
        n_sq = len(X_sq)
        perp_sq = max(2, min(30, max(5, n_sq // 3)))
        X2_sq = TSNE(n_components=2, random_state=42, perplexity=min(perp_sq, n_sq - 1)).fit_transform(X_sq.numpy())

        s_sup = sizes_sq[0]
        s_ref = sizes_sq[1]
        s_q   = sizes_sq[2]
        sup2_sq = X2_sq[:s_sup]
        ref2_sq = X2_sq[s_sup:s_sup+s_ref]
        qry2_sq = X2_sq[s_sup+s_ref:s_sup+s_ref+s_q]

        # 颜色映射（基于支持+查询的标签全集，使用tab20提高对比度）
        uniq_all = torch.unique(torch.cat([y_sup.cpu(), q_labels.cpu()])).tolist()
        _palette_all = plt.cm.tab20(np.linspace(0, 1, 20))
        cmap_all = {int(lbl): _palette_all[i % len(_palette_all)] for i, lbl in enumerate(uniq_all)}

        y_sup_np = y_sup.cpu().numpy()
        y_q_np = q_labels.cpu().numpy()

        # 创建并排子图
        fig, axes = plt.subplots(1, 2, figsize=(18, 7))

        # 左图：原始支持集 + 全部查询集
        ax = axes[0]
        for c in uniq_all:
            m_sup = (y_sup_np == c)
            if m_sup.any():
                ax.scatter(sup2_sq[m_sup,0], sup2_sq[m_sup,1], c=[cmap_all[c]], s=60, alpha=0.9,
                           edgecolors='black', linewidth=0.8, label=f'Support C{c}')
            m_q = (y_q_np == c)
            if m_q.any():
                ax.scatter(qry2_sq[m_q,0], qry2_sq[m_q,1], c=[cmap_all[c]], s=50, alpha=0.6,
                           edgecolors='none', marker='s', label=f'Query C{c}')
        ax.set_title('Original Support + All Query')
        ax.set_xlabel('t-SNE 1'); ax.set_ylabel('t-SNE 2')
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), fontsize=8)
        ax.grid(True, alpha=0.3)

        # 右图：Refined 支持集 + 原始查询集
        ax = axes[1]
        for c in uniq_all:
            m_sup = (y_sup_np == c)
            if m_sup.any():
                ax.scatter(ref2_sq[m_sup,0], ref2_sq[m_sup,1], c=[cmap_all[c]], s=80, alpha=0.7,
                           facecolors='none', edgecolors=cmap_all[c], linewidth=1.6, label=f'Refined C{c}')
            m_q = (y_q_np == c)
            if m_q.any():
                ax.scatter(qry2_sq[m_q,0], qry2_sq[m_q,1], c=[cmap_all[c]], s=50, alpha=0.6,
                           edgecolors='none', marker='s', label=f'Query C{c}')
        ax.set_title('Refined Support + Original Query')
        ax.set_xlabel('t-SNE 1'); ax.set_ylabel('t-SNE 2')
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), fontsize=8)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        if save_path:
            out_path = save_path[:-4] + '_sup_query.png' if save_path.lower().endswith('.png') else save_path + '_sup_query.png'
            plt.savefig(out_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()
        plt.close()

            



    

