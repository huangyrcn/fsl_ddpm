
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import os
from copy import deepcopy
from tqdm import tqdm
from torch.optim.lr_scheduler import LambdaLR, ReduceLROnPlateau
import math
import json
import hashlib

# 添加 wandb 导入
import wandb

# 添加 sklearn 导入
from sklearn.cluster import KMeans


class ZScore:
    def __init__(self): 
        self.mu = None
        self.std = None
    
    def fit(self, X, eps=1e-6):
        self.mu = X.mean(0, keepdim=True)
        self.std = (X.var(0, unbiased=False, keepdim=True) + eps).sqrt()
    
    def fwd(self, X): 
        # +1e-12 提升数值稳定性（与 inv 对齐）
        return (X - self.mu) / (self.std + 1e-12)
    
    def inv(self, Z): 
        return Z * (self.std + 1e-12) + self.mu


from gnn_model import Model, Prompt, LogReg
from ldm import LDM, finetune_param_filter
from dataset import Dataset
from aug import feature_mask, node_drop, feature_drop, feature_dropout


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
        
        # z-score标准化相关
        self.use_znorm = bool(not getattr(args, "ldm_unit_sphere", False))
        self.znorm = ZScore() if self.use_znorm else None
        
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
        elif self.args.aug1 in ['node_drop', 'drop_node']:
            graph_aug1 = node_drop(self.dataset.train_graphs, self.args.seed)
        elif self.args.aug1 in ['feature_mask', 'mask_feature']:
            graph_aug1 = feature_mask(self.dataset.train_graphs, self.args.seed)
        elif self.args.aug1 in ['feature_drop', 'drop_feature']:
            graph_aug1 = feature_drop(self.dataset.train_graphs, self.args.seed)
        elif self.args.aug1 == 'feature_dropout':
            graph_aug1 = feature_dropout(self.dataset.train_graphs, self.args.seed)
        else:
            # 如果没有匹配，默认使用identity
            print(f"⚠️ 未识别的aug1类型: {self.args.aug1}，使用identity")
            graph_aug1 = self.dataset.train_graphs

        if self.args.aug2 in ['node_drop', 'drop_node']:
            graph_aug2 = node_drop(graph_copy_2, self.args.seed)
        elif self.args.aug2 in ['feature_mask', 'mask_feature']:
            graph_aug2 = feature_mask(graph_copy_2, self.args.seed)
        elif self.args.aug2 in ['feature_drop', 'drop_feature']:
            graph_aug2 = feature_drop(self.dataset.train_graphs, self.args.seed)
        elif self.args.aug2 == 'feature_dropout':
            graph_aug2 = feature_dropout(self.dataset.train_graphs, self.args.seed)
        else:
            # 如果没有匹配，默认使用identity
            print(f"⚠️ 未识别的aug2类型: {self.args.aug2}，使用identity")
            graph_aug2 = graph_copy_2

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
        """初始化LDM相关组件（新版LDM）"""
        print("=== 初始化LDM组件 ===")
        embedding_dim = self.model.sample_input_emb_size
        print(f"检测到embedding维度: {embedding_dim}")

        # 一些合理默认超参（没有就用默认）
        predict_type   = getattr(self.args, "ldm_predict", "v")          # 'v' 或 'eps'
        unit_sphere    = getattr(self.args, "ldm_unit_sphere", False)    # 默认False，使用z-score
        widths         = getattr(self.args, "ldm_widths", (128, 256, 512))
        n_blocks       = getattr(self.args, "ldm_n_blocks", 2)
        use_zero_mlp   = getattr(self.args, "ldm_use_zero_mlp", True)
        time_steps     = int(self.args.time_steps)

        self.ldm = LDM(
            device=self.args.device,
            latent_dim=embedding_dim,
            timesteps=time_steps,
            cond_dim=embedding_dim,       # 条件就是同维嵌入原型
            predict=predict_type,
            unit_sphere=unit_sphere,
            self_condition=True,
            widths=widths,
            n_blocks_per_stage=n_blocks,
            use_zero_mlp=use_zero_mlp
        ).to(self.args.device)

        self.ldm_optimizer = torch.optim.AdamW(
            self.ldm.parameters(),
            lr=getattr(self.args, "learning_rate_ldm", 2e-4),
            weight_decay=getattr(self.args, "weight_decay_ldm", 1e-4)
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
        
        if self.args.use_prompt:
            self.prompt.eval()
            prompt_embeds = self.prompt()
        else:
            prompt_embeds = torch.zeros(self.args.num_token, self.args.node_fea_size).to(self.args.device)
        
        with torch.no_grad():
            all_graphs = self.dataset.train_graphs
            self.training_embeddings = self.model.encode_graphs(all_graphs, prompt_embeds)  # [N,D]
            self.training_labels = torch.tensor([g.label for g in all_graphs], device=self.args.device)

        # —— 拟合 z-score & 转 z 空间 ——
        if self.use_znorm:
            self.znorm.fit(self.training_embeddings)
            print(f"[Z] mean={self.znorm.fwd(self.training_embeddings).mean().item():.3f}, "
                  f"std={self.znorm.fwd(self.training_embeddings).std().item():.3f}")
            self.training_embeddings_z = self.znorm.fwd(self.training_embeddings)
        else:
            self.training_embeddings_z = self.training_embeddings

        # —— z 空间做 KMeans/条件 ——
        kmeans = KMeans(n_init=10, n_clusters=self.args.train_classes_num, random_state=42)
        cluster_labels = kmeans.fit_predict(self.training_embeddings_z.cpu().numpy())
        prototypes_z = torch.tensor(kmeans.cluster_centers_, dtype=torch.float, device=self.args.device)
        self.kmeans_prototypes_z = prototypes_z
        self.kmeans_prototypes   = self.znorm.inv(prototypes_z) if self.use_znorm else prototypes_z

        # 直接使用 kmeans_proto 作为训练条件
        cl_t = torch.tensor(cluster_labels, dtype=torch.long, device=self.args.device)
        self.conditions_z = prototypes_z[cl_t]
        
    def train_ldm(self):
        """训练LDM，基于encoder的embeddings"""
        print("=== 开始训练LDM ===")
        
        # 使用 kmeans_proto 作为训练条件
        conditions_z = self.conditions_z
        
        print(f"训练数据: {self.training_embeddings_z.shape[0]} 个embeddings")
        print(f"条件数据: {conditions_z.shape[0]} 个条件")
        
        best_loss = float('inf')
        patience_count = 0
        patience = self.args.patience_ldm
        check_interval = int(self.args.ldm_es_interval)
        
        # 创建进度条 - 动态长度显示
        pbar = tqdm(range(1, self.args.num_epochs_ldm + 1), desc="LDM Training", 
                   ncols=120, dynamic_ncols=True, leave=True)
        
        for epoch in pbar:
            # 随机打乱数据
            perm = torch.randperm(self.training_embeddings_z.shape[0])
            shuffled_embeddings = self.training_embeddings_z[perm]
            shuffled_conditions = conditions_z[perm]
            
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
                    "ldm_loss": avg_loss
                }, step=epoch)
            
            # 简单的早停检查（基于当前loss）
            if avg_loss < best_loss:
                best_loss = avg_loss
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
                        "ldm_patience_count": patience_count,
                        "ldm_patience": patience
                    }, step=epoch)
            if patience_count >= patience:
                break
        
    def _train_ldm_step(self, z, conditions):
        """新版LDM训练一步（无control，支持lambda_proto/proto）"""
        self.ldm.train()
        self.ldm_optimizer.zero_grad()

        p_uncond     = float(getattr(self.args, "ldm_p_uncond", 0.1))
        lambda_proto = float(getattr(self.args, "ldm_lambda_proto", 0.0))
        # 用 "条件本身" 作为原型一致性目标（也可以用你单独算的原型张量）
        proto = conditions

        loss = self.ldm.loss(
            x0=z, cond=conditions,
            p_uncond=p_uncond,
            lambda_proto=lambda_proto,
            proto=proto
        )
        loss.backward()
        self.ldm_optimizer.step()
        
        return loss
        


        
    def test_model(self, num_augmented_samples=0, ldm_model=None, test_name=None):
        """统一的测试函数"""
        # 统一命名
        test_name = test_name or ("LDM增强测试" if num_augmented_samples > 0 else "原始Encoder测试")
        self.model.eval()

        # 是否需要LDM：当需要生成增强时
        need_ldm = bool(num_augmented_samples > 0)

        # 根据 refine/augmentation 情况，统一打印更清晰的评估模式
        if need_ldm:
            if num_augmented_samples > 0:
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
            test_acc = self._evaluate_one_task_with_ldm(start_test_idx, num_augmented_samples, ldm_model, start_test_idx=start_test_idx)
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
    

        
    def _evaluate_one_task_with_ldm(
        self,
        test_idx: int,
        num_augmented_samples: int,
        ldm_model,
        start_test_idx=None,  # 兼容你原来的签名
    ):
        """评估一个few-shot任务（新版LDM流程）
        流程：采样任务 → (可选) per-task微调 → (可选) refine支持集 → (可选) 条件生成增强 → 线性分类评测
        """
        self.model.eval()
        
        # 选择用于采样的模型
        ldm_for_sampling = ldm_model or self.ldm

        # 评估期不使用 prompt（置零）
        prompt_embeds = torch.zeros(self.args.num_token, self.args.node_fea_size).to(self.args.device)

        # 取一个任务
        first_N_class_sample = np.array(list(range(self.dataset.test_classes_num)))
        current_task = self.dataset.sample_one_task(
            self.dataset.test_tasks,
            first_N_class_sample,
            K_shot=self.args.K_shot,
            query_size=self.args.query_size,
            test_start_idx=test_idx
        )

        # ------- 支持集：取前 K-shot -------
        support_embs, _ = self.model.sample_input_GNN([current_task], prompt_embeds, True)
        support_data = support_embs.reshape(
            self.args.N_way, -1, self.model.sample_input_emb_size
        )[:, :self.args.K_shot, :].reshape(
            self.args.N_way * self.args.K_shot, self.model.sample_input_emb_size
        ).detach()

        # 支持集标签
        support_label = []
        for graphs in current_task['support_set']:
            support_label.append(np.array([graph.label for graph in graphs[:self.args.K_shot]]))
        support_label = torch.LongTensor(np.hstack(support_label)).to(self.args.device)

        # ------- (可选) 任务级微调 LDM（只训FiLM/Zero-MLP） -------
        # 若传入了 ldm_model 且配置步数>0，则在当前任务支持集上做少量步的适配
        if (ldm_for_sampling is not None) and int(getattr(self.args, "task_finetune_steps", 0)) > 0:
            # 注意：_task_level_finetune 使用 self.ldm；主程序中 ldm_model = self.ldm，因此一致
            self._task_level_finetune(
                support_data=support_data,
                support_labels=support_label,
                task_id=test_idx // max(1, (self.args.N_way * self.args.query_size))
            )

        # 不再执行 refine，直接使用原始支持集
        support_data_for_enh = support_data

        # ------- (可选) 条件生成增强 -------
        enhanced_support_data, enhanced_support_labels = self.generate_augmented_embeddings(
            embeddings=support_data_for_enh,
            labels=support_label,
            num_to_generate=int(num_augmented_samples),
            ldm_model=ldm_for_sampling,
            device=self.args.device,
            condition_type="label_proto"
        )

        # ------- 训练线性分类器（与 baseline 保持一致）-------
        self.log = LogReg(self.model.sample_input_emb_size, self.args.N_way).to(self.args.device)
        self._train_classifier(
            enhanced_support_data,
            enhanced_support_labels
        )

        # ------- 查询集评估 -------
        query_embs, _ = self.model.sample_input_GNN([current_task], prompt_embeds, False)
        query_data = query_embs.reshape(
            self.args.N_way, self.args.query_size, self.model.sample_input_emb_size
        ).reshape(self.args.N_way * self.args.query_size, self.model.sample_input_emb_size).detach()

        query_label = []
        for graphs in current_task['query_set']:
            query_label.append(np.array([graph.label for graph in graphs]))
        query_label = torch.LongTensor(np.hstack(query_label)).to(self.args.device)

        # 处理填充数据（按你的 Dataset 接口）
        if current_task.get('append_count', 0) != 0:
            query_len = query_label.shape[0]
            query_data = query_data[:query_len - current_task['append_count'], :]
            query_label = query_label[:query_len - current_task['append_count']]

        # 预测
        logits = self.log(query_data)
        preds = torch.argmax(logits, dim=1)
        acc = torch.sum(preds == query_label).float() / query_label.shape[0]
        return acc.item()

    def _task_level_finetune(self, support_data, support_labels, task_id=None):
        """任务级 LDM 微调：只训 FiLM/Zero-MLP 小分支"""
        steps = int(self.args.task_finetune_steps)
        if steps <= 0:
            return

        # 1) 冻结主干，只开小分支
        trainable = []
        for n, p in self.ldm.named_parameters():
            flag = finetune_param_filter(n)  # 只启用 film/cond_proj/zero/gate/（可选lora）
            p.requires_grad_(flag)
            if flag: trainable.append(p)

        # 2) 构造条件：使用类别原型（z空间）
        support_z = self.znorm.fwd(support_data) if self.use_znorm else support_data
        with torch.no_grad():
            cond_z = torch.zeros_like(support_z)
            for cls in torch.unique(support_labels):
                m = (support_labels == cls)
                if m.any():
                    cond_z[m] = support_z[m].mean(0, keepdim=True)

        # 3) 优化器
        lr = float(getattr(self.args, "task_finetune_lr", 1e-3))
        wd = float(getattr(self.args, "task_finetune_weight_decay", 1e-4))
        opt = torch.optim.AdamW(trainable, lr=lr, weight_decay=wd)
        # 学习率调度：ReduceLROnPlateau，按当前损失自适应降低LR
        scheduler = ReduceLROnPlateau(
            opt, mode='min', factor=0.5, patience=50, threshold=1e-3,
            cooldown=0, min_lr=1e-7, verbose=True
        )

        self.ldm.train()
        drop_p    = float(getattr(self.args, "task_finetune_cond_dropout", 0.1))
        grad_clip = float(getattr(self.args, "task_finetune_grad_clip", 1.0))
        patience  = int(getattr(self.args, "task_finetune_patience", 50))
        lambda_proto = float(getattr(self.args, "task_lambda_proto", 0.0))

        # 体检：记录可训练参数数量与初始LR
        num_trainable = sum(p.numel() for p in trainable if p.requires_grad)
        print(f"[finetune] trainable params: {num_trainable}")
        if wandb.run is not None:
            wandb.log({f"task_{task_id}_lr": opt.param_groups[0]['lr']})

        # 稳压超参
        accum_steps   = int(getattr(self.args, "task_finetune_accum_steps", 4))
        micro_repeats = int(getattr(self.args, "task_finetune_micro_repeats", 4))
        ema_alpha     = float(getattr(self.args, "task_finetune_ema_alpha", 0.98))
        ema_loss = None

        best_loss, best_state, wait = float('inf'), None, 0
        pbar = tqdm(range(steps), desc=f"Task Finetune ({task_id})", position=1, leave=False)
        for step in pbar:
            total_loss = 0.0
            # micro-repeats：同批多次前向（不同 cond dropout 面具），均值降噪
            for _ in range(micro_repeats):
                cond_in = cond_z.clone()
                if drop_p > 0:
                    mask = (torch.rand(cond_in.size(0), device=cond_in.device) < drop_p).float().unsqueeze(-1)
                    cond_in = cond_in * (1 - mask)

                loss_once = self.ldm.loss(
                    x0=support_z,
                    cond=cond_in,
                    p_uncond=0.0,
                    lambda_proto=lambda_proto,
                    proto=cond_z
                )
                total_loss = total_loss + loss_once

            loss = total_loss / float(max(1, micro_repeats))

            # 梯度累积
            (loss / float(max(1, accum_steps))).backward()

            cur = float(loss.item())
            ema_loss = cur if ema_loss is None else (ema_alpha * ema_loss + (1.0 - ema_alpha) * cur)

            if (step + 1) % max(1, accum_steps) == 0:
                torch.nn.utils.clip_grad_norm_(trainable, grad_clip)
                opt.step()
                opt.zero_grad(set_to_none=True)
                # 调度器用 EMA 更稳
                scheduler.step(ema_loss)

            pbar.set_postfix({'Loss': f'{cur:.4f}', 'EMA': f'{(ema_loss if ema_loss is not None else cur):.4f}'})
            if wandb.run is not None:
                wandb.log({
                    f"task_{task_id}_loss": cur,
                    f"task_{task_id}_ema_loss": (ema_loss if ema_loss is not None else cur),
                    f"task_{task_id}_lr": opt.param_groups[0]['lr'],
                })

            # 早停依据使用 EMA
            score_for_es = ema_loss if ema_loss is not None else cur
            if score_for_es < best_loss:
                best_loss, wait = score_for_es, 0
                from copy import deepcopy
                best_state = deepcopy(self.ldm.state_dict())
            else:
                wait += 1
                if wait >= patience:
                    break

        if best_state is not None:
            self.ldm.load_state_dict(best_state)
        self.ldm.eval()
        
    def _train_classifier(self, support_data, support_label):
        """训练线性分类器（与 baseline 一致：优化器包含 prompt）"""
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
            
            logits = self.log(support_data)
            loss_ori = self.xent(logits, support_label)
            
            # L2正则化
            l2_reg = torch.tensor(0.).to(self.args.device)
            for param in self.log.parameters():
                l2_reg += torch.norm(param)
            loss_total = loss_ori + 0.1 * l2_reg
            
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
        
 
    def generate_augmented_embeddings(self, embeddings, labels, num_to_generate, *, ldm_model, device, condition_type):
        if (num_to_generate <= 0) or (ldm_model is None):
            return embeddings, labels
        D = embeddings.shape[1]

        # → z 空间
        emb_z = self.znorm.fwd(embeddings) if self.use_znorm else embeddings

        # 条件（z 空间，建议 label_proto = 类均值）
        if condition_type == 'label_proto':
            conds_z = torch.zeros_like(emb_z)
            for cls in torch.unique(labels):
                m = (labels == cls)
                if m.any():
                    proto_z = emb_z[m].mean(0, keepdim=True)
                    conds_z[m] = proto_z
        else:
            conds_z = emb_z

        # 采样（加大多样性：simple_var=False）
        expanded_conds_z = conds_z.repeat_interleave(num_to_generate, dim=0)
        expanded_labels  = labels.repeat_interleave(num_to_generate, dim=0)
        
        with torch.no_grad():
            temp = float(getattr(self.args, "ldm_aug_temp", 0.9))
            simp = bool(getattr(self.args, "ldm_aug_simple_var", False))
            guid = float(getattr(self.args, "ldm_guidance", 1.8))

            init_r = emb_z.norm(dim=1).repeat_interleave(num_to_generate, dim=0)

            z_gen = ldm_model.sample(
                shape=(len(expanded_conds_z), D),
                cond=expanded_conds_z,
                guidance=guid,
                simple_var=simp,
                temp=temp,
                init_match_radius=init_r
            )

        # ← 还原到原空间
        z_gen_orig = self.znorm.inv(z_gen) if self.use_znorm else z_gen
        return torch.cat([embeddings, z_gen_orig], 0), torch.cat([labels, expanded_labels], 0)
    
    @torch.no_grad()
    def refine_embeddings_with_ldm(self, embeddings, labels, ldm_model, alpha=0.3, use_slerp=False, batch_size=256):
        ldm_model.eval()
        T = ldm_model.timesteps
        t_ref = max(1, int(0.1 * T))
        x0z_all = self.znorm.fwd(embeddings) if self.use_znorm else embeddings
        outs = []
        for i in range(0, x0z_all.size(0), batch_size):
            x0_z = x0z_all[i:i+batch_size]
            B = x0_z.size(0)
            t = torch.full((B,), t_ref, dtype=torch.long, device=x0_z.device)

            # 条件：类均值（z 空间）
            if labels is not None:
                yb = labels[i:i+B]
                cond_z = torch.zeros_like(x0_z)
                for cls in torch.unique(yb):
                    m = (yb == cls)
                    if m.any():
                        cond_z[m] = x0_z[m].mean(0, keepdim=True)
            else:
                cond_z = x0_z

            x_t, _ = ldm_model.addnoise(x0_z, t)
            out = ldm_model._predict_model_out(x_t, t, cond=cond_z, guidance=float(getattr(self.args,"ldm_guidance",1.8)), x0_sc=torch.zeros_like(x_t))
            x0_hat_z = ldm_model._x0_from_v(x_t, t, out) if ldm_model.predict=='v' else ldm_model._x0_from_eps(x_t, t, out)
            x_ref_z = (1 - alpha) * x0_z + alpha * x0_hat_z  # 线性插值
            outs.append(x_ref_z)
        zref = torch.cat(outs, 0)
        return self.znorm.inv(zref) if self.use_znorm else zref

    def visualize_test_data(self, save_path=None):
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
        ldm_for_refine = self.ldm
        z_ref = self.refine_embeddings_with_ldm(
            embeddings=z_sup,
            labels=y_sup,
            ldm_model=ldm_for_refine,
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
                ldm_model=ldm_for_refine,
                device=self.args.device,
                condition_type="label_proto"
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

    def _compute_label_prototypes(self, embeddings, labels, normalize=True, device=None):
        """计算标签原型：先对每个样本 L2 归一化，再求类均值，再归一化一次"""
        if device is None:
            device = embeddings.device
            
        # 2) 余弦一致的原型：先对每个样本 L2 归一化，再求类均值，再归一化一次
        E = F.normalize(embeddings, dim=1) if normalize else embeddings

        uniq = torch.unique(labels).to(device)
        uniq, _ = torch.sort(uniq)            # 保证有序，便于配色/可视化对齐
        protos = []
        proto_ids = []
        for c in uniq.tolist():
            m = (labels == c)
            if m.any():
                p = E[m].mean(0, keepdim=True)
                if normalize:
                    p = F.normalize(p, dim=1)
                protos.append(p)
                proto_ids.append(c)

        if len(protos) == 0:
            D = embeddings.size(1)
            return torch.empty(0, D, device=device), []

        return torch.cat(protos, dim=0), proto_ids

    def visualize_train_data(self, save_path=None):
        """可视化训练集嵌入的t-SNE图"""
        import matplotlib.pyplot as plt
        from sklearn.manifold import TSNE
        import torch.nn.functional as F
        import os
        
        
        Z = F.normalize(self.training_embeddings, dim=1).cpu()
        Y = self.training_labels.cpu().numpy()
        
        # 计算标签原型
        label_protos, proto_ids = self._compute_label_prototypes(
            self.training_embeddings, self.training_labels, normalize=True, device=self.args.device
        )
        label_protos = label_protos.cpu()
        uniq = proto_ids
        
        # 使用已计算的K-means原型
        kmeans_protos = self.kmeans_prototypes.cpu()
        
        # t-SNE
        X_all = torch.cat([Z, label_protos, kmeans_protos], 0)
        X2 = TSNE(n_components=2, random_state=42).fit_transform(X_all.numpy())
        
        # 分离数据
        n_samples = len(Z)
        n_label_protos = len(label_protos)
        emb2 = X2[:n_samples]
        label_proto2 = X2[n_samples:n_samples+n_label_protos]
        kmeans_proto2 = X2[n_samples+n_label_protos:]
        
        # 获取坐标范围，用于保持一致的坐标轴
        x_min, x_max = X2[:, 0].min(), X2[:, 0].max()
        y_min, y_max = X2[:, 1].min(), X2[:, 1].max()
        x_margin = (x_max - x_min) * 0.05
        y_margin = (y_max - y_min) * 0.05
        
        # 为每个标签原型生成单独的图
        for i, c in enumerate(proto_ids):
            plt.figure(figsize=(10, 8))
            colors = plt.cm.tab20(np.linspace(0, 1, len(uniq)))
            
            # 绘制所有训练样本（灰色，透明度较低）
            plt.scatter(emb2[:, 0], emb2[:, 1], c='lightgray', s=30, alpha=0.3, label='All samples')
            
            # 高亮当前类别的样本
            mask = (Y == c)
            if mask.any():
                plt.scatter(emb2[mask, 0], emb2[mask, 1], c=[colors[i]], s=80, alpha=0.8, 
                           label=f'Class {c} samples', edgecolors='black', linewidth=0.5)
            
            # 绘制当前类别的标签原型
            if i < len(label_proto2):
                plt.scatter(label_proto2[i, 0], label_proto2[i, 1], c=[colors[i]], s=300, marker='*', 
                           edgecolors='black', linewidth=3, label=f'Label Proto {c}')
            
            # 绘制所有K-means原型（红色）
            for j in range(len(kmeans_proto2)):
                plt.scatter(kmeans_proto2[j, 0], kmeans_proto2[j, 1], c='red', s=200, marker='^', 
                           edgecolors='black', linewidth=2, label=f'K-means Proto {j}' if j == 0 else "")
            
            # 设置一致的坐标轴范围
            plt.xlim(x_min - x_margin, x_max + x_margin)
            plt.ylim(y_min - y_margin, y_max + y_margin)
            
            plt.title(f'Class {c} Prototype Visualization')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            
            if save_path:
                # 为每个原型生成单独的文件
                base_name = os.path.splitext(save_path)[0]
                proto_save_path = f"{base_name}_class_{c}.png"
                plt.savefig(proto_save_path, dpi=300, bbox_inches='tight')
                print(f"保存类别 {c} 的原型图到: {proto_save_path}")
            else:
                plt.show()
            plt.close()
        
        # 生成包含所有原型的总图
        plt.figure(figsize=(12, 8))
        colors = plt.cm.tab20(np.linspace(0, 1, len(uniq)))
        
        # 绘制训练样本
        for i, c in enumerate(uniq):
            mask = (Y == c)
            if mask.any():
                plt.scatter(emb2[mask, 0], emb2[mask, 1], c=[colors[i]], s=50, alpha=0.7, label=f'Class {c}')
        
        # 绘制标签原型
        for i, c in enumerate(proto_ids):
            if i < len(label_proto2):
                color_idx = uniq.index(c) if c in uniq else i
                plt.scatter(label_proto2[i, 0], label_proto2[i, 1], c=[colors[color_idx]], s=200, marker='*', 
                           edgecolors='black', linewidth=2, label=f'Label Proto {c}' if i == 0 else "")
        
        # 绘制K-means原型
        for i in range(len(kmeans_proto2)):
            plt.scatter(kmeans_proto2[i, 0], kmeans_proto2[i, 1], c='red', s=200, marker='^', 
                       edgecolors='black', linewidth=2, label=f'K-means Proto {i}' if i == 0 else "")
        
        # 设置一致的坐标轴范围
        plt.xlim(x_min - x_margin, x_max + x_margin)
        plt.ylim(y_min - y_margin, y_max + y_margin)
        
        plt.title('All Prototypes Visualization (Samples + Label Prototypes + K-means Prototypes)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"保存总图到: {save_path}")
        else:
            plt.show()
        plt.close()

    # =========================
    # 内在质量评估（MMD/FD/多样性/NN）
    # =========================
    @staticmethod
    def _mmd_rbf(X: torch.Tensor, Y: torch.Tensor, sigmas=(0.5, 1.0, 2.0, 4.0)) -> torch.Tensor:
        def pdist2(a, b):
            a2 = (a * a).sum(1, keepdim=True)
            b2 = (b * b).sum(1, keepdim=True).t()
            return a2 - 2 * (a @ b.t()) + b2
        dxx = pdist2(X, X); dyy = pdist2(Y, Y); dxy = pdist2(X, Y)
        Kxx = X.new_zeros(())
        Kyy = X.new_zeros(())
        Kxy = X.new_zeros(())
        for s in sigmas:
            g = torch.exp(-dxx / (2 * s * s)); Kxx = Kxx + (g.sum() - g.diag().sum()) / (X.size(0) * (X.size(0) - 1))
            g = torch.exp(-dyy / (2 * s * s)); Kyy = Kyy + (g.sum() - g.diag().sum()) / (Y.size(0) * (Y.size(0) - 1))
            g = torch.exp(-dxy / (2 * s * s)); Kxy = Kxy + g.mean()
        return Kxx + Kyy - 2 * Kxy

    @staticmethod
    def _frechet_distance(X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
        # 均值
        mu1, mu2 = X.mean(0), Y.mean(0)
        diff = mu1 - mu2

        # 协方差（无偏/有偏都可，保持一致即可）
        Xc = X - mu1
        Yc = Y - mu2
        C1 = (Xc.t() @ Xc) / max(1, (X.size(0) - 1))
        C2 = (Yc.t() @ Yc) / max(1, (Y.size(0) - 1))

        # sqrt(C1)
        e1, V1 = torch.linalg.eigh(C1)           # C1 对称
        e1 = e1.clamp_min(0)
        C1_sqrt = (V1 * e1.sqrt().unsqueeze(0)) @ V1.t()

        # S = sqrt(C1) @ C2 @ sqrt(C1) 也是对称
        S = C1_sqrt @ C2 @ C1_sqrt
        es, _ = torch.linalg.eigh(S)             # S 对称
        es = es.clamp_min(0)

        # tr(sqrt(S)) = sum(sqrt(eigvals(S)))
        tr_sqrt = es.sqrt().sum()

        fid = (diff @ diff) + torch.trace(C1) + torch.trace(C2) - 2.0 * tr_sqrt
        # 数值稳定：避免 -1e-6 之类的微小负数
        return fid.clamp_min(0.0)

    @staticmethod
    def _pairwise_stats(X: torch.Tensor) -> dict:
        # 返回均值/方差/分位数，作为多样性指标
        with torch.no_grad():
            d = torch.cdist(X, X, p=2)
            triu = torch.triu_indices(d.size(0), d.size(1), offset=1)
            vals = d[triu[0], triu[1]]
            return {
                'dist_mean': float(vals.mean().item()),
                'dist_std': float(vals.std(unbiased=False).item()),
                'dist_p10': float(vals.kthvalue(max(1, int(0.10 * len(vals))))[0].item()),
                'dist_p50': float(vals.median().item()),
                'dist_p90': float(vals.kthvalue(max(1, int(0.90 * len(vals))))[0].item()),
            }

    @staticmethod
    def _nn_distances(X: torch.Tensor, Y: torch.Tensor) -> dict:
        # 每个生成样本到最近真实样本的距离统计
        with torch.no_grad():
            d = torch.cdist(X, Y, p=2)
            mins = d.min(dim=1).values
            return {
                'nn_mean': float(mins.mean().item()),
                'nn_std': float(mins.std(unbiased=False).item()),
                'nn_p10': float(mins.kthvalue(max(1, int(0.10 * len(mins))))[0].item()),
                'nn_p50': float(mins.median().item()),
                'nn_p90': float(mins.kthvalue(max(1, int(0.90 * len(mins))))[0].item()),
            }

    @torch.no_grad()
    def evaluate_ldm_intrinsic(self, num_samples: int = 1000, log_to_wandb: bool = True) -> dict:
        assert self.ldm is not None, "LDM not initialized"
        device = self.args.device

        # —— 选同一批索引 —— #
        all_emb = self.training_embeddings.detach().to(device)
        N = all_emb.size(0)
        if N > num_samples:
            sel_idx = torch.randperm(N, device=device)[:num_samples]
        else:
            sel_idx = torch.arange(N, device=device)

        real = all_emb[sel_idx]

        # —— z 空间对齐 —— #
        if self.use_znorm:
            real_z = self.znorm.fwd(real)
        else:
            real_z = real
            if bool(getattr(self.args, 'ldm_unit_sphere', False)):
                real_z = F.normalize(real_z, dim=1)

        D = real_z.size(1)

        # —— 条件：与 sel_idx 对齐的 labels —— #
        labels = getattr(self, "training_labels", None)
        if labels is not None:
            labels = labels[sel_idx]

        # 直接使用 label_proto 作为生成条件
        if labels is not None:
            cond_z = torch.zeros_like(real_z)
            for cls in torch.unique(labels):
                m = (labels == cls)
                if m.any():
                    cond_z[m] = real_z[m].mean(0, keepdim=True)
        else:
            cond_z = real_z

        # —— 生成（z 空间）→ 还原到原空间再评估 —— #
        # 评估用采样和增强一致：无引导 + 非简单方差 + 降温 + 半径对齐
        temp = float(getattr(self.args, 'ldm_eval_temp', 0.85))
        simp = bool(getattr(self.args, 'ldm_eval_simple_var', False))
        guid = float(getattr(self.args, 'ldm_guidance_eval', 0.0))

        # 用真实样本的半径对齐初始噪声，防止非 unit_sphere 时一开局半径失配
        init_r = real_z.norm(dim=1)  # [N]

        # 使用模型进行采样
        ldm_for_eval = self.ldm
        gen_z = ldm_for_eval.sample(
            shape=(len(cond_z), D),
            cond=cond_z,
            guidance=guid,
            simple_var=simp,
            temp=temp,
            init_match_radius=init_r
        )
        gen = self.znorm.inv(gen_z) if self.use_znorm else gen_z

        if bool(getattr(self.args, 'ldm_unit_sphere', False)):
            norms = gen_z.norm(dim=1)
            norm_violation = float((norms - 1.0).abs().mean().item())
        else:
            norm_violation = 0.0

        mmd = float(self._mmd_rbf(real, gen).item())
        fd  = float(self._frechet_distance(real, gen).item())
        
        # 仅调试：看看 z 空间是否也方差炸了
        z_div_real = self._pairwise_stats(real_z)
        z_div_gen  = self._pairwise_stats(gen_z)
        print(f"[z-space] pairwise real/gen: {z_div_real['dist_mean']:.3f} / {z_div_gen['dist_mean']:.3f}")
        
        metrics = {
            'mmd_rbf': mmd,
            'frechet_distance': fd,
            'real_pairwise': self._pairwise_stats(real),
            'gen_pairwise': self._pairwise_stats(gen),
            'gen_to_real_nn': self._nn_distances(gen, real),
            'unit_sphere_norm_violation_mean_abs': norm_violation,
        }
        if log_to_wandb and wandb.run is not None:
            wandb.log({
                'ldm_intrinsic/mmd_rbf': mmd,
                'ldm_intrinsic/frechet_distance': fd,
                'ldm_intrinsic/nn_mean': metrics['gen_to_real_nn']['nn_mean'],
                'ldm_intrinsic/gen_dist_mean': metrics['gen_pairwise']['dist_mean'],
            })
        return metrics

    # refine 相关逻辑已移除