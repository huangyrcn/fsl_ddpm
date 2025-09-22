import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random  # 🔧 添加 random 模块导入
import os
from copy import deepcopy
from tqdm import tqdm
import json
import wandb
from sklearn.cluster import KMeans
import warnings

from gin_model import LogReg, Model as GINModel
from models.model import HGINModel
from custom_dataset import CustomDataset
from aug import get_optimized_augmentation  # 使用优化版本
from torch_geometric.data import Data, Batch

from ldm import LDM, finetune_param_filter


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

class PyGTrainer:
    """
    PyTorch Geometric Trainer for few-shot learning, using GIN and PyG augmentations.
    Compatible with unified_trainer.py interface.
    """

    def __init__(self, args, logf=None):
        self.args = args
        self.logf = logf

        # 初始化保存目录
        self.save_dir = "./savepoint"
        os.makedirs(self.save_dir, exist_ok=True)

        # 🎯 对齐main.py: 移除重复的种子设置，依赖main_pyg.py中的全局设置

        # 使用CustomDataset替代原来的Dataset
        self.dataset = CustomDataset(args.dataset_name, args)
        args.train_classes_num = self.dataset.train_classes_num
        args.test_classes_num = self.dataset.test_classes_num

        # 从实际的PyG Data中获取特征维度
        args.node_fea_size = self.dataset.feature_dim
        print(f"节点特征维度: {args.node_fea_size}")

        args.N_way = args.test_classes_num

        # Model - 根据encodertype参数选择模型
        encodertype = getattr(args, 'encodertype', 'gin')  # 默认为gin
        if encodertype == 'lgin':
            self.model = HGINModel(args).to(args.device)
        else:  # 默认为gin
            self.model = GINModel(args).to(args.device)
        self.encoder_optimizer = optim.Adam(
            self.model.parameters(), lr=args.lr, weight_decay=args.weight_decay
        )

        # Z-score normalization
        self.use_znorm = bool(not getattr(args, "ldm_unit_sphere", False))
        self.znorm = ZScore() if self.use_znorm else None

        # Evaluation components
        in_dim = self.model.sample_input_emb_size
        num_cls = self.args.N_way
        self.log = LogReg(in_dim, num_cls).to(self.args.device)
        self.xent = nn.CrossEntropyLoss()

        # LDM 相关（延迟构建）
        self.ldm = None
        self.ldm_optimizer = None

        # 训练集 embedding/条件（用于 LDM 训练）
        self.training_embeddings = None  # raw, unprojected (for LDM)
        self.training_embeddings_proj = None  # projected, for linear probe
        self.training_embeddings_z = None
        self.training_labels = None
        self.kmeans_proto = None  # 条件原型（z 空间）
        self.conditions_z = None  # 训练条件（z 空间）

        # Graph augmentation for contrastive learning - 与main.py保持一致
        # mode选项: 'ultra_fast'(仅测试), 'consistent'(默认，数学一致)
        aug_mode = getattr(args, "aug_mode", "consistent")  # 默认数学一致模式

        # 🎯 对齐main.py: 两个增强使用相同的seed确保一致性
        self.contrastive_aug1 = get_optimized_augmentation(
            aug_type=args.aug1,
            aug_ratio=getattr(args, "aug_ratio", 0.1),
            device=args.device,  # 传递设备参数
            seed=args.seed,
            mode=aug_mode,
        )

        self.contrastive_aug2 = get_optimized_augmentation(
            aug_type=args.aug2,
            aug_ratio=getattr(args, "aug_ratio", 0.1),
            device=args.device,  # 传递设备参数
            seed=args.seed,  # 🎯 关键修复：使用相同seed而不是seed+1
            mode=aug_mode,
        )

    def train_encoder_contrastive(self):
        """
        阶段1：对比学习训练Encoder (重构版本)。
        - 合并了 _encoder_contrastive_step 的逻辑。
        - 将 DataLoader 的创建移到循环外以提高性能。
        """
        print("=== 阶段1：开始对比学习训练Encoder (优化版) ===")

        best_loss = 1e9
        best_epoch = 0
        patience_counter = 0

        encoder_batch_size = getattr(self.args, "encoder_batch_size", 32)
        num_workers = getattr(self.args, "num_workers", 4)

        print(f"训练图数量: {len(self.dataset.train_graphs)}")
        print(f"编码器训练批处理大小: {encoder_batch_size}")
        print(f"数据加载器Workers: {num_workers}")


        for epoch in tqdm(
            range(self.args.epoch_num),
            desc="对比学习训练Encoder",
            dynamic_ncols=True,
            leave=True,
        ):
            encoder_loader1, encoder_loader2 = self.dataset.aug(
            aug1_type=self.args.aug1,
            aug2_type=self.args.aug2,
            batch_size=encoder_batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            seed=self.args.seed,
            aug_ratio=getattr(self.args, "aug_ratio", 0.1),
            )

            self.model.train()  # 每个epoch开始时设置模型为训练模式
            epoch_losses = []

            for batch1, batch2 in zip(encoder_loader1, encoder_loader2):
                if batch1.num_graphs < 2 or batch2.num_graphs < 2:
                    continue
                batch1 = batch1.to(self.args.device, non_blocking=True)
                batch2 = batch2.to(self.args.device, non_blocking=True)
                encoder_embs1 = self.model(batch1)
                encoder_embs2 = self.model(batch2)

                contrastive_loss = self.model.loss_cal(encoder_embs1, encoder_embs2)
                if not torch.isfinite(contrastive_loss):
                    tqdm.write("检测到 NaN/Inf loss，本批次跳过。")
                    continue

                self.encoder_optimizer.zero_grad()
                contrastive_loss.backward()
                self.encoder_optimizer.step()

                epoch_losses.append(contrastive_loss.item())

            if not epoch_losses:
                tqdm.write(f"Epoch {epoch}: 没有有效的批次进行训练，跳过。")
                continue

            avg_loss = np.mean(epoch_losses)

            # --- 早停和模型保存逻辑 ---
            if avg_loss < best_loss:
                best_loss = avg_loss
                best_epoch = epoch
                patience_counter = 0
                torch.save(
                    self.model.state_dict(),
                    os.path.join(
                        self.save_dir, f"{self.args.dataset_name}_encoder.pkl"
                    ),
                )
            else:
                patience_counter += 1

            if epoch % 10 == 0:  # 每10个epoch打印一次
                tqdm.write(
                    f"Epoch {epoch} | Loss: {avg_loss:.4f} | Best Loss: {best_loss:.4f} @ Epoch {best_epoch}"
                )
                if self.logf:
                    self.logf.write(f"Epoch {epoch} Loss {avg_loss:.4f}\n")

            if patience_counter > self.args.patience:
                tqdm.write(f"在Epoch {epoch}提前停止!")
                break

        print(f"对比学习训练完成，最佳epoch: {best_epoch}，最佳loss: {best_loss:.4f}")

    def load_pretrained_encoder(self, ckpt_path=None):
        """从已保存的pkl加载encoder权重并切到eval模式"""
        if ckpt_path is None:
            ckpt_path = os.path.join(
                self.save_dir, f"{self.args.dataset_name}_encoder.pkl"
            )
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"找不到encoder权重文件: {ckpt_path}")
        try:
            state = torch.load(
                ckpt_path, map_location=self.args.device, weights_only=True
            )
        except TypeError:
            state = torch.load(ckpt_path, map_location=self.args.device)
        self.model.load_state_dict(state)
        self.model.eval()
        print(f"已从 {os.path.basename(ckpt_path)} 加载Encoder权重，并切换到eval模式")

    def test_model(self, num_augmented_samples=0, test_name=None):
        """测试模型性能 - 正确的few-shot任务采样逻辑"""
        test_name = test_name or (
            "增强测试" if num_augmented_samples > 0 else "原始Encoder测试"
        )
        print(f"=== {test_name} ===")

        self.model.eval()
        test_accs = []
        num_test_tasks = self.dataset.test_task_Num  # 使用预计算的测试任务数量
        for task_idx in range(num_test_tasks):
            # 使用任务ID直接获取任务（按顺序划分）
            task_support_graphs, task_query_graphs = self.dataset.sample_one_task(
                task_idx
            )

            # 传递具体的任务数据给评估函数
            test_acc = self._evaluate_one_task(
                task_support_graphs, task_query_graphs, num_augmented_samples
            )
            test_accs.append(test_acc)

        avg_acc = np.mean(test_accs) if test_accs else 0.0
        std_acc = np.std(test_accs) if test_accs else 0.0

        print(f"测试完成，准确率: {avg_acc:.4f} ± {std_acc:.4f}")

        return test_accs

    def _evaluate_one_task(self, support_graphs, query_graphs, num_augmented_samples=0):
        """评估单个few-shot任务 - 使用真实标签版本"""
        self.model.eval()  # 确保模型处于eval模式

        with torch.no_grad():
            # 准备support set batch
            from torch_geometric.data import Batch

            support_batch = Batch.from_data_list(support_graphs).to(self.args.device)

            # 准备query set batch
            query_batch = Batch.from_data_list(query_graphs).to(self.args.device)

            # 处理support和query set
            support_embs, _ = self.model.sample_input_GNN(support_batch, None)
            query_embs, _ = self.model.sample_input_GNN(query_batch, None)

            # 直接使用数据中的真实标签
            # 支持集标签（应该是有序的：每个类别K_shot个样本）
            y_support = support_batch.y.to(self.args.device)

            # 查询集标签（可能是随机分布的）
            y_query = query_batch.y.to(self.args.device)

        # 重新初始化线性分类器确保参数需要梯度
        from copy import deepcopy

        in_dim = support_embs.size(1)
        temp_log = LogReg(in_dim, self.args.N_way).to(self.args.device)
        opt = torch.optim.SGD(temp_log.parameters(), lr=0.01)

        best_loss, best_state, wait, patience = 1e9, None, 0, 100
        xent = self.xent

        temp_log.train()
        for _ in range(500):
            opt.zero_grad()
            logits = temp_log(support_embs.detach())
            loss_ori = xent(logits, y_support)
            # 显式 L2（与原版一致）
            l2 = sum(torch.norm(p) for p in temp_log.parameters())
            loss = loss_ori + 0.1 * l2
            loss.backward()
            opt.step()

            cur = loss.item()
            if cur < best_loss:
                best_loss, best_state, wait = cur, deepcopy(temp_log.state_dict()), 0
            else:
                wait += 1
            if wait > patience:
                break

        if best_state is not None:
            temp_log.load_state_dict(best_state)
        temp_log.eval()
        with torch.no_grad():
            logits = temp_log(query_embs.detach())
            pred = logits.argmax(dim=1)
            acc = (pred == y_query).float().mean().item()

        return acc

    @torch.no_grad()
    def _collect_training_embeddings_pyg(self):
        """用当前 Encoder 对训练图计算 embeddings，并在 z 空间构条件原型（kmeans 或类均值）"""
        self.model.eval()
        loader = self.dataset.get_encoder_trainloader(
            batch_size=getattr(self.args, "batch_size_for_embedding", 512),
            shuffle=False,
            num_workers=0,
            pin_memory=False,
        )
        embs_raw, embs_proj, lbs = [], [], []
        for batch in loader:
            batch = batch.to(self.args.device, non_blocking=True)
            # Raw pooled features (unprojected) for LDM
            e_raw = self.model.encode_graphs(batch)  # [B, D_raw]
            embs_raw.append(e_raw)
            # Projected features for linear probe
            e_proj = self.model(batch)  # [B, D_proj]
            embs_proj.append(e_proj.detach())
            lbs.append(batch.y)
        self.training_embeddings = torch.cat(embs_raw, 0)  # [N, D_raw]
        self.training_embeddings_proj = torch.cat(embs_proj, 0)  # [N, D_proj]
        self.training_labels = torch.cat(lbs, 0).to(self.args.device)

        if self.use_znorm:
            self.znorm.fit(self.training_embeddings)
            self.training_embeddings_z = self.znorm.fwd(self.training_embeddings)
        else:
            self.training_embeddings_z = self.training_embeddings

        # 条件：用类均值（更稳定）；如需 kmeans 可替换为 KMeans
        D = self.training_embeddings_z.size(1)
        num_cls = int(self.training_labels.max().item() + 1)
        protos = []
        for c in range(num_cls):
            m = self.training_labels == c
            if m.any():
                protos.append(self.training_embeddings_z[m].mean(0, keepdim=True))
        self.kmeans_proto = torch.cat(protos, 0)  # [C,D]
        self.conditions_z = self.kmeans_proto

    def _build_ldm_pyg(self):
        """按配置构建 LDM（与 unified_trainer 一致的超参命名）"""
        if self.ldm is not None:
            return
        embedding_dim = int(self.model.sample_input_emb_size)
        widths = list(getattr(self.args, "ldm_widths", [256, 512, 512]))
        n_blocks = int(getattr(self.args, "ldm_n_blocks", 2))
        predict_type = str(getattr(self.args, "ldm_predict", "v"))
        unit_sphere = bool(getattr(self.args, "ldm_unit_sphere", False))
        time_steps = int(self.args.time_steps)
        use_zero_mlp = bool(getattr(self.args, "ldm_use_zero_mlp", False))
        device = self.args.device

        # Initialize LDM
        self.ldm = LDM(
            device=device,
            latent_dim=embedding_dim,
            timesteps=time_steps,
            cond_dim=embedding_dim,
            predict=predict_type,
            unit_sphere=unit_sphere,
            self_condition=True,
            widths=widths,
            n_blocks_per_stage=n_blocks,
            use_zero_mlp=use_zero_mlp,
        ).to(device)
        self.ldm_optimizer = optim.AdamW(
            self.ldm.parameters(),
            lr=float(getattr(self.args, "learning_rate_ldm", 1e-4)),
            weight_decay=float(getattr(self.args, "weight_decay_ldm", 1e-4)),
        )

    def train_ldm_pyg(self):
        """训练 LDM（PyG 版，与 unified_trainer 流程等价）"""
        tqdm.write("=== 收集训练集 embeddings（用于 LDM） ===")
        self._collect_training_embeddings_pyg()
        self._build_ldm_pyg()
        tqdm.write("=== 开始训练 LDM ===")
        best = float("inf")
        wait = 0
        patience = int(getattr(self.args, "patience_ldm", 1000))
        es_interval = int(getattr(self.args, "ldm_es_interval", 200))
        T = int(self.args.time_steps)
        p_uncond = float(getattr(self.args, "ldm_p_uncond", 0.3))
        lam_proto = float(getattr(self.args, "ldm_lambda_proto", 0.1))
        bs = int(getattr(self.args, "ldm_batch_size", 512))
        for epoch in tqdm(
            range(1, int(getattr(self.args, "num_epochs_ldm", 30000)) + 1),
            desc="LDM Training",
            ncols=120, dynamic_ncols=True, leave=True
        ):
            # 随机小批
            perm = torch.randperm(
                self.training_embeddings_z.size(0), device=self.args.device
            )
            loss_epoch = 0.0
            nstep = 0
            for i in range(0, perm.numel(), bs):
                idx = perm[i : i + bs]
                x0 = self.training_embeddings_z[idx]  # [B,D]
                # 高效的条件计算：使用索引映射
                cls = self.training_labels[idx]
                cond = self.kmeans_proto[cls]  # 直接索引，避免循环

                # 无条件丢弃
                m_un = torch.rand(x0.size(0), device=x0.device) < p_uncond
                cond[m_un] = 0.0

                # 前向
                self.ldm_optimizer.zero_grad(set_to_none=True)
                loss = self.ldm.loss(
                    x0, cond, p_uncond=p_uncond, lambda_proto=lam_proto, proto=cond
                )
                loss.backward()
                self.ldm_optimizer.step()
                loss_epoch += loss.item()
                nstep += 1

            avg = loss_epoch / max(1, nstep)
            if wandb.run is not None:
                wandb.log({"ldm_loss": avg}, step=epoch)

            if avg < best:
                best = avg
                wait = 0
                torch.save(
                    self.ldm.state_dict(),
                    os.path.join(self.save_dir, f"{self.args.dataset_name}_ldm.pkl"),
                )
            else:
                wait += 1
            if (wait >= patience) or (epoch % es_interval == 0 and epoch > 0):
                tqdm.write(
                    f"[LDM] epoch={epoch} avg_loss={avg:.4f} best={best:.4f} wait={wait}"
                )
            if wait >= patience:
                tqdm.write("[LDM] Early stop.")
                break

    def load_ldm_pyg(self, path: str):
        self._collect_training_embeddings_pyg()
        self._build_ldm_pyg()
        state = torch.load(path, map_location=self.args.device, weights_only=True)
        self.ldm.load_state_dict(state)
        self.ldm.eval()

    @torch.no_grad()
    def _refine_embeddings_with_ldm(
        self, X: torch.Tensor, y: torch.Tensor, alpha: float = 0.3
    ):
        """评测期：先做一次轻量 refine（z 空间线性插值），再回原空间"""
        if self.ldm is None:
            return X
        T = int(self.args.time_steps)
        t_ref = max(1, int(0.1 * T))
        Xz = self.znorm.fwd(X) if self.use_znorm else X
        B, D = Xz.size()
        t = torch.full((B,), t_ref, dtype=torch.long, device=X.device)
        # 条件：同类均值（z）
        cond = torch.zeros_like(Xz)
        for c in torch.unique(y):
            m = y == c
            if m.any():
                cond[m] = Xz[m].mean(0, keepdim=True)
        x_t, _ = self.ldm.addnoise(Xz, t)
        out = self.ldm._predict_model_out(
            x_t,
            t,
            cond=cond,
            guidance=float(getattr(self.args, "ldm_guidance_eval", 0.0)),
            x0_sc=torch.zeros_like(x_t),
        )
        x0_hat = (
            self.ldm._x0_from_v(x_t, t, out)
            if self.ldm.predict == "v"
            else self.ldm._x0_from_eps(x_t, t, out)
        )
        Xz_ref = (1 - alpha) * Xz + alpha * x0_hat
        return self.znorm.inv(Xz_ref) if self.use_znorm else Xz_ref

    @torch.no_grad()
    def _generate_augmented_embeddings(self, X: torch.Tensor, y: torch.Tensor):
        """
        生成增强样本（z 空间采样）+ 质量过滤（双阈值）+ 每类配额 + 条件多样化
        返回：X_all, y_all, n_real
        """
        if (self.ldm is None) or (
            int(getattr(self.args, "num_augmented_samples", 0)) <= 0
        ):
            n_real = X.size(0)
            return X, y, n_real

        # 可配置的质量过滤阈值（提升性能和灵活性）
        cos_lo = float(
            getattr(self.args, "ldm_filter_cos_lo", 0.15)
        )  # 与类原型最小相似度
        cos_hi = float(
            getattr(self.args, "ldm_filter_cos_hi", 0.92)
        )  # 与最近真实样本最大相似度
        max_ratio = float(
            getattr(self.args, "ldm_aug_max_ratio", 1.50)
        )  # 每类最多扩增比例
        mix_alpha = float(
            getattr(self.args, "ldm_cond_mix_alpha", 0.30)
        )  # 条件混合比例
        cond_noise = float(getattr(self.args, "ldm_cond_noise", 0.05))  # 条件微噪声

        temp = float(getattr(self.args, "ldm_aug_temp", 0.9))
        guid = float(getattr(self.args, "ldm_guidance", 2.5))
        simp = bool(getattr(self.args, "ldm_aug_simple_var", True))
        per_real = int(getattr(self.args, "num_augmented_samples", 0))

        # → z 空间
        Xz = self.znorm.fwd(X) if self.use_znorm else X
        D = Xz.size(1)
        y = y.clone()

        # 原型（z）：按有序类列表构建
        classes = torch.unique(y, sorted=True)
        protos = [Xz[y == c].mean(0, keepdim=True) for c in classes]
        proto_z = torch.cat(protos, 0)  # [C,D]

        X_aug, y_aug = [], []
        for i, c in enumerate(classes):
            m = y == c
            Xc = Xz[m]  # 该类真实
            Nc = Xc.size(0)
            quota = min(Nc * per_real, int((1.0 + max_ratio) * Nc) - Nc)
            if quota <= 0:
                continue

            # 条件多样化：与同类随机样本混合 + 微噪声
            cond = proto_z[i].expand(quota, -1).clone()
            ridx = torch.randint(low=0, high=Nc, size=(quota,), device=Xc.device)
            cond = (1 - mix_alpha) * cond + mix_alpha * Xc[ridx]
            cond = cond + cond_noise * torch.randn_like(cond)

            # 采样
            z_gen = self.ldm.sample(
                shape=(quota, D),
                cond=cond,
                guidance=guid,
                simple_var=simp,
                temp=temp,
                init_match_radius=Xc.norm(dim=1).mean().expand(quota),  # 距离尺度
            )

            # 质量过滤（与原型远/与真实过近）
            zp = proto_z[i].expand(quota, -1)
            cos_to_proto = F.cosine_similarity(z_gen, zp, dim=1)
            keep1 = cos_to_proto >= cos_lo
            if keep1.any():
                z_gen = z_gen[keep1]
                quota1 = z_gen.size(0)
                # 性能优化：使用归一化向量进行余弦相似度计算
                Xc_n = F.normalize(Xc, dim=1)
                z_n = F.normalize(z_gen, dim=1)
                near = (z_n @ Xc_n.t()).max(dim=1).values
                keep2 = near <= cos_hi
                z_gen = z_gen[keep2]
            else:
                z_gen = z_gen[:0]

            if z_gen.numel() == 0:
                continue

            X_aug.append(z_gen)
            y_aug.append(
                torch.full(
                    (z_gen.size(0),), c.item(), dtype=torch.long, device=X.device
                )
            )

        if len(X_aug) == 0:
            print(f"⚠️  警告：所有生成样本都被质量过滤掉了，考虑放宽过滤阈值")
            n_real = X.size(0)
            return X, y, n_real

        # ← 回原空间
        Z = torch.cat(X_aug, 0)
        Xg = self.znorm.inv(Z) if self.use_znorm else Z

        X_all = torch.cat([X, Xg], 0)
        y_all = torch.cat([y, torch.cat(y_aug, 0)], 0)
        n_real = X.size(0)

        # 性能监控信息
        n_aug = Xg.size(0)
        print(
            f"✅ 增强样本生成: 原始={n_real}, 生成={n_aug}, 总计={X_all.size(0)} (将复用于所有测试任务)"
        )

        return X_all, y_all, n_real

    def _train_classifier_weighted(self, X, y, n_real):
        """加权 CE（真实权重大、生成权重小）+ 小概率 MixUp + 动态训练步数"""
        in_dim = X.size(1)
        num_cls = int(self.args.N_way)
        clf = LogReg(in_dim, num_cls).to(self.args.device)
        opt = torch.optim.SGD(clf.parameters(), lr=0.01)
        xent = nn.CrossEntropyLoss(reduction="none")

        # 优化的超参数
        w_real, w_gen = 1.0, 0.7  # 提升生成样本权重
        mix_prob, mix_beta = 0.12, 0.2  # 略降低MixUp概率

        # 动态训练步数：基于样本数量调整
        base_steps = 300
        data_factor = min(2.0, X.size(0) / 50.0)  # 样本越多，训练越久
        max_steps = int(base_steps * data_factor)

        best, best_state, wait, patience = 1e9, None, 0, max(50, max_steps // 6)

        for step in range(max_steps):
            opt.zero_grad()
            Xb, yb = X, y
            # mixup（小概率）
            if torch.rand(1).item() < mix_prob:
                lam = torch.distributions.Beta(mix_beta, mix_beta).sample().item()
                perm = torch.randperm(Xb.size(0), device=X.device)
                Xb = lam * Xb + (1 - lam) * Xb[perm]
                y_a, y_b = yb, yb[perm]
            else:
                lam, y_a, y_b = 1.0, yb, yb

            logits = clf(Xb)
            loss_vec = (
                (lam * xent(logits, y_a) + (1 - lam) * xent(logits, y_b))
                if lam < 1.0
                else xent(logits, yb)
            )
            # 优化的加权策略
            w = torch.ones_like(loss_vec)
            w[n_real:] = w_gen
            w[:n_real] = w_real
            loss = (w * loss_vec).mean() + 0.05 * sum(
                torch.norm(p) for p in clf.parameters()
            )  # 降低正则化
            loss.backward()
            opt.step()
            if loss.item() < best:
                best, best_state, wait = loss.item(), deepcopy(clf.state_dict()), 0
            else:
                wait += 1
            if wait > patience:
                break
        if best_state is not None:
            clf.load_state_dict(best_state)
        clf.eval()
        return clf

    def _sample_test_task(self, task_idx=None):
        """
        与 test_model 使用相同的任务采样接口：
        返回 (support_graphs, query_graphs)
        """
        if task_idx is None:
            # 简单轮转或随机
            task_idx = random.randint(0, self.dataset.test_task_Num - 1)
        return self.dataset.sample_one_task(task_idx)

    def test_model_with_ldm_pyg(self, num_test=20):
        """few-shot 测试（一次增强，多次复用）"""
        self.model.eval()
        self._build_ldm_pyg()

        # 确认LDM已训练或加载
        if self.ldm is None:
            raise RuntimeError(
                "LDM 尚未初始化，请先调用 train_ldm_pyg() 或 load_ldm_pyg()"
            )

        # === 一次性生成增强支持集（支持集固定不变） ===
        support_graphs, _ = self._sample_test_task(0)  # 获取固定支持集
        with torch.no_grad():
            s_batch = Batch.from_data_list(support_graphs).to(self.args.device)
            Xs = self.model.encode_graphs(s_batch)  # [S,D]
            ys = torch.tensor(
                [g.label for g in support_graphs], device=self.args.device
            ).long()

        # 一次性生成增强数据
        print("🚀 开始一次性生成增强支持集...")
        Xs_ref = self._refine_embeddings_with_ldm(Xs, ys, alpha=0.3)
        Xtrain_cached, ytrain_cached, n_real_cached = (
            self._generate_augmented_embeddings(Xs_ref, ys)
        )
        print(f"✅ 增强支持集生成完成，将复用于所有 {num_test} 个测试任务")

        # === 对每个测试任务使用相同的增强支持集 ===
        accs = []
        for test_i in range(num_test):
            # 只获取不同的查询集
            _, query_graphs = self._sample_test_task(test_i)

            # 编码查询集
            with torch.no_grad():
                q_batch = Batch.from_data_list(query_graphs).to(self.args.device)
                Xq = self.model.encode_graphs(q_batch)  # [Q,D]
                yq = torch.tensor(
                    [g.label for g in query_graphs], device=self.args.device
                ).long()

            # 使用缓存的增强支持集训练分类器
            clf = self._train_classifier_weighted(
                Xtrain_cached, ytrain_cached, n_real_cached
            )

            # 评测
            with torch.no_grad():
                logits = clf(Xq)
                pred = logits.argmax(dim=1)
                acc = (pred == yq).float().mean().item()
            accs.append(acc)

        accs = np.array(accs)
        return float(accs.mean()), float(accs.std())

    # 可以添加其他方法，如LDM相关的训练等
