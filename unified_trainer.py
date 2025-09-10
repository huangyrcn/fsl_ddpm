# unified_trainer.py  —— 直接整文件替换

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import os
from copy import deepcopy
from tqdm import tqdm
from torch.optim.lr_scheduler import ReduceLROnPlateau
import wandb
from sklearn.cluster import KMeans  # 仅保留以兼容，但默认改用 label proto

from gnn_model import Model, Prompt, LogReg
from train_ldm import LDM, finetune_param_filter
from dataset import Dataset
from aug import aug_fea_mask, aug_drop_node, aug_fea_drop, aug_fea_dropout


class ZScore:
    def __init__(self): 
        self.mu = None
        self.std = None
    def fit(self, X, eps=1e-6):
        self.mu = X.mean(0, keepdim=True)
        self.std = (X.var(0, unbiased=False, keepdim=True) + eps).sqrt()
    def fwd(self, X): 
        return (X - self.mu) / (self.std + 1e-12)
    def inv(self, Z): 
        return Z * (self.std + 1e-12) + self.mu


class UnifiedTrainer:
    """
    Encoder 预训练 → LDM 训练 → 测试 (refine→augment→线性分类)
    针对 Letter_high/TRIANGLES 做了更稳的增强与过滤。
    """
    def __init__(self, args, logf=None):
        self.args = args
        self.logf = logf
        self.save_dir = './savepoint'
        os.makedirs(self.save_dir, exist_ok=True)

        # ====== 数据 ======
        self.dataset = Dataset(args.dataset_name, args)
        args.train_classes_num = self.dataset.train_classes_num
        args.test_classes_num = self.dataset.test_classes_num
        args.node_fea_size = self.dataset.train_graphs[0].node_features.shape[1]
        args.N_way = self.dataset.test_classes_num

        # ====== Encoder ======
        self.model = Model(args).to(args.device)
        self.prompt = Prompt(self.args).to(self.args.device)
        self.encoder_optimizer = optim.Adam(self.model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        # ====== LDM（延迟初始化）======
        self.ldm = None
        self.ldm_optimizer = None

        # ====== 归一化空间（默认用 z-score；若配置 ldm_unit_sphere=True 则跳过）======
        self.use_znorm = bool(not getattr(args, "ldm_unit_sphere", False))
        self.znorm = ZScore() if self.use_znorm else None

        # ====== 评估：线性分类器 ======
        in_dim = self.model.sample_input_emb_size
        num_cls = self.args.N_way
        self.log = LogReg(in_dim, num_cls).to(self.args.device)
        self.xent = nn.CrossEntropyLoss(reduction='none')  # 用于加权 CE

        # ====== 针对 Letter_high/TRIANGLES 的更稳缺省 ======
        name = str(getattr(self.args, "dataset_name", "")).lower()
        self.is_letter = ("letter" in name)
        self.is_tri = ("triangle" in name or "triangles" in name)

        # 缺省策略（可在 YAML 覆盖）
        self.cfg_refine_first = bool(getattr(self.args, "refine_first", True if self.is_letter else False))
        self.cfg_refine_alpha = float(getattr(self.args, "refine_alpha", 0.25 if self.is_letter else 0.2))
        self.cfg_aug_ratio    = float(getattr(self.args, "aug_per_class_max_ratio", 0.5 if self.is_letter else 1.0))
        self.cfg_cos_lo       = float(getattr(self.args, "aug_cosine_min_to_proto", 0.45 if self.is_letter else 0.35))
        self.cfg_cos_hi       = float(getattr(self.args, "aug_cosine_max_to_nn", 0.96 if self.is_letter else 0.98))
        self.cfg_mix_alpha    = float(getattr(self.args, "aug_mix_alpha", 0.6 if self.is_letter else 0.5))
        self.cfg_cond_noise   = float(getattr(self.args, "aug_cond_noise_std", 0.10 if self.is_letter else 0.15))
        self.cfg_real_w       = float(getattr(self.args, "clf_real_weight", 1.0))
        self.cfg_gen_w        = float(getattr(self.args, "clf_gen_weight", 0.5 if self.is_letter else 0.7))
        self.cfg_mixup_p      = float(getattr(self.args, "clf_mixup_p", 0.2 if self.is_letter else 0.1))
        self.cfg_mixup_alpha  = float(getattr(self.args, "clf_mixup_alpha", 0.4 if self.is_letter else 0.3))

    # ===================== 阶段1：Encoder =====================
    def load_pretrained_encoder(self, ckpt_path=None):
        if ckpt_path is None:
            ckpt_path = os.path.join(self.save_dir, f'{self.args.dataset_name}_encoder.pkl')
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"找不到encoder权重文件: {ckpt_path}")
        state = torch.load(ckpt_path, map_location=self.args.device, weights_only=True)
        self.model.load_state_dict(state)
        self.model.eval()
        print(f"已从 {os.path.basename(ckpt_path)} 加载Encoder权重，并切换到eval模式")

    def train_encoder(self):
        print("=== 阶段1：开始训练Encoder（对比学习） ===")
        best = 1e9; cnt_wait = 0
        graph_copy_2 = deepcopy(self.dataset.train_graphs)
        # 图增强（可在 YAML 配置 aug1/aug2）
        def _apply_aug(graphs, name):
            if name == 'identity':        return graphs
            if name == 'node_drop':       return aug_drop_node(graphs, self.args.seed)
            if name == 'feature_mask':    return aug_fea_mask(graphs, self.args.seed)
            if name == 'feature_drop':    return aug_fea_drop(graphs, self.args.seed)
            if name == 'feature_dropout': return aug_fea_dropout(graphs, self.args.seed)
            return graphs
        graph_aug1 = _apply_aug(self.dataset.train_graphs, self.args.aug1)
        graph_aug2 = _apply_aug(graph_copy_2, self.args.aug2)
        print("图增强完成!")
        for i in tqdm(range(self.args.epoch_num), desc="训练Encoder"):
            loss = self._pretrain_step(graph_aug1, graph_aug2)
            if loss is None: continue
            if loss < best:
                best = loss; cnt_wait = 0
                torch.save(self.model.state_dict(), os.path.join(self.save_dir, f'{self.args.dataset_name}_encoder.pkl'))
            else:
                cnt_wait += 1
            if i % 50 == 0:
                tqdm.write(f'Epoch {i} Loss {loss:.4f}')
                if self.logf: self.logf.write(f'Epoch {i} Loss {loss:.4f}\n')
            if cnt_wait > self.args.patience:
                tqdm.write("提前停止!")
                break
        print("Encoder 训练完成。")

    def _pretrain_step(self, graph_aug1, graph_aug2):
        self.model.train()
        z1 = self.model(graph_aug1)
        z2 = self.model(graph_aug2)
        loss = self.model.loss_cal(z1, z2)
        self.encoder_optimizer.zero_grad(); loss.backward(); self.encoder_optimizer.step()
        return loss

    # ===================== 阶段2：LDM =====================
    def init_ldm_components(self):
        print("=== 初始化LDM组件 ===")
        embedding_dim = self.model.sample_input_emb_size
        predict_type   = getattr(self.args, "ldm_predict", "v")
        unit_sphere    = getattr(self.args, "ldm_unit_sphere", False)
        widths         = getattr(self.args, "ldm_widths", (128, 256, 512))
        n_blocks       = getattr(self.args, "ldm_n_blocks", 2)
        use_zero_mlp   = getattr(self.args, "ldm_use_zero_mlp", True)
        time_steps     = int(self.args.time_steps)

        self.ldm = LDM(
            device=self.args.device,
            latent_dim=embedding_dim,
            timesteps=time_steps,
            cond_dim=embedding_dim,
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
        if ckpt_path is None:
            ckpt_path = os.path.join(self.save_dir, f'{self.args.dataset_name}_ldm.pkl')
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"找不到LDM权重文件: {ckpt_path}")
        self.ldm.load_state_dict(torch.load(ckpt_path, map_location=self.args.device, weights_only=True))
        self.ldm.eval()
        print(f"已从 {os.path.basename(ckpt_path)} 加载LDM权重，并切换到eval模式")

    def collect_training_embeddings(self):
        print("=== 收集训练集embeddings（并构造训练条件） ===")
        self.model.eval()
        if self.args.use_prompt:
            self.prompt.eval(); prompt = self.prompt()
        else:
            prompt = torch.zeros(self.args.num_token, self.args.node_fea_size, device=self.args.device)

        with torch.no_grad():
            all_g = self.dataset.train_graphs
            E = self.model.encode_graphs(all_g, prompt)            # [N,D]
            Y = torch.tensor([g.label for g in all_g], device=self.args.device)

        self.training_embeddings = E
        self.training_labels = Y

        # → z 空间
        if self.use_znorm:
            self.znorm.fit(E)
            Ez = self.znorm.fwd(E)
        else:
            Ez = E if not bool(getattr(self.args, 'ldm_unit_sphere', False)) else F.normalize(E, dim=1)
        self.training_embeddings_z = Ez

        # 条件：**改为 label prototype（监督的类均值）**，避免 KMeans 与标签错位
        prototypes = torch.zeros(self.args.train_classes_num, Ez.size(1), device=self.args.device)
        for c in torch.unique(Y):
            m = (Y == c)
            if m.any():
                prototypes[int(c.item())] = Ez[m].mean(0)
        self.label_prototypes_z = prototypes
        self.conditions_z = prototypes[Y]  # 每样本取其类原型作为 cond
        print("条件改为 label-prototype，训练/推理保持一致。")

    def train_ldm(self):
        print("=== 开始训练LDM（cond=label prototype） ===")
        conditions_z = self.conditions_z
        best_loss = float('inf'); patience = int(getattr(self.args, "patience_ldm", 300)); wait = 0
        pbar = tqdm(range(1, int(getattr(self.args, "num_epochs_ldm", 2000)) + 1), desc="LDM Training")
        bs = int(self.args.ldm_batch_size)

        for ep in pbar:
            perm = torch.randperm(self.training_embeddings_z.size(0), device=self.args.device)
            Ez = self.training_embeddings_z[perm]
            Cz = conditions_z[perm]
            tot = 0.0; n = 0
            for i in range(0, Ez.size(0), bs):
                loss = self._train_ldm_step(Ez[i:i+bs], Cz[i:i+bs])
                tot += loss.item(); n += 1
            avg = tot / max(1, n)
            pbar.set_postfix({'Loss': f'{avg:.4f}'})
            if wandb.run is not None: wandb.log({"ldm_loss": avg}, step=ep)

            if avg < best_loss:
                best_loss = avg; wait = 0
                torch.save(self.ldm.state_dict(), os.path.join(self.save_dir, f'{self.args.dataset_name}_ldm.pkl'))
            else:
                wait += 1
                if wait >= patience: break

    def _train_ldm_step(self, z, conditions):
        self.ldm.train(); self.ldm_optimizer.zero_grad()
        p_uncond     = float(getattr(self.args, "ldm_p_uncond", 0.1))
        lambda_proto = float(getattr(self.args, "ldm_lambda_proto", 0.0))
        loss = self.ldm.loss(x0=z, cond=conditions, p_uncond=p_uncond, lambda_proto=lambda_proto, proto=conditions)
        loss.backward(); self.ldm_optimizer.step()
        return loss

    # ===================== 阶段3：统一测试 =====================
    def test_model(self, num_augmented_samples=0, ldm_model=None, test_name=None):
        self.model.eval()
        need_ldm = bool(num_augmented_samples > 0)
        label = "LDM增强评估" if need_ldm else "原始Encoder评估"
        test_name = test_name or label
        print(f"=== {test_name} ===")

        test_accs = []
        start = 0
        total_tasks = (len(self.dataset.test_graphs) - self.args.K_shot * self.args.test_classes_num) // (self.args.N_way * self.args.query_size)
        pbar = tqdm(total=total_tasks, desc=test_name)
        end = start + total_tasks * (self.args.N_way * self.args.query_size)

        while start < end:
            acc = self._evaluate_one_task_with_ldm(start, num_augmented_samples, ldm_model, start_test_idx=start)
            test_accs.append(acc)
            if self.logf: self.logf.write(f"任务起始 {start} {label}: {acc:.4f}\n")
            pbar.write(f"任务 {len(test_accs)}: {label} = {acc:.4f}")
            pbar.update(1); pbar.set_postfix({'Acc': f'{acc:.4f}'})
            start += self.args.N_way * self.args.query_size
        pbar.close()

        mean_acc = float(np.mean(test_accs)); std = float(np.std(test_accs))
        print(f'{test_name}: {mean_acc:.4f} ± {std:.4f}')
        if self.logf: self.logf.write(f'{test_name}: {mean_acc:.4f} ± {std:.4f}\n')
        return mean_acc, std

    def _evaluate_one_task_with_ldm(self, test_idx, num_augmented_samples, ldm_model, start_test_idx=None):
        self.model.eval()
        ldm_for_sampling = ldm_model or self.ldm
        prompt = torch.zeros(self.args.num_token, self.args.node_fea_size, device=self.args.device)

        # 任务采样
        first_N = np.array(list(range(self.dataset.test_classes_num)))
        task = self.dataset.sample_one_task(self.dataset.test_tasks, first_N, K_shot=self.args.K_shot,
                                            query_size=self.args.query_size, test_start_idx=test_idx)

        # 支持集
        sup_emb, _ = self.model.sample_input_GNN([task], prompt, True)
        sup = sup_emb.reshape(self.args.N_way, -1, self.model.sample_input_emb_size)[:, :self.args.K_shot, :]\
                    .reshape(self.args.N_way * self.args.K_shot, self.model.sample_input_emb_size).detach()
        sup_y = []
        for gs in task['support_set']: sup_y.append(np.array([g.label for g in gs[:self.args.K_shot]]))
        sup_y = torch.LongTensor(np.hstack(sup_y)).to(self.args.device)

        # 先 refine（更稳），再 augment（小比例）
        if self.cfg_refine_first and (ldm_for_sampling is not None):
            sup = self.refine_embeddings_with_ldm(sup, sup_y, ldm_for_sampling,
                                                  alpha=self.cfg_refine_alpha,
                                                  batch_size=int(self.args.ldm_batch_size))

        if int(num_augmented_samples) > 0 and (ldm_for_sampling is not None):
            sup_aug, sup_y_aug, n_real = self.generate_augmented_embeddings(
                embeddings=sup, labels=sup_y, num_to_generate=int(num_augmented_samples),
                ldm_model=ldm_for_sampling, device=self.args.device, condition_type="label_proto",
                # 传入更稳的过滤与上限（Letter_high 默认更严格）
                cos_lo=self.cfg_cos_lo, cos_hi=self.cfg_cos_hi, max_ratio=self.cfg_aug_ratio,
                mix_alpha=self.cfg_mix_alpha, cond_noise_std=self.cfg_cond_noise
            )
        else:
            sup_aug, sup_y_aug, n_real = sup, sup_y, sup.size(0)

        # 线性分类器（真实样本大权重，生成样本小权重，且可选 MixUp）
        self.log = LogReg(self.model.sample_input_emb_size, self.args.N_way).to(self.args.device)
        self._train_classifier(sup_aug, sup_y_aug, n_real=n_real)

        # 查询集
        q_emb, _ = self.model.sample_input_GNN([task], prompt, False)
        q = q_emb.reshape(self.args.N_way, self.args.query_size, self.model.sample_input_emb_size)\
                 .reshape(self.args.N_way * self.args.query_size, self.model.sample_input_emb_size).detach()
        q_y = []
        for gs in task['query_set']: q_y.append(np.array([g.label for g in gs]))
        q_y = torch.LongTensor(np.hstack(q_y)).to(self.args.device)

        if task.get('append_count', 0) != 0:
            q = q[: q_y.size(0) - task['append_count'], :]
            q_y = q_y[: q_y.size(0) - task['append_count']]

        logits = self.log(q)
        preds = torch.argmax(logits, dim=1)
        acc = torch.mean((preds == q_y).float()).item()
        return acc

    # ===== 任务级微调（保留，默认关闭）=====
    def _task_level_finetune(self, support_data, support_labels, task_id=None):
        steps = int(getattr(self.args, "task_finetune_steps", 0))
        if steps <= 0: return
        trainable = []
        for n, p in self.ldm.named_parameters():
            flag = finetune_param_filter(n)
            p.requires_grad_(flag)
            if flag: trainable.append(p)
        support_z = self.znorm.fwd(support_data) if self.use_znorm else support_data
        with torch.no_grad():
            cond_z = torch.zeros_like(support_z)
            for cls in torch.unique(support_labels):
                m = (support_labels == cls)
                if m.any(): cond_z[m] = support_z[m].mean(0, keepdim=True)
        lr = float(getattr(self.args, "task_finetune_lr", 1e-3))
        wd = float(getattr(self.args, "task_finetune_weight_decay", 1e-4))
        opt = torch.optim.AdamW(trainable, lr=lr, weight_decay=wd)
        scheduler = ReduceLROnPlateau(opt, mode='min', factor=0.5, patience=50, threshold=1e-3,
                                      cooldown=0, min_lr=1e-7, verbose=True)
        self.ldm.train()
        drop_p = float(getattr(self.args, "task_finetune_cond_dropout", 0.1))
        grad_clip = float(getattr(self.args, "task_finetune_grad_clip", 1.0))
        patience = int(getattr(self.args, "task_finetune_patience", 50))
        lambda_proto = float(getattr(self.args, "task_lambda_proto", 0.0))
        accum_steps = int(getattr(self.args, "task_finetune_accum_steps", 4))
        ema_alpha = float(getattr(self.args, "task_finetune_ema_alpha", 0.98))
        ema_loss = None; best_loss = float('inf'); best_state = None; wait = 0

        pbar = tqdm(range(steps), desc=f"Task Finetune ({task_id})", position=1, leave=False)
        for step in pbar:
            cond_in = cond_z.clone()
            if drop_p > 0:
                mask = (torch.rand(cond_in.size(0), device=cond_in.device) < drop_p).float().unsqueeze(-1)
                cond_in = cond_in * (1 - mask)
            loss = self.ldm.loss(x0=support_z, cond=cond_in, p_uncond=0.0,
                                 lambda_proto=lambda_proto, proto=cond_z)
            (loss / float(max(1, accum_steps))).backward()
            cur = loss.item()
            ema_loss = cur if ema_loss is None else (ema_alpha * ema_loss + (1 - ema_alpha) * cur)
            if (step + 1) % max(1, accum_steps) == 0:
                torch.nn.utils.clip_grad_norm_(trainable, grad_clip)
                opt.step(); opt.zero_grad(set_to_none=True)
                scheduler.step(ema_loss)
            pbar.set_postfix({'Loss': f'{cur:.4f}', 'EMA': f'{(ema_loss if ema_loss is not None else cur):.4f}'})
            if ema_loss < best_loss: best_loss, wait, best_state = ema_loss, 0, deepcopy(self.ldm.state_dict())
            else:
                wait += 1
                if wait >= patience: break
        if best_state is not None: self.ldm.load_state_dict(best_state)
        self.ldm.eval()

    # ===== 线性分类器：加权 + 可选 MixUp =====
    def _train_classifier(self, X, y, n_real):
        self.log.train()
        params = [{'params': self.log.parameters()}]
        if getattr(self.args, 'use_prompt', True):
            params.append({'params': self.prompt.parameters()})
        opt = torch.optim.SGD(params, lr=0.01, momentum=0.0)

        real_w = float(getattr(self.args, "clf_real_weight", self.cfg_real_w))
        gen_w  = float(getattr(self.args, "clf_gen_weight",  self.cfg_gen_w))
        mix_p  = float(getattr(self.args, "clf_mixup_p",     self.cfg_mixup_p))
        mix_a  = float(getattr(self.args, "clf_mixup_alpha", self.cfg_mixup_alpha))

        w = torch.ones(X.size(0), device=X.device) * gen_w
        w[:n_real] = real_w  # 真实样本更大权重

        best, best_state, wait, patience = 1e9, None, 0, 100
        for _ in range(500):
            opt.zero_grad()
            Xb, yb, wb = X, y, w
            # 可选 MixUp（避免生成样本把边界拉偏）
            if torch.rand(1).item() < mix_p and Xb.size(0) >= 2:
                lam = np.random.beta(mix_a, mix_a)
                idx = torch.randperm(Xb.size(0), device=Xb.device)
                Xb = lam * Xb + (1 - lam) * Xb[idx]
                y_onehot = F.one_hot(yb, num_classes=self.args.N_way).float()
                yb_soft = lam * y_onehot + (1 - lam) * y_onehot[idx]
                logits = self.log(Xb)
                loss_vec = F.cross_entropy(logits, yb_soft, reduction='none')
            else:
                logits = self.log(Xb)
                loss_vec = F.cross_entropy(logits, yb, reduction='none')

            # 加权 CE + L2
            loss = (loss_vec * wb).mean()
            l2_reg = torch.tensor(0., device=X.device)
            for p in self.log.parameters(): l2_reg += torch.norm(p)
            loss = loss + 0.1 * l2_reg

            loss.backward(); opt.step()
            if loss < best:
                best, wait, best_state = loss.item(), 0, deepcopy(self.log.state_dict())
            else:
                wait += 1
            if wait > patience: break

        if best_state is not None: self.log.load_state_dict(best_state)
        self.log.eval()

    # ===== 生成：多样化条件 + 余弦双阈值过滤 + 每类配额上限 =====
    @torch.no_grad()
    def generate_augmented_embeddings(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor,
        num_to_generate: int,
        *,
        ldm_model,
        device,
        condition_type: str = "label_proto",
        cos_lo: float = None, cos_hi: float = None,
        max_ratio: float = None,
        mix_alpha: float = None,
        cond_noise_std: float = None,
    ):
        if (num_to_generate <= 0) or (ldm_model is None) or (embeddings.numel() == 0):
            return embeddings, labels, embeddings.size(0)

        D = embeddings.shape[1]
        dev = embeddings.device
        # 采样参数（继承你的默认）
        temp = float(getattr(self.args, "ldm_aug_temp", 0.9))
        simp = bool(getattr(self.args, "ldm_aug_simple_var", False))
        guid = float(getattr(self.args, "ldm_guidance", 1.8))

        # Letter_high/其它数据集的默认
        cos_lo = self.cfg_cos_lo if cos_lo is None else cos_lo
        cos_hi = self.cfg_cos_hi if cos_hi is None else cos_hi
        max_ratio = self.cfg_aug_ratio if max_ratio is None else max_ratio
        mix_alpha = self.cfg_mix_alpha if mix_alpha is None else mix_alpha
        cond_noise_std = self.cfg_cond_noise if cond_noise_std is None else cond_noise_std

        # → z 空间
        Ez = self.znorm.fwd(embeddings) if self.use_znorm else embeddings

        # 类原型（z 空间）
        proto_z = torch.zeros_like(Ez)
        uniq = torch.unique(labels)
        for c in uniq:
            m = (labels == c)
            if m.any(): proto_z[m] = Ez[m].mean(0, keepdim=True)

        # 与同类随机样本混合，增加多样性
        idx_all = torch.arange(Ez.size(0), device=dev)
        rand_idx_list = []
        for c in uniq.tolist():
            idx_c = (labels == c).nonzero(as_tuple=False).squeeze(1)
            if len(idx_c) <= 1:
                rand_idx_list.append(idx_c)
            else:
                rand_idx_list.append(idx_c[torch.randperm(len(idx_c), device=dev)])
        rand_idx = torch.cat(rand_idx_list, dim=0)
        rand_same_z = Ez[rand_idx]
        cond_base = mix_alpha * proto_z + (1.0 - mix_alpha) * rand_same_z
        if cond_noise_std > 0:
            cond_base = cond_base + cond_noise_std * torch.randn_like(cond_base)

        # 每类配额：不超过 (1+max_ratio)*原始数量
        per_k = int(max(1, num_to_generate))
        target_repeats = torch.full((Ez.size(0),), per_k, dtype=torch.long, device=dev)
        count = {int(c.item()): int((labels == c).sum().item()) for c in uniq}
        cap   = {c: int(round((1.0 + max_ratio) * n)) for c, n in count.items()}
        for c in uniq.tolist():
            m = (labels == c)
            want = int(target_repeats[m].sum().item())
            room = max(0, cap[c] - count[c])
            if want > room:
                flat = m.nonzero(as_tuple=False).squeeze(1)
                expanded = torch.repeat_interleave(flat, per_k)
                keep = torch.randperm(len(expanded), device=dev)[:room]
                kept = expanded[keep]
                new_rep = torch.zeros_like(target_repeats[m])
                pos_map = {int(flat[i].item()): i for i in range(len(flat))}
                for j in kept.tolist():
                    new_rep[pos_map[int(j)]] += 1
                target_repeats[m] = new_rep

        if int(target_repeats.sum().item()) == 0:
            return embeddings, labels, embeddings.size(0)

        conds = cond_base.repeat_interleave(target_repeats, dim=0)
        labs  = labels.repeat_interleave(target_repeats, dim=0)
        init_r = Ez.norm(dim=1).repeat_interleave(target_repeats, dim=0)

        z_gen = ldm_model.sample(
            shape=(conds.size(0), D),
            cond=conds,
            guidance=guid,
            simple_var=simp,
            temp=temp,
            init_match_radius=init_r
        )

        # 过滤：与类原型余弦过低（过远）或与最近真实过高（过近/重复）
        proto_unit = torch.zeros_like(proto_z)
        for c in uniq:
            m = (labels == c)
            if m.any():
                p = Ez[m].mean(0, keepdim=True)
                proto_unit[m] = F.normalize(p, dim=1)

        Ez_u = F.normalize(Ez, dim=1)
        Zg_u = F.normalize(z_gen, dim=1)
        proto_for_gen = proto_unit[labels.repeat_interleave(target_repeats)]
        cos_to_proto = (Zg_u * proto_for_gen).sum(dim=1)

        # batched 最近邻同类相似度
        def _batched_max_cos(gen_u, yg, real_u, yr, bs=2048):
            out = torch.empty(gen_u.size(0), device=gen_u.device)
            for i in range(0, gen_u.size(0), bs):
                j = min(i + bs, gen_u.size(0))
                g = gen_u[i:j]; yb = yg[i:j]
                sims = []
                for c in torch.unique(yb):
                    gm = (yb == c).nonzero(as_tuple=False).squeeze(1)
                    rm = (yr == c).nonzero(as_tuple=False).squeeze(1)
                    if len(rm) == 0: continue
                    s = g[gm] @ real_u[rm].t()
                    sims.append(s.max(dim=1).values if s.numel() > 0 else torch.full((len(gm),), -1.0, device=g.device))
                out[i:j] = torch.cat(sims, dim=0) if len(sims) else torch.full((j-i,), -1.0, device=g.device)
            return out

        max_cos_to_nn = _batched_max_cos(Zg_u, labs, Ez_u, labels, bs=int(getattr(self.args, "aug_filter_batch_size", 2048)))
        keep = (cos_to_proto >= cos_lo) & (max_cos_to_nn <= cos_hi)
        if keep.sum() == 0:
            keep = (cos_to_proto >= (cos_lo * 0.8))  # 放宽一次，避免全空
        z_keep = z_gen[keep]; y_keep = labs[keep]
        if z_keep.numel() == 0:
            return embeddings, labels, embeddings.size(0)

        gen = self.znorm.inv(z_keep) if self.use_znorm else z_keep
        X = torch.cat([embeddings, gen], 0)
        Y = torch.cat([labels, y_keep], 0)
        n_real = embeddings.size(0)
        return X, Y, n_real

    # ===== 轻量 refine（默认给 Letter_high 打开）=====
    @torch.no_grad()
    def refine_embeddings_with_ldm(self, embeddings, labels, ldm_model, alpha=0.25, batch_size=256):
        ldm_model.eval()
        T = ldm_model.timesteps
        t_ref = max(1, int(0.1 * T))
        x0z = self.znorm.fwd(embeddings) if self.use_znorm else embeddings
        outs = []
        for i in range(0, x0z.size(0), batch_size):
            xb = x0z[i:i+batch_size]
            B = xb.size(0)
            t = torch.full((B,), t_ref, dtype=torch.long, device=xb.device)
            # 同类条件 = 局部类均值
            yb = labels[i:i+B]
            cb = torch.zeros_like(xb)
            for c in torch.unique(yb):
                m = (yb == c)
                if m.any(): cb[m] = xb[m].mean(0, keepdim=True)
            xt, _ = ldm_model.addnoise(xb, t)
            out = ldm_model._predict_model_out(xt, t, cond=cb, guidance=float(getattr(self.args,"ldm_guidance",1.8)),
                                               x0_sc=torch.zeros_like(xt))
            x0_hat = ldm_model._x0_from_v(xt, t, out) if ldm_model.predict=='v' else ldm_model._x0_from_eps(xt, t, out)
            outs.append((1 - alpha) * xb + alpha * x0_hat)
        zr = torch.cat(outs, 0)
        return self.znorm.inv(zr) if self.use_znorm else zr

    # =====================（可选）内在质量评估 =====================
    @staticmethod
    def _mmd_rbf(X: torch.Tensor, Y: torch.Tensor, sigmas=(0.5, 1.0, 2.0, 4.0)) -> torch.Tensor:
        def pdist2(a, b):
            a2 = (a * a).sum(1, keepdim=True)
            b2 = (b * b).sum(1, keepdim=True).t()
            return a2 - 2 * (a @ b.t()) + b2
        dxx = pdist2(X, X); dyy = pdist2(Y, Y); dxy = pdist2(X, Y)
        Kxx = X.new_zeros(()); Kyy = X.new_zeros(()); Kxy = X.new_zeros(())
        for s in sigmas:
            g = torch.exp(-dxx / (2 * s * s)); Kxx = Kxx + (g.sum() - g.diag().sum()) / (X.size(0) * (X.size(0) - 1))
            g = torch.exp(-dyy / (2 * s * s)); Kyy = Kyy + (g.sum() - g.diag().sum()) / (Y.size(0) * (Y.size(0) - 1))
            g = torch.exp(-dxy / (2 * s * s)); Kxy = Kxy + g.mean()
        return Kxx + Kyy - 2 * Kxy

    @staticmethod
    def _frechet_distance(X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
        mu1, mu2 = X.mean(0), Y.mean(0)
        diff = mu1 - mu2
        Xc, Yc = X - mu1, Y - mu2
        C1 = (Xc.t() @ Xc) / max(1, (X.size(0) - 1))
        C2 = (Yc.t() @ Yc) / max(1, (Y.size(0) - 1))
        e1, V1 = torch.linalg.eigh(C1); e1 = e1.clamp_min(0)
        C1_sqrt = (V1 * e1.sqrt().unsqueeze(0)) @ V1.t()
        S = C1_sqrt @ C2 @ C1_sqrt
        es, _ = torch.linalg.eigh(S); es = es.clamp_min(0)
        tr_sqrt = es.sqrt().sum()
        fid = (diff @ diff) + torch.trace(C1) + torch.trace(C2) - 2.0 * tr_sqrt
        return fid.clamp_min(0.0)

    @staticmethod
    def _pairwise_stats(X: torch.Tensor) -> dict:
        with torch.no_grad():
            d = torch.cdist(X, X, p=2)
            triu = torch.triu_indices(d.size(0), d.size(1), offset=1)
            vals = d[triu[0], triu[1]]
            return {
                'dist_mean': float(vals.mean().item()),
                'dist_std': float(vals.std(unbiased=False).item()),
                'dist_p50': float(vals.median().item())
            }

    @staticmethod
    def _nn_distances(X: torch.Tensor, Y: torch.Tensor) -> dict:
        with torch.no_grad():
            d = torch.cdist(X, Y, p=2)
            mins = d.min(dim=1).values
            return {
                'nn_mean': float(mins.mean().item()),
                'nn_std': float(mins.std(unbiased=False).item()),
                'nn_p50': float(mins.median().item())
            }

    @torch.no_grad()
    def evaluate_ldm_intrinsic(self, num_samples: int = 1000, log_to_wandb: bool = True) -> dict:
        assert self.ldm is not None, "LDM not initialized"
        device = self.args.device
        E = self.training_embeddings.detach().to(device)
        N = E.size(0)
        sel = torch.randperm(N, device=device)[:min(N, num_samples)]
        real = E[sel]
        if self.use_znorm: real_z = self.znorm.fwd(real)
        else: real_z = F.normalize(real, dim=1) if bool(getattr(self.args, 'ldm_unit_sphere', False)) else real
        D = real_z.size(1)
        labels = getattr(self, "training_labels", None)
        labels = labels[sel] if labels is not None else None
        if labels is not None:
            cond = torch.zeros_like(real_z)
            for c in torch.unique(labels):
                m = (labels == c)
                if m.any(): cond[m] = real_z[m].mean(0, keepdim=True)
        else:
            cond = real_z
        temp = float(getattr(self.args, 'ldm_eval_temp', 0.85))
        simp = bool(getattr(self.args, 'ldm_eval_simple_var', False))
        guid = float(getattr(self.args, 'ldm_guidance_eval', 0.0))
        init_r = real_z.norm(dim=1)
        gen_z = self.ldm.sample(shape=(len(cond), D), cond=cond, guidance=guid, simple_var=simp, temp=temp,
                                init_match_radius=init_r)
        gen = self.znorm.inv(gen_z) if self.use_znorm else gen_z
        mmd = float(self._mmd_rbf(real, gen).item())
        fd  = float(self._frechet_distance(real, gen).item())
        metrics = {
            'mmd_rbf': mmd,
            'frechet_distance': fd,
            'real_pairwise': self._pairwise_stats(real),
            'gen_pairwise': self._pairwise_stats(gen),
            'gen_to_real_nn': self._nn_distances(gen, real),
        }
        if log_to_wandb and wandb.run is not None:
            wandb.log({'ldm_intrinsic/mmd_rbf': mmd, 'ldm_intrinsic/frechet_distance': fd}, commit=False)
        return metrics
