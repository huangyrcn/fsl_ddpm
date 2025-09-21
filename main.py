import os
import json
import random
import numpy as np
import torch


from omegaconf import OmegaConf

import subprocess   
from unified_trainer import UnifiedTrainer


import wandb

def build_cfg():
    """构建配置：优先级 CLI > 配置文件 > 代码默认值。
    - 从 CLI 读取（OmegaConf.from_cli），支持传入 config=path.yaml
    - 若未提供 config，使用默认路径 'configs/TRIANGLES_ldm.yaml'
    - 以代码内默认值为最低优先级
    """
    # 代码默认值（最低优先级）
    default_config = {
        'seed': 42,
        'dataset_name': 'TRIANGLES',
        'baseline_mode': None,
        'N_way': 3,
        'K_shot': 5,
        'query_size': 10,
        'patience': 5,
        'dropout': 0.5,
        'sample_input_size': 256,
        'gin_layer': 3,
        'gin_hid': 128,
        'aug1': 'node_drop',
        'aug2': 'feature_mask',
        't': 0.2,
        'lr': 0.001,
        'weight_decay': 1e-7,
        'eval_interval': 100,
        'epoch_num': 3000,
        'use_select_sim': False,
        'save_test_emb': True,
        'num_token': 1,
        'use_prompt': False,
        'device': 'cuda',
        'graph_pooling_type': 'sum',
        # LDM 相关默认
        'use_pretrained_encoder': False,
        'use_pretrained_ldm': True,
        'num_augmented_samples': 0,
        'learning_rate_ldm': 1e-4,
        'weight_decay_ldm': 1e-4,
        'time_steps': 50,
        'beta_start': 1e-4,
        'beta_end': 2e-2,
        'ldm_batch_size': 512,
        'ldm_es_interval': 100,
        'condition_type': 'class_proto',
        'batch_size_for_embedding': 512,
        # wandb 默认
        'use_wandb': False,
        'wandb_online': True,
        # 内在质量评估开关
        'evaluate_ldm_intrinsic': False,
        'evaluate_ldm_intrinsic_num_samples': 1000,
    }
    defaults = OmegaConf.create(default_config)

    # CLI（最高优先级），支持 config=xxx 覆盖路径
    cli = OmegaConf.from_cli()
    config_path = cli.get('config', 'configs/Letter_high_ldm.yaml')

    # 配置文件（中间优先级）
    base = OmegaConf.load(config_path)
    # 合并：默认 < 文件 < CLI
    cfg = OmegaConf.merge(defaults, base, cli)

    # 清理：不把 config 路径残留在最终配置里
    if 'config' in cfg:
        cfg.pop('config')


    cfg.device = select_device(cfg.device)
    return cfg
    
def select_device(device_pref):
    # 规范化输入
    s = str(device_pref).lower().strip()

    # 优先选择 CPU
    if s == 'cpu':
        return 'cpu'

    # CUDA 设备选择（显式索引）
    if s.startswith('cuda:'):
        if torch.cuda.is_available():
            return s
        return 'cpu'

    if s in ('cuda', 'gpu', 'auto'):
        if torch.cuda.is_available():
            try:
                out = subprocess.check_output([
                    'nvidia-smi',
                    '--query-gpu=memory.used,utilization.gpu',
                    '--format=csv,noheader,nounits'
                ], stderr=subprocess.DEVNULL, timeout=1.5)

                lines = out.decode().strip().splitlines()
                stats = []
                for i, line in enumerate(lines):
                    parts = [p.strip() for p in line.split(',')]
                    if len(parts) != 2:
                        continue
                    mem, util = int(parts[0]), int(parts[1])
                    stats.append((i, mem, util))

                if not stats:
                    return 'cuda:0'

                # 按显存使用和 GPU 利用率排序，选择最优的 GPU
                stats.sort(key=lambda x: (x[1], x[2]))
                best_idx = stats[0][0]
                return f'cuda:{best_idx}'
            except Exception:
                # nvidia-smi 不可用或失败，退回第一个可用 CUDA
                return 'cuda:0'
        # CUDA 不可用
        return 'cpu'

    # 兜底（GPU > CPU）
    if torch.cuda.is_available():
        return 'cuda:0'
    return 'cpu'

def main():
    cfg = build_cfg()
    
    # 初始化 wandb（如果配置中启用）
    if cfg.use_wandb:
        wandb.init(
            project=cfg.wandb_project if 'wandb_project' in cfg else 'fsl_ddpm',
            entity=cfg.wandb_entity if 'wandb_entity' in cfg else None,
            name=cfg.wandb_run_name if 'wandb_run_name' in cfg else None,
            config=OmegaConf.to_container(cfg, resolve=True),
            tags=cfg.wandb_tags if 'wandb_tags' in cfg else [],
            notes=cfg.wandb_notes if 'wandb_notes' in cfg else '',
            mode='online' if cfg.wandb_online else 'disabled'
        )
        print(f"✅ wandb 初始化成功: {wandb.run.name}")
    else:
        print("ℹ️ wandb 未启用")

    # 设置随机性

    os.environ['PYTHONHASHSEED'] = str(cfg.seed)
    random.seed(cfg.seed); np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed); torch.cuda.manual_seed(cfg.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


    #保存点配置
    results_dir = './our_results'
    save_dir = './savepoint'
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(save_dir, exist_ok=True)

    dataset = cfg.dataset_name

    text_file_name = os.path.join(results_dir, f"{dataset}-{ cfg.K_shot}-shot.txt")



    with open(text_file_name, 'w') as logf:
        # 写入最终配置（YAML）
        logf.write("=== Config (CLI > File > Defaults) ===\n")
        logf.write(OmegaConf.to_yaml(cfg))
        logf.write("\n")
        print(cfg)
        # 使用统一训练器进行LDM增强流程
        print("=== 启用LDM增强训练模式 ===")
        unified_trainer = UnifiedTrainer(cfg, logf=logf)
        
        # 阶段1：训练encoder 或 直接加载预训练权重
        if cfg.use_pretrained_encoder:
            print("=== 跳过Encoder训练，加载已保存的权重 ===")
            unified_trainer.load_pretrained_encoder(cfg.encoder_ckpt_path if 'encoder_ckpt_path' in cfg else None)
        else:
            unified_trainer.train_encoder()
        
        # 阶段2：初始化LDM并收集embeddings
        unified_trainer.init_ldm_components()
        unified_trainer.collect_training_embeddings()
        
        # 降维分析 训练数据
        # unified_trainer.visualize_train_data(save_path=os.path.join(results_dir, f"{dataset}-train-embeddings.png"))

        # # 阶段3：训练LDM（或从ckpt加载跳过训练）
        # if cfg.use_pretrained_ldm:
        #     print("=== 跳过LDM训练，加载已保存的LDM权重 ===")
        #     unified_trainer.load_pretrained_ldm(cfg.ldm_ckpt_path if 'ldm_ckpt_path' in cfg else None)
        # else:
        #     unified_trainer.train_ldm()
        

        # 可选：内在质量评估（不依赖下游分类）
        if bool(getattr(cfg, 'evaluate_ldm_intrinsic', False)):
            print("=== 内在质量评估（MMD/FD/多样性/最近邻） ===")
            # 确保有LDM权重
            if unified_trainer.ldm is None:
                unified_trainer.init_ldm_components()
                if os.path.exists(os.path.join(save_dir, f'{cfg.dataset_name}_ldm.pkl')):
                    unified_trainer.load_pretrained_ldm(os.path.join(save_dir, f'{cfg.dataset_name}_ldm.pkl'))
            metrics = unified_trainer.evaluate_ldm_intrinsic(
                num_samples=int(getattr(cfg, 'evaluate_ldm_intrinsic_num_samples', 1000)),
                log_to_wandb=bool(getattr(cfg, 'use_wandb', False))
            )
            print({k: (v if not isinstance(v, dict) else {ik: iv for ik, iv in v.items()}) for k, v in metrics.items()})

        # ==================== 测试评估阶段 ====================
        print("=== 开始测试评估 ===")
        
        # 加载LDM模型（如果需要的话）
        ldm_model = None
        if cfg.num_augmented_samples > 0:
            print("=== 加载LDM模型 ===")
            ldm_model_path = os.path.join(save_dir, f'{cfg.dataset_name}_ldm.pkl')
            if os.path.exists(ldm_model_path):
                unified_trainer.ldm.load_state_dict(torch.load(ldm_model_path, map_location=cfg.device, weights_only=True))
                print(f"LDM模型加载成功: {os.path.basename(ldm_model_path)}")
                unified_trainer.ldm.eval()
                ldm_model = unified_trainer.ldm

        print("=== 测试 ===")
        test_acc, test_std = unified_trainer.test_model(
            num_augmented_samples=cfg.num_augmented_samples,
            ldm_model=ldm_model,
            test_name="测试"
        )
        
        # 记录测试结果
        logf.write(f"=== 测试结果 ===\n")
        logf.write(f"测试准确率: {test_acc:.4f} ± {test_std:.4f}\n")
        
        # ==================== 结果记录 ====================
        logf.write(f"=== 最终结果比较 ===\n")
        logf.write(f"最终测试准确率: {test_acc:.4f}\n")

    # 完成 wandb 运行
    if wandb.run is not None:
        wandb.finish()
        print("✅ wandb 运行完成")

if __name__ == '__main__':
    main()
