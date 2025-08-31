import os
import json

import torch
import random
import numpy as np

from omegaconf import OmegaConf

from train import Trainer
from unified_trainer import UnifiedTrainer

# 添加 wandb 导入
import wandb

def build_cfg():
    # 仅用 OmegaConf，从命令行读取：必须提供 config=path/to.yaml，其余键值用于覆盖
    cli = OmegaConf.from_cli()
    if 'config' not in cli:
        raise ValueError("Missing required CLI arg: config=path/to.yaml")
    base = OmegaConf.load(cli.config)
    cli.pop('config')
    cfg = OmegaConf.merge(base, cli)  # CLI 覆盖 YAML
    return cfg
    
def pick_best_device() -> str:
    # 返回 'cuda:x' 或 'cpu'
    if not torch.cuda.is_available():
        return 'cpu'
    try:
        best_i, best_free = 0, -1
        for i in range(torch.cuda.device_count()):
            free, _ = torch.cuda.mem_get_info(i)
            if free > best_free:
                best_i, best_free = i, free
        return f'cuda:{best_i}'
    except Exception:
        return 'cuda:0'

def main():
    cfg = build_cfg()

    # 初始化 wandb（如果配置中启用）
    if getattr(cfg, 'use_wandb', False):
        wandb.init(
            project=getattr(cfg, 'wandb_project', 'fsl_ddpm'),
            entity=getattr(cfg, 'wandb_entity', None),
            name=getattr(cfg, 'wandb_run_name', None),
            config=OmegaConf.to_container(cfg, resolve=True),
            tags=getattr(cfg, 'wandb_tags', []),
            notes=getattr(cfg, 'wandb_notes', ''),
            mode='online' if getattr(cfg, 'wandb_online', True) else 'disabled'
        )
        print(f"✅ wandb 初始化成功: {wandb.run.name}")
    else:
        print("ℹ️ wandb 未启用")

    # 设置随机性（原逻辑保持）
    os.environ['PYTHONHASHSEED'] = str(72)
    random.seed(72); np.random.seed(72)
    torch.manual_seed(72); torch.cuda.manual_seed(72)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    # 若配置里写的是 'cuda'，自动挑选空闲显存最大的 GPU
    if str(getattr(cfg, 'device', '')).lower() == 'cuda':
        cfg.device = pick_best_device()

    results_dir = './our_results'
    save_dir = './savepoint'
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(save_dir, exist_ok=True)

    dataset = cfg.dataset_name
    k = cfg.K_shot
    text_file_name = os.path.join(results_dir, f"{dataset}-{k}-shot.txt")
    params_txt = os.path.join(results_dir, f"{dataset}-{k}-shot-params.txt")
    params_yaml = os.path.join(results_dir, f"{dataset}-{k}-shot-params.yaml")

    # 最终配置
    OmegaConf.save(config=cfg, f=params_yaml)

    # 检查是否启用LDM增强模式
    use_ldm_augmentation = getattr(cfg, 'use_ldm_augmentation', False)
    
    with open(text_file_name, 'w') as logf:
        if use_ldm_augmentation:
            # 使用统一训练器进行LDM增强流程
            print("=== 启用LDM增强训练模式 ===")
            unified_trainer = UnifiedTrainer(cfg, logf=logf)
            
            # 阶段1：训练encoder 或 直接加载预训练权重
            if getattr(cfg, 'use_pretrained_encoder', False):
                print("=== 跳过Encoder训练，加载已保存的权重 ===")
                unified_trainer.load_pretrained_encoder(getattr(cfg, 'encoder_ckpt_path', None))
            else:
                unified_trainer.train_encoder()
            
            # 阶段2：初始化LDM并收集embeddings
            unified_trainer.init_ldm_components()
            unified_trainer.collect_training_embeddings()
            
            # 阶段3：训练LDM（或从ckpt加载跳过训练）
            if getattr(cfg, 'use_pretrained_ldm', False):
                print("=== 跳过LDM训练，加载已保存的LDM权重 ===")
                unified_trainer.load_pretrained_ldm(getattr(cfg, 'ldm_ckpt_path', None))
            else:
                unified_trainer.train_ldm()
            
            # 阶段3.5：LDM训练完成后的数据可视化
            print("=== 开始LDM训练完成后的数据可视化 ===")
            # 调用可视化函数，生成支持集、测试数据和生成数据的可视化
            save_path = os.path.join(results_dir, f"{dataset}-{k}-shot-ldm-visualization.png")
            support_embs, test_embs, generated_embs = unified_trainer.visualize_support_query_generation(
                num_samples_per_prototype=getattr(cfg, 'num_augmented_samples', 30),
                save_path=save_path
            )
            print(f"✅ 数据可视化完成，图片保存到: {save_path}")
            
            # 阶段4：测试
            original_acc, original_std = unified_trainer.test_model(use_ldm_augmentation=False, test_name="原始Encoder测试（无LDM增强）")
            ldm_acc, ldm_std = unified_trainer.test_model(use_ldm_augmentation=True, test_name="最终测试（使用LDM增强）")
            
            logf.write(f"=== 最终结果比较 ===\n")
            logf.write(f"原始Encoder: {original_acc:.4f} ± {original_std:.4f}\n")
            logf.write(f"LDM增强: {ldm_acc:.4f} ± {ldm_std:.4f}\n")
            logf.write(f"提升: {ldm_acc - original_acc:.4f}\n")
            
            test_acc = ldm_acc  # 返回LDM增强的结果
        else:
            # 使用原始训练器
            print("=== 使用原始训练模式 ===")
            trainer = Trainer(cfg, logf=logf)
            trainer.train()
            test_acc = trainer.test()

    with open(params_txt, 'a') as f:
        json.dump({str(test_acc): str(cfg)}, f, indent=4)

    # 完成 wandb 运行
    if wandb.run is not None:
        wandb.finish()
        print("✅ wandb 运行完成")

if __name__ == '__main__':
    main()
