import os
import json
import random
import numpy as np
import torch
import wandb
from omegaconf import OmegaConf
from tqdm import tqdm

from trainer import PyGTrainer
import subprocess   


def build_cfg():
    """构建配置：优先级 CLI > 配置文件 > 代码默认值。
    - 支持传入 config=path.yaml
    - 默认使用与 main.py 相同的 'configs/Letter_high_ldm.yaml'
    """
   

    cli = OmegaConf.from_cli()
    config_path = cli.get('config', 'configs/Letter_high_ldm.yaml')
    base = OmegaConf.load(config_path)
    cfg = OmegaConf.merge( base, cli)
    if 'config' in cfg:
        cfg.pop('config')
    cfg.device = select_device(cfg.device)
    print("选择设备: ", cfg.device)
    return cfg




def select_device(device_pref: str = 'auto') -> str:
    """
    一个更健壮的 PyTorch 设备选择函数（使用 print 输出信息）。

    策略 (按比率和选择):
    1. 强制使用 CPU: 'cpu'
    2. 指定 GPU: 'cuda:x', 会验证设备索引 x 的有效性。
    3. 自动选择 GPU: 'auto', 'gpu', 'cuda'
       - 计算 (显存使用率 % + GPU核心利用率 %) 之和，选择总和最小的 GPU。
       - 如果 `nvidia-smi` 失败，会打印警告并回退到 'cuda:0'。
    4. 如果 CUDA 不可用，所有 GPU 相关的选项都会回退到 'cpu'。

    参数:
    - device_pref (str): 用户的设备偏好。

    返回:
    - str: 选择的设备字符串 (例如, 'cuda:0', 'cpu')。
    """
    s = str(device_pref).lower().strip()

    if s == 'cpu':
        print("INFO: Device preference set to 'cpu'.")
        return 'cpu'

    if not torch.cuda.is_available():
        if s != 'cpu':
            print(f"WARNING: CUDA is not available. Falling back to 'cpu' from preference '{device_pref}'.")
        return 'cpu'

    num_gpus = torch.cuda.device_count()

    if s.startswith('cuda:'):
        try:
            device_idx = int(s.split(':')[-1])
            if 0 <= device_idx < num_gpus:
                print(f"INFO: Successfully selected specified device: 'cuda:{device_idx}'.")
                return f'cuda:{device_idx}'
            else:
                print(
                    f"WARNING: Invalid device index {device_idx}. Machine has {num_gpus} GPUs (indices 0 to {num_gpus-1}). "
                    f"Falling back to 'cuda:0'."
                )
                return 'cuda:0'
        except (ValueError, IndexError):
            print(f"WARNING: Could not parse device index from '{s}'. Falling back to 'cuda:0'.")
            return 'cuda:0'

    if s in ('auto', 'gpu', 'cuda'):
        try:
            # ### <<< 修改点 1: 查询总显存 (memory.total) ###
            out = subprocess.check_output([
                'nvidia-smi',
                '--query-gpu=memory.used,memory.total,utilization.gpu',
                '--format=csv,noheader,nounits'
            ], stderr=subprocess.PIPE, timeout=1.5)

            lines = out.decode('utf-8').strip().splitlines()
            if not lines:
                 raise ValueError("`nvidia-smi` returned empty output.")
            
            stats = []
            for i, line in enumerate(lines):
                if i >= num_gpus: continue
                parts = [p.strip() for p in line.split(',')]
                if len(parts) == 3:
                    mem_used = int(parts[0])
                    mem_total = int(parts[1])
                    gpu_util = int(parts[2])
                    # 存储原始数据用于后续计算
                    stats.append((i, mem_used, mem_total, gpu_util))
            
            if not stats:
                raise ValueError("Failed to parse `nvidia-smi` output.")

            # ### <<< 修改点 2: 计算比率和并排序 ###
            scores = []
            for i, mem_used, mem_total, gpu_util in stats:
                if mem_total == 0:
                    # 避免除零错误，给这种情况一个极差的分数
                    mem_ratio_percent = float('inf')
                else:
                    mem_ratio_percent = (mem_used / mem_total) * 100
                
                total_score = mem_ratio_percent + gpu_util
                scores.append({
                    'index': i,
                    'score': total_score,
                    'mem_ratio': mem_ratio_percent,
                    'util': gpu_util
                })

            # 按总分排序
            scores.sort(key=lambda x: x['score'])
            
            best_device_info = scores[0]
            best_idx = best_device_info['index']
            
            print(
                f"INFO: Auto-selected 'cuda:{best_idx}' "
                f"(Score: {best_device_info['score']:.1f}, "
                f"Mem-Ratio: {best_device_info['mem_ratio']:.1f}%, "
                f"GPU-Util: {best_device_info['util']}%)"
            )
            return f'cuda:{best_idx}'

        except (subprocess.TimeoutExpired, FileNotFoundError, ValueError, Exception) as e:
            print(
                f"WARNING: Failed to run `nvidia-smi` for auto-selection due to: {type(e).__name__}: {e}. "
                "Falling back to 'cuda:0'. Make sure `nvidia-smi` is in your PATH and drivers are working."
            )
            return 'cuda:0'

    print(f"WARNING: Unrecognized device preference '{device_pref}'. Defaulting to 'cuda:0' as CUDA is available.")
    return 'cuda:0'

def main():
    cfg = build_cfg()

    # 初始化 wandb（与 main.py 相同逻辑）
    if cfg.use_wandb:
        wandb.init(
            project=cfg.get('wandb_project', 'fsl_ddpm'),
            entity=cfg.get('wandb_entity', None),
            name=cfg.get('wandb_run_name', None),
            config=OmegaConf.to_container(cfg, resolve=True),
            tags=cfg.get('wandb_tags', []),
            notes=cfg.get('wandb_notes', ''),
            mode='online' if cfg.wandb_online else 'disabled'
        )
        print(f"✅ wandb 初始化成功: {wandb.run.name}")
    else:
        print("ℹ️ wandb 未启用")

    # 随机性设置（逐行对齐 main.py）
    os.environ['PYTHONHASHSEED'] = str(cfg.seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    random.seed(cfg.seed); np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed); torch.cuda.manual_seed(cfg.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


    # 结果/保存目录（与 main.py 一致）
    results_dir = './our_results'
    save_dir = './savepoint'
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(save_dir, exist_ok=True)

    dataset = cfg.dataset_name
    text_file_name = os.path.join(results_dir, f"{dataset}-{cfg.K_shot}-shot.txt")

    with open(text_file_name, 'w') as logf:
        logf.write("=== Config (PyG, 对齐main.py) ===\n")
        logf.write(OmegaConf.to_yaml(cfg))
        logf.write("\n")
        print(cfg)

        print("=== 创建 PyGTrainer ===")
        trainer = PyGTrainer(cfg, logf=logf)

        if cfg.use_pretrained_encoder:
            print("=== 跳过Encoder训练，加载已保存的权重 ===")
            trainer.load_pretrained_encoder(cfg.get('encoder_ckpt_path'))
        else:
            print("=== 开始训练Encoder ===")
            trainer.train_encoder_contrastive()

        print("=== 测试原始Encoder性能 ===")
        test_accs = trainer.test_model(num_augmented_samples=0, test_name="原始Encoder测试")
        test_acc = float(np.mean(test_accs)) if test_accs else 0.0
        test_std = float(np.std(test_accs)) if test_accs else 0.0

        logf.write("=== 测试结果 ===\n")
        logf.write(f"测试准确率: {test_acc:.4f} ± {test_std:.4f}\n")
        print("=== 最终结果 ===")
        print(f"测试准确率: {test_acc:.4f} ± {test_std:.4f}")

        # # ------- 阶段2：训练/载入 LDM -------
        # if not cfg.use_pretrained_ldm:
        #     tqdm.write("=== 开始训练LDM（PyG） ===")
        #     trainer.train_ldm_pyg()  # 新增：见 pyg_trainer.py
        # else:
        #     tqdm.write("=== 加载预训练LDM权重 ===")
        #     trainer.load_ldm_pyg(cfg.ldm_ckpt_path)

        # # ------- 阶段3：带 LDM 的 few-shot 评测（先 refine 再增强，再做加权线性分类） -------
        # tqdm.write("=== 带 LDM 的评测（Refine→Augment→Weighted-CE） ===")
        # test_acc_ldm, test_std_ldm = trainer.test_model_with_ldm_pyg()
        # with open(text_file_name, 'a') as logf:
        #     logf.write("=== LDM 评测结果（Refine→Augment） ===\n")
        #     logf.write(f"LDM测试准确率: {test_acc_ldm:.4f} ± {test_std_ldm:.4f}\n")
        # tqdm.write(f"LDM测试准确率: {test_acc_ldm:.4f} ± {test_std_ldm:.4f}")

        # if cfg.get('collect_embeddings', False):
        #     print("=== 收集训练集embeddings ===")
        #     trainer.collect_training_embeddings()
        #     print("embeddings收集完成")


if __name__ == '__main__':
    main()
