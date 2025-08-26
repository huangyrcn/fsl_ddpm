import os
import json

import torch
import random
import numpy as np

from omegaconf import OmegaConf

from train import Trainer

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

    with open(text_file_name, 'w') as logf:
        trainer = Trainer(cfg, logf=logf)
        trainer.train()
        test_acc = trainer.test()

    with open(params_txt, 'a') as f:
        json.dump({str(test_acc): str(cfg)}, f, indent=4)

if __name__ == '__main__':
    main()
