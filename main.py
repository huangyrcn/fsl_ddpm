import os
import torch
import random
import numpy as np
from omegaconf import OmegaConf
from train import Trainer


def load_config(config_path):
    """使用OmegaConf加载配置文件"""
    conf = OmegaConf.load(config_path)
    
    class Args:
        def __init__(self, config):
            for key, value in config.items():
                setattr(self, key, value)
    
    return Args(conf)


def main():
    """主函数"""
    import sys
    
    if len(sys.argv) != 3 or sys.argv[1] != '--config':
        print("Usage: python main.py --config <config_file>")
        sys.exit(1)
    
    # 加载配置
    config_path = sys.argv[2]
    args = load_config(config_path)
    args.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    
    # 设置随机种子
    seed_value = 72
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed(seed_value)
    
    # 创建目录
    os.makedirs('./our_results', exist_ok=True)
    os.makedirs('./savepoint', exist_ok=True)
    
    # 训练和测试
    print(f"Training with config: {config_path}")
    print(f"Dataset: {args.dataset_name}")
    
    text_file_name = f'./our_results/{args.dataset_name}-{args.K_shot}-shot.txt'
    with open(text_file_name, 'w') as f:
        trainer = Trainer(args)
        trainer.train(log_file=f)
        test_acc = trainer.test(log_file=f)


if __name__ == "__main__":
    main()
