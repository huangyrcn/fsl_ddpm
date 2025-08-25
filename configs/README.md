# 配置文件使用说明

## 概述
本项目已从 `argparse` 迁移到 YAML 配置文件系统，使用更加灵活和可维护。

## 使用方法

### 基本用法
```bash
python train.py --config <配置文件路径>
```

### 示例
```bash
# 使用默认配置
python train.py --config configs/default.yaml

# 使用ENZYMES数据集配置
python train.py --config configs/enzymes.yaml

# 使用Letter_high数据集配置
python train.py --config configs/letter_high.yaml

# 使用Reddit数据集配置
python train.py --config configs/reddit.yaml
```

## 配置文件结构

### 默认配置文件 (configs/default.yaml)
包含所有参数的默认值，作为其他配置文件的基础。

### 数据集专用配置文件
- `configs/triangles.yaml`: TRIANGLES数据集配置
- `configs/enzymes.yaml`: ENZYMES数据集优化配置
- `configs/letter_high.yaml`: Letter_high数据集优化配置
- `configs/reddit.yaml`: Reddit数据集优化配置

## 配置参数说明

### 数据集配置
- `dataset_name`: 数据集名称 (TRIANGLES, ENZYMES, Letter_high, Reddit)
- 每个数据集都有专门的配置文件，针对其特性进行了优化

### 少样本学习参数
- `N_way`: 分类类别数
- `K_shot`: 每类支持样本数
- `query_size`: 查询集大小
- `patience`: 早停耐心值

### 模型参数
- `dropout`: Dropout率
- `gin_layer`: GIN网络层数
- `gin_hid`: GIN隐藏层维度

### 数据增强参数
- `aug1`, `aug2`: 数据增强策略
- `gen_train_num`: 训练图生成数量
- `gen_test_num`: 测试图生成数量

### 优化器参数
- `lr`: 学习率
- `weight_decay`: 权重衰减
- `epoch_num`: 训练轮数

## 数据集特性对比

| 数据集 | 特点 | 推荐配置 | 特殊考虑 |
|--------|------|----------|----------|
| TRIANGLES | 图结构简单，节点特征基于度 | `configs/triangles.yaml` | 使用默认GIN配置即可 |
| ENZYMES | 生物分子图，结构复杂 | `configs/enzymes.yaml` | 需要更深的网络(4层)和更大隐藏维度(256) |
| Letter_high | 字母识别图，中等复杂度 | `configs/letter_high.yaml` | 调整数据增强策略 |
| Reddit | 社交网络图，规模大 | `configs/reddit.yaml` | 需要更深的网络、更多训练轮数、更小学习率 |

## 自定义配置

1. 复制现有配置文件作为模板
2. 修改需要的参数值
3. 使用 `--config` 参数指定你的配置文件

## 注意事项

- 配置文件使用YAML格式，注意缩进
- 未在配置文件中指定的参数将使用默认值
- 确保配置文件路径正确，否则程序会报错并退出


