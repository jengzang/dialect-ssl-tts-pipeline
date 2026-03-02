# Project 1: 增强方言翻译评估系统

## 概述

将 Lesson 8 从基础的 LoRA 微调项目升级为具有完善评估指标的生产级方言翻译系统。

## 新增功能

### 1. 机器翻译评估指标
- **BLEU** (Bilingual Evaluation Understudy): 标准机器翻译指标
- **ROUGE** (Recall-Oriented Understudy): 召回率导向的评估
- **ChrF** (Character n-gram F-score): 字符级 F 分数
- **METEOR** (可选): 显式排序的翻译评估

### 2. 数据增强
- 同义词替换
- 数据扩展（从 24 样本到 500+ 样本）
- 自动划分训练/验证/测试集（70/15/15）

### 3. WandB 实验跟踪
- 训练损失曲线
- 验证损失曲线
- 学习率调度
- 超参数记录

### 4. 完整的评估流程
- 批量翻译
- 自动指标计算
- 结果保存（JSON 格式）

## 安装依赖

```bash
pip install peft accelerate bitsandbytes evaluate sacrebleu rouge-score wandb
```

或使用清华镜像：

```bash
pip install peft accelerate bitsandbytes evaluate sacrebleu rouge-score wandb \
    -i https://pypi.tuna.tsinghua.edu.cn/simple --trusted-host pypi.tuna.tsinghua.edu.cn
```

## 使用流程

### 步骤 1: 数据增强

将原始的 24 个样本扩展到 500 个样本：

```bash
python scripts/augment_dialect_data.py \
    --input material/lesson_8/dialect2mandarin.csv \
    --output_dir data/dialect_translation \
    --target_size 500 \
    --train_ratio 0.7 \
    --val_ratio 0.15 \
    --test_ratio 0.15
```

输出：
- `data/dialect_translation/train.json` (350 样本)
- `data/dialect_translation/val.json` (75 样本)
- `data/dialect_translation/test.json` (75 样本)

### 步骤 2: 训练模型（带 WandB）

```bash
python scripts/lesson_08_dialect_translation.py \
    --mode train \
    --model_name Qwen/Qwen-7B-Chat \
    --train_data data/dialect_translation/train.json \
    --val_data data/dialect_translation/val.json \
    --output_dir checkpoints/dialect_translation_v2 \
    --epochs 3 \
    --batch_size 4 \
    --learning_rate 2e-4 \
    --lora_r 8 \
    --lora_alpha 32 \
    --use_wandb \
    --wandb_project dialect-translation
```

### 步骤 3: 评估模型

```bash
python scripts/lesson_08_dialect_translation.py \
    --mode evaluate \
    --model_path checkpoints/dialect_translation_v2/best \
    --test_data data/dialect_translation/test.json \
    --output_dir results/lesson_08 \
    --batch_size 8
```

输出：
- `results/lesson_08/evaluation/predictions.json` - 预测结果
- `results/lesson_08/evaluation/metrics.json` - 评估指标

### 步骤 4: 推理测试

```bash
python scripts/lesson_08_dialect_translation.py \
    --mode inference \
    --model_path checkpoints/dialect_translation_v2/best \
    --dialect_text "侬好，今朝天氣老好个"
```

## 评估指标说明

### BLEU (0-100)
- **> 40**: 优秀
- **30-40**: 良好
- **20-30**: 可接受
- **< 20**: 需要改进

目标：方言→普通话 BLEU > 30

### ROUGE-L F1 (0-100)
- 衡量召回率和精确率的平衡
- 关注最长公共子序列

目标：ROUGE-L > 50

### ChrF (0-100)
- 字符级评估，对中文更友好
- 不依赖分词

目标：ChrF > 40

## 项目结构

```
JNU/
├── src/
│   ├── evaluation/
│   │   └── mt_metrics.py          # 新增：MT 评估指标
│   ├── data_pipeline/
│   │   └── dialect_augmentation.py # 新增：数据增强
│   └── training/
│       └── dialect_translation_trainer.py  # 修改：添加 WandB
├── scripts/
│   ├── lesson_08_dialect_translation.py    # 修改：添加评估模式
│   └── augment_dialect_data.py             # 新增：数据增强脚本
├── data/
│   └── dialect_translation/
│       ├── train.json
│       ├── val.json
│       └── test.json
├── checkpoints/
│   └── dialect_translation_v2/
│       ├── best/
│       └── epoch_*/
└── results/
    └── lesson_08/
        └── evaluation/
            ├── predictions.json
            └── metrics.json
```

## 成功指标

- [x] 数据集扩展到 500+ 样本
- [ ] BLEU-4 > 30
- [ ] ROUGE-L F1 > 50
- [ ] ChrF > 40
- [ ] WandB 仪表板显示训练曲线
- [ ] 完整的评估报告

## 下一步

完成 Project 1 后，继续：
- **Project 2**: 超参数优化与 LoRA 架构搜索
- **Project 3**: 多任务学习（翻译 + 口音分类）
- **Project 4**: 指令微调

## 故障排查

### 问题 1: WandB 登录失败
```bash
wandb login
# 输入你的 API key
```

### 问题 2: CUDA out of memory
- 减小 `--batch_size`（尝试 2 或 1）
- 使用 `--quantization` 启用 4-bit 量化

### 问题 3: 数据增强效果不佳
- 检查同义词字典（`src/data_pipeline/dialect_augmentation.py`）
- 增加 `--target_size`
- 调整增强策略

## 参考资源

- [SacreBLEU 文档](https://github.com/mjpost/sacrebleu)
- [ROUGE Score 文档](https://github.com/google-research/google-research/tree/master/rouge)
- [WandB 文档](https://docs.wandb.ai/)
- [LoRA 论文](https://arxiv.org/abs/2106.09685)
