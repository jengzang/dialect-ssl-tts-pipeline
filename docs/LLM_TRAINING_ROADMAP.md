# LLM 训练学习路径 - 实施指南

## 概述

这是一个为人工智能学院申请准备的 LLM 训练项目组合，包含 7 个渐进式项目，从基础评估到 RLHF。

## 项目路线图

### ✅ Project 1: 增强方言翻译评估系统（已完成）
**状态:** 实现完成，待测试
**复杂度:** 低
**时长:** 1-2 周
**文档:** [docs/project_01_enhanced_evaluation.md](project_01_enhanced_evaluation.md)

**核心功能:**
- ✅ 机器翻译评估指标（BLEU、ROUGE、ChrF、METEOR）
- ✅ 数据增强（24 → 500+ 样本）
- ✅ WandB 实验跟踪
- ✅ 完整的评估流程

**关键文件:**
- `src/evaluation/mt_metrics.py`
- `src/data_pipeline/dialect_augmentation.py`
- `scripts/augment_dialect_data.py`
- `scripts/lesson_08_dialect_translation.py` (evaluate 模式)

**下一步:**
1. 安装依赖: `pip install peft accelerate bitsandbytes evaluate sacrebleu rouge-score wandb`
2. 运行测试: `python scripts/test_project_01.py`
3. 数据增强: `python scripts/augment_dialect_data.py --input material/lesson_8/dialect2mandarin.csv --output_dir data/dialect_translation --target_size 500`
4. 训练模型: 使用 `--use_wandb` 标志
5. 评估模型: 使用 `--mode evaluate`

---

### 🔜 Project 2: 超参数优化与 LoRA 架构搜索
**状态:** 待实现
**复杂度:** 中
**时长:** 2-3 周
**优先级:** 高

**目标:**
- 系统性探索 LoRA 超参数（rank, alpha, 学习率）
- 实现 Optuna 贝叶斯优化
- 分析 LoRA 权重分布
- 比较 QLoRA vs LoRA vs 全量微调

**关键文件（待创建）:**
- `src/training/hyperparameter_search.py`
- `src/evaluation/lora_analysis.py`
- `scripts/lesson_08_hp_search.py`
- `notebooks/lora_architecture_analysis.ipynb`

**成功指标:**
- BLEU > 35
- 50+ 次试验的 Optuna 研究
- 帕累托前沿图（性能 vs 参数）

---

### 🔜 Project 3: 多任务学习
**状态:** 待实现
**复杂度:** 中高
**时长:** 2-3 周
**优先级:** 高

**目标:**
- 结合 Lesson 8（翻译）+ Lesson 9（口音识别）
- 实现多任务 LoRA
- 任务平衡策略

**关键文件（待创建）:**
- `src/models/multitask_dialect_model.py`
- `src/training/multitask_trainer.py`
- `scripts/lesson_08_09_multitask.py`

**成功指标:**
- 翻译 BLEU > 33
- 口音分类准确率 > 85%
- 相比独立模型减少 30% 参数

---

### 🔜 Project 4: 指令微调与提示工程
**状态:** 待实现
**复杂度:** 中高
**时长:** 2-3 周
**优先级:** 高

**目标:**
- 创建指令数据集（翻译、识别、解释）
- 实现指令模板（Alpaca、Vicuna）
- 少样本提示能力

**关键文件（待创建）:**
- `src/data_pipeline/instruction_builder.py`
- `src/models/instruction_tuned_model.py`
- `scripts/lesson_08_instruction_tuning.py`

**成功指标:**
- 指令遵循准确率 > 90%
- 零样本 BLEU > 25
- 少样本（3-shot）BLEU > 32

---

### 🔜 Project 5: 模型比较与缩放定律
**状态:** 待实现
**复杂度:** 高
**时长:** 3-4 周
**优先级:** 中

**目标:**
- 微调多个模型（Qwen-1.8B, 7B, 14B）
- 分析缩放定律
- 模型蒸馏

**关键文件（待创建）:**
- `src/models/model_factory.py`
- `src/evaluation/scaling_analysis.py`
- `notebooks/scaling_laws_analysis.ipynb`

**成功指标:**
- 5+ 个模型训练和评估
- 缩放定律图
- 蒸馏模型：50% 大小，90% 性能

---

### 🔜 Project 6: 高级微调技术
**状态:** 待实现
**复杂度:** 高
**时长:** 3-4 周
**优先级:** 中

**目标:**
- 实现 Prefix Tuning、Adapter、IA³
- 比较效率（参数、内存、速度）
- 混合精度训练

**关键文件（待创建）:**
- `src/models/prefix_tuning_model.py`
- `src/models/adapter_model.py`
- `notebooks/finetuning_comparison.ipynb`

**成功指标:**
- 5+ 种微调方法实现
- 相比全量微调减少 40% 内存
- 完整的技术比较报告

---

### 🔜 Project 7: RLHF 与人类反馈集成（顶点项目）
**状态:** 待实现
**复杂度:** 非常高
**时长:** 4-5 周
**优先级:** 中高

**目标:**
- 收集人类偏好数据
- 训练奖励模型
- 实现 PPO
- RLHF 微调

**关键文件（待创建）:**
- `src/data_pipeline/preference_dataset.py`
- `src/models/reward_model.py`
- `src/training/rlhf_trainer.py`
- `scripts/rlhf_training.py`

**成功指标:**
- 奖励模型准确率 > 70%
- 人类评估: RLHF 模型在 60%+ 案例中被偏好
- BLEU 保持 > 30

---

## 实施时间线

### 阶段 1: 基础（第 1-4 周）
- ✅ Week 1-2: Project 1 - 增强评估
- 🔜 Week 3-4: Project 2 - 超参数优化

### 阶段 2: 高级技术（第 5-10 周）
- 🔜 Week 5-6: Project 3 - 多任务学习
- 🔜 Week 7-8: Project 4 - 指令微调
- 🔜 Week 9-10: Project 6 - 高级微调方法

### 阶段 3: 研究与缩放（第 11-16 周）
- 🔜 Week 11-14: Project 5 - 模型比较与缩放
- 🔜 Week 15-19: Project 7 - RLHF（顶点项目）

---

## 人工智能学院申请交付物

### 1. 代码仓库（GitHub）
- [ ] 7 个完整项目
- [ ] 可复现结果
- [ ] 综合 README
- [ ] Jupyter 笔记本

### 2. 技术报告（3-5 篇）
- [ ] "低资源方言翻译的 LoRA 系统评估"
- [ ] "方言处理的多任务学习"
- [ ] "方言翻译模型的缩放定律"
- [ ] "文化感知方言翻译的 RLHF"

### 3. 训练好的模型（HuggingFace Hub）
- [ ] 5+ 个微调模型
- [ ] 模型卡和基准测试
- [ ] 推理演示

### 4. 博客文章
- [ ] "从 24 到 1000：构建方言翻译数据集"
- [ ] "LoRA 超参数调优：实用指南"
- [ ] "从零实现 RLHF"

### 5. 演示材料
- [ ] 总结幻灯片
- [ ] 演示视频
- [ ] 作品集网站

---

## 当前状态

**已完成:**
- ✅ Project 1 实现（代码完成，待测试）
- ✅ 依赖更新（requirements.txt）
- ✅ 文档创建（project_01_enhanced_evaluation.md）
- ✅ 测试脚本（test_project_01.py）

**下一步行动:**
1. 测试 Project 1 功能
2. 运行数据增强
3. 训练基线模型
4. 评估并记录结果
5. 开始 Project 2 设计

---

## 快速开始

### 安装依赖
```bash
pip install peft accelerate bitsandbytes evaluate sacrebleu rouge-score wandb
```

### 测试 Project 1
```bash
python scripts/test_project_01.py
```

### 数据增强
```bash
python scripts/augment_dialect_data.py \
    --input material/lesson_8/dialect2mandarin.csv \
    --output_dir data/dialect_translation \
    --target_size 500
```

### 训练模型
```bash
python scripts/lesson_08_dialect_translation.py \
    --mode train \
    --train_data data/dialect_translation/train.json \
    --val_data data/dialect_translation/val.json \
    --use_wandb
```

### 评估模型
```bash
python scripts/lesson_08_dialect_translation.py \
    --mode evaluate \
    --model_path checkpoints/dialect_translation_v2/best \
    --test_data data/dialect_translation/test.json
```

---

## 成功指标总结

| 项目 | 关键指标 | 目标 | 状态 |
|------|---------|------|------|
| 1. 增强评估 | BLEU-4 | > 30 | 🔜 待测试 |
| 2. 超参数优化 | BLEU-4 | > 35 | ⏳ 待实现 |
| 3. 多任务 | BLEU + 准确率 | > 33 + > 85% | ⏳ 待实现 |
| 4. 指令微调 | 指令遵循 | > 90% | ⏳ 待实现 |
| 5. 模型比较 | 模型数 | 5+ | ⏳ 待实现 |
| 6. 高级微调 | 方法数 | 5+ | ⏳ 待实现 |
| 7. RLHF | 人类偏好 | > 60% | ⏳ 待实现 |

---

**最后更新:** 2026-03-02
**项目状态:** Project 1 实现完成，待测试和验证
