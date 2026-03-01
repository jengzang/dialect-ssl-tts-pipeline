# 智能语音与方言大模型本地实验工程

## 项目概述

这是一个专注于方言语音处理、模型微调与量化评估的本地化研究项目。项目包含 10 个课程模块，从语音合成基础到方言虚拟人搭建的完整学习路径。

**核心目标：** 在完全本地化的环境中，构建端到端的方言语音处理、模型微调与量化评估流水线。

**项目特点：**
- 纯本地算法研究，无 Web 服务或容器部署
- 基于 Python + PyTorch 的模块化开发
- 严格的工作流协议和文档规范
- 从小样本分类到大模型微调的渐进式学习

---

## 技术栈

### 核心框架
- **Python**: 3.10
- **深度学习**: PyTorch
- **机器学习**: Scikit-learn
- **环境管理**: Conda/Mamba (Miniforge)

### 语音处理工具
- **强制对齐**: Montreal Forced Aligner (MFA)
- **语音分析**: Praat, Parselmouth
- **音频处理**: Librosa
- **语音合成**: GPT-SoVITS, VITS

### 模型与算法
- **传统机器学习**: SVM (元音分类)
- **序列模型**: LSTM (声调分类)
- **自监督学习**: wav2vec 2.0 (IPA 识别、口音识别)
- **生成模型**: GPT-SoVITS (方言语音合成)

---

## 项目结构

```
JNU/
├── 01_workflow/              # 工作流协议
│   ├── code_commit_protocol.md
│   └── readme_update_protocol.md
├── material/                 # 课程材料（10 个 lesson）
│   ├── lesson_1/            # 语音合成概述
│   ├── lesson_2/            # MFA 强制对齐
│   ├── lesson_3/            # Praat 特征提取
│   ├── lesson_4/            # SVM 元音分类
│   ├── lesson_5/            # MFA 模型训练
│   ├── lesson_6/            # LSTM 声调分类
│   ├── lesson_7/            # wav2vec IPA 识别
│   ├── lesson_8/            # 方言翻译
│   ├── lesson_9/            # 方言口音识别
│   └── lesson_10/           # 方言虚拟人
├── src/                      # 核心源码模块
│   ├── data_pipeline/       # 数据处理管线
│   ├── models/              # 模型定义
│   ├── training/            # 训练逻辑
│   └── evaluation/          # 评估工具
├── scripts/                  # 可执行脚本（CLI 入口）
├── notebooks/                # Jupyter 探索性分析
├── checkpoints/              # 模型权重（不提交）
├── results/                  # 实验结果（不提交）
├── logs/                     # 日志文件（不提交）
├── CLAUDE.md                 # 本文件
├── README.md                 # 项目文档（唯一文档）
├── skill.md                  # 技术规范
├── requirements.txt          # Python 依赖
├── environment.yml           # Conda 环境
└── .gitignore                # Git 忽略规则
```

---

## 课程模块概览

### Lesson 1: 大模型时代的语音合成
- 语音合成发展史
- GPT-SoVITS 框架介绍
- Kaldi 语音识别工具

### Lesson 2: 方言语音合成资源构建
- Montreal Forced Aligner (MFA) 配置与使用
- 自动标注流程
- 语料数据库构建

### Lesson 3: 语音特征参数提取
- Praat 脚本使用
- Parselmouth Python 接口
- VS Code 环境配置

### Lesson 4: SVM 元音分类器
- 机器学习分类任务
- 支持向量机 (SVM)
- 语音参数特征提取

### Lesson 5: MFA 模型训练实战
- 开源数据集使用
- 自定义声学模型训练

### Lesson 6: LSTM 声调分类
- 长短期记忆网络 (LSTM)
- 时序数据处理
- 基频序列特征

### Lesson 7: wav2vec IPA 识别
- 自监督学习 (Self-Supervised Learning)
- wav2vec 2.0 预训练模型
- 预训练-微调范式

### Lesson 8: 方言平行语料翻译
- 方言与普通话翻译
- LoRA 微调技术

### Lesson 9: 方言口音识别
- wav2vec 口音分类
- 低资源方言识别

### Lesson 10: 方言虚拟人搭建
- GPT-SoVITS 语音合成
- Sadtalker 虚拟人技术

---

## 开发规范

### 代码风格
- 使用 Python 类型注解
- 函数和类必须有 docstring
- 遵循 PEP 8 规范
- 使用 `logging` 模块，禁止 `print` 调试

### 命名约定
- 文件名: `snake_case.py`
- 类名: `PascalCase`
- 函数/变量: `snake_case`
- 常量: `UPPER_CASE`

### CLI 脚本规范
所有核心脚本必须：
- 使用 `argparse` 或 `click` 封装
- 支持命令行参数控制实验变量
- 输出日志到 `logs/` 目录
- 保存结果到 `results/` 目录

示例：
```bash
python scripts/lesson_04_svm_vowel.py \
    --data_dir material/lesson_4/data \
    --target_vowels a e i \
    --model_path checkpoints/svm_vowel.pkl \
    --output_dir results/lesson_4
```

---

## 工作流协议

### 代码提交规范（code_commit_protocol）

**触发条件：** 用户明确说"提交"

**执行流程：**
1. 暂存相关更改（排除数据文件）
2. 执行 `git commit` 并写详细的中文 commit message
3. 自动执行 `git push` 到远程仓库

**Commit Message 格式：**
```
[模块/范围] 简短总结
- 新增功能说明
- 修改内容说明
- 算法变更说明
- 结构调整说明
```

**必须排除的文件：**
- 数据文件: `*.db`, `*.sqlite`, `*.csv`, `*.xlsx`
- 模型权重: `checkpoints/`
- 实验结果: `results/`, `*.png`
- 虚拟环境: `.venv/`, `venv/`
- 临时文件: `*.tmp`, `*.cache`

**只提交：**
- 源码: `*.py`
- 配置: `requirements.txt`, `environment.yml`
- 文档: `README.md`, `*.md`

### README 更新规范（readme_update_protocol）

**核心原则：**
- `README.md` 是唯一的文档文件
- 必须用简体中文编写
- 必须维护更新日志
- 禁止创建额外的文档文件

**触发条件：**
- 新增功能或模块
- 算法更新
- 结构变更
- 依赖变更

**必需章节：**
- 项目简介
- 环境配置
- 运行方式
- 核心模块说明
- 更新日志

**更新日志格式：**
```markdown
## 更新日志

### 2026-03-02
- 新增功能说明
- 修改内容说明
- 算法变更说明
```

---

## 环境配置

### 创建 Conda 环境
```bash
# 使用 Miniforge/Mamba（推荐）
mamba create -n dialect-speech python=3.10
mamba activate dialect-speech

# 或使用 Conda
conda create -n dialect-speech python=3.10
conda activate dialect-speech
```

### 安装依赖
```bash
# 基础依赖
pip install -r requirements.txt

# 使用清华镜像（国内推荐）
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple --trusted-host pypi.tuna.tsinghua.edu.cn
```

### VS Code 配置
1. 打开项目文件夹
2. 选择 Python 解释器：右下角选择 `dialect-speech` 环境
3. 安装推荐插件：Python, Pylance, Jupyter

---

## 常用命令

### 数据处理
```bash
# MFA 强制对齐
mfa align input_dir dict.txt acoustic_model.zip output_dir

# 提取语音特征
python scripts/extract_features.py --input_dir data/audio --output_dir data/features
```

### 模型训练
```bash
# SVM 元音分类
python scripts/lesson_04_svm_vowel.py --train

# LSTM 声调分类
python scripts/lesson_06_lstm_tone.py --train --epochs 50

# wav2vec 微调
python scripts/lesson_07_wav2vec_ipa.py --finetune --model_name facebook/wav2vec2-base
```

### 模型评估
```bash
# 推理测试
python scripts/lesson_04_svm_vowel.py --inference --model_path checkpoints/svm_vowel.pkl

# 误差分析
python src/evaluation/error_analysis.py --predictions results/predictions.csv
```

---

## 实验流程示例

### Lesson 4: SVM 元音分类器

1. **准备数据**
   ```bash
   # 解压课程材料
   unzip material/lesson_4/20251125_001617_svm_e575b55b.zip -d data/lesson_4
   ```

2. **提取特征**
   ```bash
   python src/data_pipeline/feature_extractor.py \
       --textgrid_dir data/lesson_4/TextGrid \
       --audio_dir data/lesson_4/audio \
       --output data/lesson_4/features.csv
   ```

3. **训练模型**
   ```bash
   python src/training/train_svm.py \
       --data data/lesson_4/features.csv \
       --target_vowels a e i o u \
       --model_path checkpoints/svm_vowel.pkl
   ```

4. **评估与可视化**
   ```bash
   python src/evaluation/error_analysis.py \
       --model checkpoints/svm_vowel.pkl \
       --test_data data/lesson_4/test_features.csv \
       --output_dir results/lesson_4
   ```

---

## 核心模块说明

### src/data_pipeline/
数据处理管线，包括：
- `mfa_aligner.py`: MFA 强制对齐封装
- `feature_extractor.py`: 语音特征提取（时长、F0、MFCCs）
- `dataset_builder.py`: 数据集构建与划分

### src/models/
模型定义，包括：
- `svm_classifier.py`: SVM 分类器
- `lstm_tone.py`: LSTM 声调分类模型
- `wav2vec_ipa.py`: wav2vec 2.0 微调模型

### src/training/
训练逻辑，包括：
- `train_svm.py`: SVM 训练脚本
- `train_lstm.py`: LSTM 训练脚本
- `finetune_wav2vec.py`: wav2vec 微调脚本

### src/evaluation/
评估工具，包括：
- `metrics.py`: 评估指标（WER, SECS, MCD）
- `error_analysis.py`: 误差分析（混淆矩阵、t-SNE）
- `visualizer.py`: 可视化工具（Matplotlib, Seaborn）

---

## 注意事项

### 数据文件管理
- 所有数据文件（音频、TextGrid、CSV）不提交到 Git
- 使用 `.gitignore` 严格控制
- 大文件使用压缩包形式存储在 `material/` 目录

### 模型权重管理
- 训练好的模型保存在 `checkpoints/` 目录
- 不提交到 Git（文件过大）
- 重要模型可上传到云存储或模型仓库

### 实验结果管理
- 所有图表、报告保存在 `results/` 目录
- 按 lesson 或实验日期组织
- 不提交到 Git

### 日志管理
- 使用 Python `logging` 模块
- 日志文件保存在 `logs/` 目录
- 格式：`YYYY-MM-DD_HH-MM-SS_experiment_name.log`

---

## 性能优化建议

### GPU 加速
- 确保 PyTorch 安装了 CUDA 支持
- 使用 `torch.cuda.is_available()` 检查 GPU 可用性
- 训练时使用 `device = torch.device("cuda" if torch.cuda.is_available() else "cpu")`

### 数据加载优化
- 使用 `torch.utils.data.DataLoader` 的 `num_workers` 参数
- 预处理后的特征保存为 `.h5` 或 `.npy` 格式
- 使用数据缓存减少重复计算

### 内存管理
- 大数据集使用批处理
- 及时释放不需要的变量
- 使用 `del` 和 `gc.collect()` 清理内存

---

## 故障排查

### 常见问题

**问题 1: ModuleNotFoundError**
```bash
# 解决方案：安装缺失的包
pip install praat-parselmouth
pip install numpy pandas scikit-learn joblib matplotlib tqdm
```

**问题 2: MFA 对齐失败**
```bash
# 检查音频采样率（MFA 要求 16kHz）
# 检查 TextGrid 层命名（words, phones）
# 检查发音词典格式
```

**问题 3: CUDA out of memory**
```bash
# 减小 batch_size
# 使用梯度累积
# 使用混合精度训练（torch.cuda.amp）
```

**问题 4: VS Code 环境选择错误**
- 右下角点击 Python 版本
- 选择正确的 Conda 环境
- 重启 VS Code

---

## 参考资源

### 官方文档
- [Praat](https://www.fon.hum.uva.nl/praat/)
- [Montreal Forced Aligner](https://montreal-forced-aligner.readthedocs.io/)
- [Parselmouth](https://parselmouth.readthedocs.io/)
- [GPT-SoVITS](https://github.com/RVC-Boss/GPT-SoVITS)
- [Hugging Face Transformers](https://huggingface.co/docs/transformers/)

### 学术论文
- wav2vec 2.0: [Baevski et al., 2020]
- VITS: [Kim et al., 2021]
- GPT-SoVITS: [GitHub Repository]

---

## 联系与支持

如遇到问题或需要帮助：
1. 查看课程材料中的 `readme_*.txt` 文件
2. 参考 `skill.md` 技术规范
3. 查看本文档的故障排查章节

---

**最后更新：** 2026-03-02
