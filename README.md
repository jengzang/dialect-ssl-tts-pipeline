# 智能语音与方言大模型本地实验工程

## 项目简介

本项目是一个专注于方言语音处理、模型微调与量化评估的本地化研究工程。通过 10 个渐进式课程模块，从传统机器学习到深度学习，从小样本分类到大模型微调，构建完整的方言语音处理技术栈。

**核心特点：**
- 🎯 完全本地化：无需云服务，所有实验在本地完成
- 📚 系统化学习：10 个课程模块，循序渐进
- 🔬 实战导向：每个模块都有完整的代码和数据
- 🛠️ 工具链完整：MFA、Praat、PyTorch、wav2vec、GPT-SoVITS

**技术栈：**
- Python 3.10 + PyTorch
- Montreal Forced Aligner (MFA)
- Praat / Parselmouth
- wav2vec 2.0
- GPT-SoVITS

---

## 课程模块

### 📖 理论基础
- **Lesson 1**: 大模型时代的语音合成与应用探讨
- **Lesson 2**: 方言语音合成的资源构建流程（MFA、自动标注）

### 🔧 特征工程
- **Lesson 3**: 语音特征参数提取与分析（Praat、Parselmouth）

### 🤖 传统机器学习
- **Lesson 4**: 基于 SVM 的元音分类器
- **Lesson 5**: MFA 训练模型实战

### 🧠 深度学习
- **Lesson 6**: 基于 LSTM 的声调分类模型
- **Lesson 7**: IPA 识别模型及训练（wav2vec 2.0）

### 🌐 高级应用
- **Lesson 8**: 方言平行语料翻译
- **Lesson 9**: 方言口音识别
- **Lesson 10**: 方言虚拟人搭建（GPT-SoVITS）

---

## 环境配置

### 1. 安装 Miniforge（推荐）

下载地址：https://github.com/conda-forge/miniforge/releases

```bash
# Windows
Miniforge3-25.9.1-0-Windows-x86_64.exe

# macOS
Miniforge3-25.9.1-0-MacOSX-x86_64.sh
```

### 2. 创建虚拟环境

```bash
# 使用 mamba（更快）
mamba create -n dialect-speech python=3.10
mamba activate dialect-speech

# 或使用 conda
conda create -n dialect-speech python=3.10
conda activate dialect-speech
```

### 3. 安装依赖

```bash
# 基础依赖
pip install -r requirements.txt

# 使用清华镜像（国内推荐）
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple --trusted-host pypi.tuna.tsinghua.edu.cn
```

### 4. 安装 MFA

```bash
# 使用 conda 安装
conda install -c conda-forge montreal-forced-aligner
```

### 5. VS Code 配置

1. 安装 VS Code：https://code.visualstudio.com/
2. 安装 Python 扩展
3. 选择 Python 解释器：右下角选择 `dialect-speech` 环境

---

## 运行方式

### Lesson 4: SVM 元音分类器

```bash
# 1. 解压课程材料
cd material/lesson_4
unzip 20251125_001617_svm_e575b55b.zip

# 2. 提取特征
python src/data_pipeline/feature_extractor.py \
    --textgrid_dir material/lesson_4/TextGrid \
    --audio_dir material/lesson_4/audio \
    --output data/lesson_4_features.csv

# 3. 训练模型
python src/training/train_svm.py \
    --data data/lesson_4_features.csv \
    --target_vowels a e i \
    --model_path checkpoints/svm_vowel.pkl

# 4. 推理测试
python src/training/train_svm.py \
    --inference \
    --model_path checkpoints/svm_vowel.pkl \
    --test_data data/lesson_4_test.csv
```

### Lesson 6: LSTM 声调分类

```bash
# 1. 准备数据
python src/data_pipeline/extract_pitch.py \
    --audio_dir material/lesson_6/audio \
    --output data/lesson_6_pitch.csv

# 2. 训练模型
python src/training/train_lstm.py \
    --data data/lesson_6_pitch.csv \
    --epochs 50 \
    --batch_size 32 \
    --model_path checkpoints/lstm_tone.pth

# 3. 评估模型
python src/evaluation/evaluate_lstm.py \
    --model checkpoints/lstm_tone.pth \
    --test_data data/lesson_6_test.csv
```

### Lesson 7: wav2vec IPA 识别

```bash
# 1. 准备数据
python src/data_pipeline/prepare_wav2vec_data.py \
    --audio_dir material/lesson_7/audio \
    --transcript_file material/lesson_7/transcripts.txt \
    --output_dir data/lesson_7_wav2vec

# 2. 微调模型
python src/training/finetune_wav2vec.py \
    --model_name facebook/wav2vec2-base \
    --data_dir data/lesson_7_wav2vec \
    --output_dir checkpoints/wav2vec_ipa \
    --epochs 10 \
    --batch_size 8

# 3. 推理测试
python src/training/finetune_wav2vec.py \
    --inference \
    --model_path checkpoints/wav2vec_ipa \
    --audio_file test.wav
```

---

## 核心分析方法概述

### 1. 特征提取

**时域特征：**
- 时长（Duration）
- 能量（Energy）
- 过零率（Zero Crossing Rate）

**频域特征：**
- 基频（F0 / Pitch）
- 共振峰（Formants: F1, F2, F3）
- 梅尔频率倒谱系数（MFCCs）

**工具：**
- Praat 脚本
- Parselmouth Python 接口
- Librosa

### 2. 传统机器学习

**SVM 元音分类：**
- 特征：F1, F2, F3, 时长
- 算法：支持向量机（RBF 核）
- 评估：准确率、混淆矩阵

**LSTM 声调分类：**
- 特征：基频序列（F0 contour）
- 算法：长短期记忆网络
- 评估：准确率、F1 分数

### 3. 深度学习与自监督学习

**wav2vec 2.0 微调：**
- 预训练模型：facebook/wav2vec2-base
- 微调任务：IPA 音素识别、口音分类
- 评估：字错误率（WER）、音素错误率（PER）

**GPT-SoVITS 语音合成：**
- Zero-shot TTS：5 秒参考音频
- Few-shot TTS：1 分钟微调数据
- 评估：音色相似度（SECS）、梅尔倒谱失真（MCD）

### 4. 误差分析

**可视化方法：**
- 混淆矩阵（Confusion Matrix）
- t-SNE 降维可视化
- 决策边界分析

**量化指标：**
- 准确率（Accuracy）
- 精确率（Precision）
- 召回率（Recall）
- F1 分数

---

## 项目结构

```
JNU/
├── 01_workflow/              # 工作流协议
│   ├── code_commit_protocol.md
│   └── readme_update_protocol.md
├── material/                 # 课程材料
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
├── scripts/                  # 可执行脚本
├── notebooks/                # Jupyter 探索性分析
├── checkpoints/              # 模型权重（不提交）
├── results/                  # 实验结果（不提交）
├── logs/                     # 日志文件（不提交）
├── CLAUDE.md                 # Claude Code 项目指南
├── README.md                 # 本文件
├── skill.md                  # 技术规范
├── requirements.txt          # Python 依赖
├── environment.yml           # Conda 环境
└── .gitignore                # Git 忽略规则
```

---

## 性能边界说明

### 硬件要求

**最低配置：**
- CPU: Intel i5 或同等性能
- 内存: 8GB RAM
- 存储: 20GB 可用空间

**推荐配置：**
- CPU: Intel i7 或 AMD Ryzen 7
- 内存: 16GB RAM
- GPU: NVIDIA GTX 1060 或更高（6GB+ 显存）
- 存储: 50GB 可用空间（SSD 推荐）

### 训练时间估算

| 模型 | 数据量 | CPU 训练时间 | GPU 训练时间 |
|------|--------|--------------|--------------|
| SVM 元音分类 | 1000 样本 | 1-2 分钟 | N/A |
| LSTM 声调分类 | 5000 样本 | 30-60 分钟 | 5-10 分钟 |
| wav2vec 微调 | 10 小时音频 | 不推荐 | 2-4 小时 |
| GPT-SoVITS 微调 | 1 分钟音频 | 不推荐 | 10-20 分钟 |

### 内存使用

- SVM 训练：< 1GB
- LSTM 训练：2-4GB
- wav2vec 微调：8-12GB（GPU）
- GPT-SoVITS 推理：4-6GB（GPU）

---

## 更新日志

### 2026-03-02
- 初始化项目结构
- 创建 CLAUDE.md 项目指南
- 创建 README.md 项目文档
- 定义工作流协议（code_commit_protocol, readme_update_protocol）
- 规划 10 个课程模块的技术路线
- 配置 Python 环境和依赖管理
- 建立 src/ 核心代码模块结构
- 建立 scripts/ 可执行脚本目录
- 配置 .gitignore 排除数据文件和模型权重

---

## 常见问题

### Q1: 如何选择 Python 版本？
A: 推荐使用 Python 3.10，兼容性最好。避免使用 3.12+（部分依赖可能不兼容）。

### Q2: MFA 对齐失败怎么办？
A: 检查以下几点：
- 音频采样率是否为 16kHz
- TextGrid 层命名是否为 `words` 和 `phones`
- 发音词典格式是否正确
- 音频文件和文本文件名是否一致

### Q3: GPU 训练时显存不足？
A: 尝试以下方法：
- 减小 `batch_size`
- 使用梯度累积
- 使用混合精度训练（`torch.cuda.amp`）
- 冻结部分模型层

### Q4: 如何加速依赖安装？
A: 使用清华镜像：
```bash
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple --trusted-host pypi.tuna.tsinghua.edu.cn
```

### Q5: VS Code 无法识别虚拟环境？
A:
1. 右下角点击 Python 版本
2. 选择 `dialect-speech` 环境
3. 重启 VS Code

---

## 参考资源

### 官方文档
- [Praat 官网](https://www.fon.hum.uva.nl/praat/)
- [MFA 文档](https://montreal-forced-aligner.readthedocs.io/)
- [Parselmouth 文档](https://parselmouth.readthedocs.io/)
- [GPT-SoVITS GitHub](https://github.com/RVC-Boss/GPT-SoVITS)
- [Hugging Face Transformers](https://huggingface.co/docs/transformers/)

### 开源数据集
- [CommonVoice](https://commonvoice.mozilla.org/)
- [标贝科技中文数据集](https://www.data-baker.com/data/index/TNtts)
- [THCHS30](https://openslr.org/18/)

### 学习资源
- [StatQuest Neural Networks](https://www.youtube.com/playlist?list=PLblh5JKOoLUIxGDQs4LFFD--41Vzf-ME1)
- [极地语音工作室公众号](https://mp.weixin.qq.com/)

---

**项目维护者：** JNU 智能语音研究团队
**最后更新：** 2026-03-02
