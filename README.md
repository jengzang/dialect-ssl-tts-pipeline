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

**项目状态：** 核心模块已完成（8/10） + LLM 训练路径（6/7）

---

## 📚 文档导航

- **[快速开始](docs/QUICKSTART.md)** - 5分钟快速上手指南
- **[使用示例](docs/USAGE_EXAMPLES.md)** - 详细的使用场景和最佳实践
- **[完成报告](docs/COMPLETION_REPORT.md)** - 项目实施总结和技术细节
- **[项目路线图](docs/ROADMAP.md)** - 当前状态和未来计划
- **[开发指南](CLAUDE.md)** - 开发规范和工作流
- **[LLM 训练路径总体进展](docs/OVERALL_PROGRESS_REPORT.md)** - LLM 训练学习路径进度
- **[Project 1 报告](docs/PROJECT_01_COMPLETION_REPORT.md)** - 增强评估系统
- **[Project 2 报告](docs/PROJECT_02_COMPLETION_REPORT.md)** - 超参数优化
- **[Project 3 报告](docs/PROJECT_03_COMPLETION_REPORT.md)** - 多任务学习
- **[Project 4 报告](docs/PROJECT_04_COMPLETION_REPORT.md)** - 指令微调
- **[Project 5 报告](docs/PROJECT_05_COMPLETION_REPORT.md)** - 模型比较与缩放定律
- **[Project 6 报告](docs/PROJECT_06_COMPLETION_REPORT.md)** - 高级微调技术 ⭐ 新增

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

### Lesson 4: SVM 元音分类器 ✅

**原理说明：**

SVM (Support Vector Machine) 是一种经典的机器学习分类算法，通过寻找最优超平面来分隔不同类别的数据。在元音分类任务中，我们使用共振峰（F1, F2, F3）和时长作为特征，训练 SVM 模型来识别不同的元音。

**特征说明：**
- **F1, F2, F3**: 共振峰频率，反映声道共振特性
- **时长**: 元音持续时间
- **基频统计值**: mean, std, min, max（可选）

**使用教程：**

```bash
# 训练模式
python scripts/lesson_04_svm_vowel.py \
    --mode train \
    --audio_dir material/lesson_4/cantonese_v2 \
    --textgrid_dir material/lesson_4/cantonese_v2_out_TG \
    --target_vowels a e i o u \
    --model_path checkpoints/svm_vowel.pkl \
    --output_dir results/lesson_04

# 推理模式
python scripts/lesson_04_svm_vowel.py \
    --mode inference \
    --model_path checkpoints/svm_vowel.pkl \
    --test_data results/lesson_04/features.csv \
    --output_dir results/lesson_04_inference
```

**输出文件：**
- `results/lesson_04/features.csv` - 提取的特征
- `results/lesson_04/confusion_matrix.png` - 混淆矩阵
- `results/lesson_04/tsne.png` - t-SNE 可视化
- `results/lesson_04/feature_distribution.png` - 特征分布
- `checkpoints/svm_vowel.pkl` - 训练好的模型

---

### Lesson 6: LSTM 声调分类 ✅

**原理说明：**

LSTM (Long Short-Term Memory) 是一种循环神经网络（RNN）的变体，特别适合处理序列数据。在声调分类任务中，我们使用基频（F0）序列作为输入，通过 LSTM 捕捉声调的时序变化模式。

**模型架构：**
1. **输入层**: 基频序列 + 统计特征（时间步 × 特征维度）
2. **双向 LSTM**: 同时捕捉前向和后向的时序信息
3. **注意力机制**: 自动学习关注重要的时间步
4. **全连接层**: 分类输出（5 个声调类别）

**特征说明：**
- **基频点序列**: 10 个均匀采样的 F0 值
- **统计特征**: f0_mean, f0_std, f0_min, f0_max, f0_range, f0_median, f0_skew, f0_kurtosis
- **归一化特征**: norm_f0_mean, norm_f0_std
- **差分特征**: delta_mean, delta2_mean（一阶和二阶差分）
- **时长**: 音素持续时间

**使用教程：**

```bash
# 训练模式
python scripts/lesson_06_lstm_tone.py \
    --mode train \
    --data_file material/lesson_6/vowel_with_tone.csv \
    --model_path checkpoints/lstm_tone.pth \
    --output_dir results/lesson_06 \
    --epochs 50 \
    --batch_size 32 \
    --test_size 0.2 \
    --val_size 0.1

# 推理模式
python scripts/lesson_06_lstm_tone.py \
    --mode inference \
    --model_path checkpoints/lstm_tone.pth \
    --test_data material/lesson_6/vowel_with_tone.csv \
    --output_dir results/lesson_06_inference
```

**训练参数说明：**
- `--epochs`: 训练轮数（默认 50）
- `--batch_size`: 批次大小（默认 32）
- `--test_size`: 测试集比例（默认 0.2）
- `--val_size`: 验证集比例（默认 0.1）
- `--device`: 计算设备（auto, cpu, cuda）

**输出文件：**
- `results/lesson_06/confusion_matrix.png` - 混淆矩阵
- `results/lesson_06/training_curves.png` - 训练曲线（损失和准确率）
- `checkpoints/lstm_tone.pth` - 训练好的模型

**训练技巧：**
1. **早停机制**: 验证集损失连续 5 个 epoch 不下降时自动停止
2. **学习率调度**: 验证集损失不下降时自动降低学习率
3. **梯度裁剪**: 防止梯度爆炸
4. **Dropout**: 防止过拟合（默认 0.3）

**性能预期：**
- 训练时间: 约 10-20 分钟（CPU）/ 2-5 分钟（GPU）
- 测试准确率: >85%（取决于数据质量）
- 内存占用: 约 2-4GB

---

### Lesson 5: MFA 模型训练 ✅

**原理说明：**

Montreal Forced Aligner (MFA) 是一个强大的语音对齐工具，可以自动将音频和文本对齐，生成 TextGrid 文件。MFA 还支持训练自定义声学模型，适应特定的语言或方言。

**功能说明：**
- **语料验证**: 检查音频和文本配对
- **声学模型训练**: 使用自定义语料训练模型
- **模型测试**: 使用训练好的模型进行对齐

**使用教程：**

```bash
# 验证语料
python scripts/lesson_05_mfa_training.py --mode validate \
    --corpus_dir data/mfa_corpus \
    --dictionary_path data/dict.txt

# 训练模型
python scripts/lesson_05_mfa_training.py --mode train \
    --corpus_dir data/mfa_corpus \
    --dictionary_path data/dict.txt \
    --output_dir checkpoints/mfa_model \
    --num_jobs 4

# 测试对齐
python scripts/lesson_05_mfa_training.py --mode align \
    --model_path checkpoints/mfa_model/acoustic_model.zip \
    --test_corpus data/test_corpus \
    --dictionary_path data/dict.txt \
    --output_dir results/alignments

# 列出可用模型
python scripts/lesson_05_mfa_training.py --mode list_models \
    --model_type acoustic
```

**语料目录结构：**
```
corpus_dir/
    speaker1/
        audio1.wav
        audio1.txt
        audio2.wav
        audio2.txt
    speaker2/
        ...
```

**输出文件：**
- `checkpoints/mfa_model/acoustic_model.zip` - 训练好的声学模型
- `checkpoints/mfa_model/training_metadata.json` - 训练元数据
- `results/alignments/*.TextGrid` - 对齐结果

---

### Lesson 8: 方言翻译 ✅

**原理说明：**

使用 LoRA (Low-Rank Adaptation) 技术微调大语言模型，实现方言到普通话的翻译。LoRA 是一种参数高效的微调方法，只需训练少量参数即可适应新任务。

**模型架构：**
- **基础模型**: Qwen/ChatGLM/LLaMA 等大语言模型
- **LoRA 适配器**: 低秩矩阵分解，减少可训练参数
- **量化支持**: 4-bit 量化，降低显存需求

**数据格式（JSON）：**
```json
[
    {
        "dialect": "我今日好开心啊",
        "mandarin": "我今天很开心"
    },
    ...
]
```

**使用教程：**

```bash
# 训练模型
python scripts/lesson_08_dialect_translation.py --mode train \
    --model_name Qwen/Qwen-7B-Chat \
    --train_data data/dialect_parallel.json \
    --output_dir checkpoints/dialect_translator \
    --epochs 3 \
    --batch_size 4 \
    --lora_r 8 \
    --quantization

# 推理翻译
python scripts/lesson_08_dialect_translation.py --mode inference \
    --model_path checkpoints/dialect_translator/best \
    --dialect_text "我今日好开心啊"

# 批量翻译
python scripts/lesson_08_dialect_translation.py --mode batch \
    --model_path checkpoints/dialect_translator/best \
    --input_file data/test_dialect.txt \
    --output_file results/translations.txt
```

**训练参数说明：**
- `--lora_r`: LoRA rank（默认 8）
- `--lora_alpha`: LoRA alpha（默认 32）
- `--quantization`: 使用 4-bit 量化
- `--learning_rate`: 学习率（默认 2e-4）

**输出文件：**
- `checkpoints/dialect_translator/best/` - 最佳模型
- `checkpoints/dialect_translator/epoch_*/` - 各轮检查点
- `results/translations.txt` - 翻译结果

**性能预期：**
- 训练时间: 约 1-2 小时（GPU，量化模式）
- 显存占用: 约 8-12GB（4-bit 量化）
- 最小数据量: 建议 >1000 对平行语料

---

### Lesson 9: 方言口音识别 ✅

**原理说明：**

基于 wav2vec 2.0 的方言口音分类器，通过微调预训练模型来识别不同的方言口音。使用全局平均池化提取音频级别的特征，然后通过分类头输出口音类别。

**模型架构：**
1. **特征提取器**: wav2vec 2.0 预训练编码器（冻结）
2. **池化层**: 全局平均池化，将序列特征聚合为固定维度
3. **分类头**: 全连接层 + Dropout + Softmax

**使用教程：**

```bash
# 训练模式
python scripts/lesson_09_accent_recognition.py --mode train \
    --data_dir material/lesson_9/audio \
    --label_file material/lesson_9/labels.csv \
    --model_name facebook/wav2vec2-base \
    --output_dir checkpoints/accent_classifier \
    --epochs 20 \
    --batch_size 8

# 推理模式
python scripts/lesson_09_accent_recognition.py --mode inference \
    --model_path checkpoints/accent_classifier/best_model.pth \
    --audio_file test.wav
```

**数据格式（CSV）：**
```csv
audio_path,accent
audio1.wav,cantonese
audio2.wav,mandarin
audio3.wav,hakka
```

**输出文件：**
- `checkpoints/accent_classifier/best_model.pth` - 最佳模型
- `checkpoints/accent_classifier/training_history.json` - 训练历史
- `results/lesson_09/confusion_matrix.png` - 混淆矩阵

**性能预期：**
- 训练时间: 约 30-60 分钟（GPU）
- 测试准确率: >85%（取决于数据质量）
- 显存占用: 约 6-8GB

---

### Lesson 10: 方言虚拟人 ✅

**原理说明：**

整合 GPT-SoVITS 语音合成和 Sadtalker 虚拟人技术，构建完整的方言虚拟人系统。GPT-SoVITS 实现零样本/少样本语音克隆，Sadtalker 从音频和图像生成虚拟人视频。

**系统架构：**
1. **语音合成**: GPT-SoVITS（5 秒参考音频即可克隆声音）
2. **虚拟人生成**: Sadtalker（音频驱动的面部动画）
3. **端到端流程**: 文本 → 语音 → 视频

**使用教程：**

```bash
# 训练 TTS 模型
python scripts/lesson_10_virtual_human.py --mode train \
    --train_data_dir data/dialect_corpus \
    --output_dir checkpoints/tts_model \
    --epochs 10

# 合成语音
python scripts/lesson_10_virtual_human.py --mode synthesize \
    --text "我今日好开心啊" \
    --ref_audio data/ref.wav \
    --ref_text "参考文本" \
    --output results/synthesized.wav

# 创建虚拟人视频
python scripts/lesson_10_virtual_human.py --mode create_video \
    --text "我今日好开心啊" \
    --ref_audio data/ref.wav \
    --ref_text "参考文本" \
    --avatar_image data/avatar.jpg \
    --output results/virtual_human.mp4

# 批量创建虚拟人
python scripts/lesson_10_virtual_human.py --mode batch \
    --input_file data/texts.txt \
    --ref_audio data/ref.wav \
    --ref_text "参考文本" \
    --avatar_image data/avatar.jpg \
    --output_dir results/virtual_humans
```

**依赖安装：**
```bash
# GPT-SoVITS
git clone https://github.com/RVC-Boss/GPT-SoVITS.git
cd GPT-SoVITS
pip install -r requirements.txt

# Sadtalker
git clone https://github.com/OpenTalker/SadTalker.git
cd SadTalker
pip install -r requirements.txt
```

**输出文件：**
- `results/synthesized.wav` - 合成的语音
- `results/virtual_human.mp4` - 虚拟人视频
- `results/virtual_humans/` - 批量生成的视频

**性能预期：**
- 语音合成: 约 5-10 秒/句（GPU）
- 视频生成: 约 30-60 秒/视频（GPU）
- 显存占用: 约 6-10GB

---

**原理说明：**

wav2vec 2.0 是 Facebook AI 提出的自监督学习模型，通过在大量无标注音频上预训练，学习通用的语音表示。在 IPA（国际音标）识别任务中，我们使用预训练的 wav2vec 2.0 模型进行微调。

**模型架构：**
1. **特征提取器**: CNN 编码器，将原始音频转换为特征序列
2. **Transformer 编码器**: 多层 Transformer，学习上下文表示
3. **CTC 解码器**: 连接时序分类（CTC）头，输出音素序列

**预训练-微调范式：**
- **预训练**: 在大规模无标注音频上学习通用表示（对比学习）
- **微调**: 在小规模标注数据上微调 CTC 头，适应特定任务

**特征说明：**
- **输入**: 原始音频波形（16kHz 采样率）
- **输出**: IPA 音素序列
- **词汇表**: 自动从训练数据提取的音素集合

**使用教程：**

```bash
# 安装额外依赖
pip install transformers==4.35.2 datasets==2.14.6 evaluate==0.4.1

# 训练模式（从 CSV）
python scripts/lesson_07_wav2vec_ipa.py \
    --mode train \
    --csv_file material/lesson_7/data.csv \
    --audio_column audio_path \
    --text_column ipa_text \
    --model_name facebook/wav2vec2-base \
    --output_dir checkpoints/wav2vec_ipa \
    --epochs 10

# 训练模式（从目录）
python scripts/lesson_07_wav2vec_ipa.py \
    --mode train \
    --data_dir material/lesson_7/audio \
    --transcript_file material/lesson_7/transcripts.txt \
    --model_name facebook/wav2vec2-base \
    --output_dir checkpoints/wav2vec_ipa \
    --epochs 10

# 推理模式
python scripts/lesson_07_wav2vec_ipa.py \
    --mode inference \
    --model_path checkpoints/wav2vec_ipa \
    --audio_file test.wav
```

**训练参数说明：**
- `--model_name`: 预训练模型名称（默认 facebook/wav2vec2-base）
- `--epochs`: 训练轮数（默认 10）
- `--csv_file`: CSV 数据文件（包含音频路径和文本）
- `--data_dir`: 音频文件目录
- `--transcript_file`: 转录文件（格式：audio_file|text）

**输出文件：**
- `checkpoints/wav2vec_ipa/` - 微调后的模型
- `checkpoints/wav2vec_ipa/checkpoint-*/` - 训练检查点
- `logs/lesson_07_wav2vec_*.log` - 训练日志

**训练技巧：**
1. **冻结特征提取器**: 只微调 Transformer 和 CTC 头，加快训练
2. **梯度累积**: 在小 GPU 上模拟大批次训练
3. **混合精度训练**: 使用 FP16 加速训练，减少显存占用
4. **学习率预热**: 前 N 步逐渐增加学习率，提高稳定性
5. **早停**: 根据验证集 WER 自动停止训练

**性能预期：**
- 训练时间: 约 2-4 小时（GPU）/ 不推荐 CPU
- 测试 WER: <20%（取决于数据质量和数量）
- 显存占用: 约 8-12GB（batch_size=8）
- 最小数据量: 建议 >1 小时标注音频

**评估指标：**
- **WER (Word Error Rate)**: 词错误率
- **CER (Character Error Rate)**: 字符错误率
- **PER (Phoneme Error Rate)**: 音素错误率

---

## 核心模块说明
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

**LLM 训练路径 - Project 6: 高级微调技术（已完成）**
- 实现 Prefix Tuning 模型（src/models/prefix_tuning_model.py）
  - Prefix 编码器（MLP 重参数化）
  - 模型包装器（冻结预训练参数）
  - 参数效率：13.36%（547K/4.1M）
- 实现 Adapter Layers 模型（src/models/adapter_model.py）
  - 瓶颈式 Adapter 层
  - 模块化设计（每层独立）
  - 参数效率：1.90%（69K/3.6M）⭐ 最高
- 实现高级训练器（src/training/advanced_trainer.py）
  - 统一接口支持 4 种方法（LoRA/Prefix/Adapter/Full）
  - 混合精度训练（FP16/BF16）
  - 自动参数和内存统计
- 实现方法比较 CLI（scripts/advanced_finetuning.py）
  - 多维度评估（参数/内存/速度）
  - 自动化基准测试
  - 最佳方法推荐
- 测试验证（scripts/test_project_06.py）
  - Prefix Tuning 测试通过
  - Adapter 测试通过（最高参数效率）
  - 方法比较测试通过

**LLM 训练路径 - Project 5: 模型比较与缩放定律（已完成）**
- 实现模型工厂（src/models/model_factory.py）
  - 统一模型创建接口
  - 支持多种 GPT-2 配置（25M-117M 参数）
  - LoRA 参数高效微调集成
  - 自动设备管理
- 实现缩放分析器（src/evaluation/scaling_analysis.py）
  - 参数缩放定律分析（Performance ∝ Params^α）
  - 效率指标计算（参数/内存/速度效率）
  - 对数线性回归拟合
  - 4 种可视化图表生成
- 实现模型比较 CLI（scripts/model_comparison.py）
  - list/simulate/compare/analyze 四种模式
  - 推理基准测试
  - 自动化实验流程
- 测试验证（scripts/test_project_05.py）
  - 模型工厂测试通过
  - 缩放分析测试通过（R²=0.996）
  - 生成完整分析报告和图表

**阶段 8: Lesson 10 (方言虚拟人)（已完成）**
- 实现 GPT-SoVITS 模型封装（src/models/gpt_sovits_model.py）
  - 语音合成接口
  - 批量合成
  - 语音克隆
  - 模型微调
- 实现虚拟人训练器（src/training/virtual_human_trainer.py）
  - TTS 模型训练
  - 语音合成
  - Sadtalker 视频生成
  - 批量创建虚拟人
- 实现 CLI 脚本（scripts/lesson_10_virtual_human.py）
  - 训练、合成、创建视频、批量模式
  - 完整的端到端流程

**阶段 7: Lesson 8 (方言翻译)（已完成）**
- 实现方言翻译模型（src/models/dialect_translator.py）
  - 基于 LoRA 的大模型微调
  - 支持 Qwen/ChatGLM/LLaMA
  - 4-bit 量化支持
  - 批量翻译
- 实现翻译训练器（src/training/dialect_translation_trainer.py）
  - LoRA 微调流程
  - 梯度累积
  - 学习率预热
  - 检查点保存
- 实现 CLI 脚本（scripts/lesson_08_dialect_translation.py）
  - 训练、推理、批量翻译模式

**阶段 6: Lesson 9 (方言口音识别)（已完成）**
- 实现口音分类器（src/models/accent_classifier.py）
  - 基于 wav2vec 2.0
  - 全局平均池化
  - 多类分类
- 实现口音训练器（src/training/accent_trainer.py）
  - PyTorch 训练循环
  - 早停机制
  - 学习率调度
- 实现 CLI 脚本（scripts/lesson_09_accent_recognition.py）
  - 训练和推理模式

**阶段 5: Lesson 5 (MFA 模型训练)（已完成）**
- 实现 MFA 封装模块（src/data_pipeline/mfa_wrapper.py）
  - 强制对齐接口
  - 声学模型训练
  - 语料验证
  - 模型下载和管理
- 实现 MFA 训练器（src/training/mfa_trainer.py）
  - 语料准备
  - 模型训练流程
  - 元数据保存
- 实现 CLI 脚本（scripts/lesson_05_mfa_training.py）
  - 验证、训练、对齐、列出模型模式

**阶段 4: Lesson 7 (wav2vec IPA 识别器)（已完成）**
- 实现 wav2vec 数据集模块（src/data_pipeline/wav2vec_dataset.py）
  - 从 CSV 构建 HuggingFace Dataset
  - 从目录构建 Dataset
  - 数据集划分和保存
  - DataCollatorCTCWithPadding（动态填充）
- 实现 wav2vec 模型（src/models/wav2vec_ipa.py）
  - 基于 HuggingFace Transformers
  - Wav2Vec2ForCTC 微调
  - 自动创建词汇表
  - CTC 解码
- 实现 wav2vec 训练器（src/training/wav2vec_trainer.py）
  - 封装 HuggingFace Trainer
  - WER 指标计算
  - 混合精度训练
  - 学习率预热
- 实现 CLI 脚本（scripts/lesson_07_wav2vec_ipa.py）
  - 支持训练模式和推理模式
  - 支持 CSV 和目录两种数据格式
  - 自动依赖检查

**阶段 3: Lesson 6 (LSTM 声调分类器)（已完成）**
- 实现音频处理工具（src/data_pipeline/audio_utils.py）
  - 音频加载和重采样
  - 基频序列提取
  - 基频统计特征计算
  - 归一化和差分特征
- 实现 LSTM 模型（src/models/lstm_tone.py）
  - 双向 LSTM 架构
  - 注意力机制
  - 支持变长序列
- 实现 LSTM 训练器（src/training/lstm_trainer.py）
  - 完整的训练循环
  - 早停机制
  - 学习率调度
  - 训练曲线记录
- 实现 CLI 脚本（scripts/lesson_06_lstm_tone.py）
  - 支持训练模式和推理模式
  - 数据加载和预处理
  - 训练曲线可视化

**阶段 1: 基础设施搭建（已完成）**
- 实现配置加载器（src/utils/config_loader.py）
- 实现日志管理系统（src/utils/logger.py）
- 实现设备管理器（src/utils/device_manager.py）
- 实现抽象基类：
  - 数据处理基类（src/data_pipeline/base.py）
  - 模型基类（src/models/base.py）
  - 训练器基类（src/training/base_trainer.py）
- 创建基础设施验证脚本（scripts/test_infrastructure.py）

**阶段 2: Lesson 4 (SVM) 实现（已完成）**
- 实现 Praat 特征提取器（src/data_pipeline/feature_extractor.py）
  - 支持批量提取共振峰（F1/F2/F3）
  - 支持基频统计值提取（mean/std/min/max）
  - 支持时长特征提取
- 实现数据集构建器（src/data_pipeline/dataset_builder.py）
  - 支持数据集划分（train/test）
  - 支持特征归一化
  - 支持标签编码
- 实现 SVM 分类器（src/models/svm_classifier.py）
  - 基于 sklearn 的 SVM 实现
  - 支持多类分类
  - 支持概率预测
- 实现评估指标模块（src/evaluation/metrics.py）
  - 准确率、精确率、召回率、F1 分数
  - 混淆矩阵
  - 分类报告
- 实现可视化模块（src/evaluation/visualizer.py）
  - 混淆矩阵热力图
  - t-SNE 降维可视化
  - 特征分布图
- 实现 CLI 脚本（scripts/lesson_04_svm_vowel.py）
  - 支持训练模式和推理模式
  - 完整的端到端流程
  - 命令行参数控制

**项目初始化**
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

## 更新日志

---

### 2026-03-02 - Project 4: 指令微调与提示工程

**新增功能：**
- 指令数据集构建器（支持 3 种模板风格）
- 指令遵循模型（少样本 + 思维链推理）
- 指令评估模块（多维度评估指标）
- 完整的 CLI 工具（5 种模式）

**新增文件：**
- `src/data_pipeline/instruction_builder.py` - 指令数据集构建器
- `src/models/instruction_tuned_model.py` - 指令遵循模型
- `src/evaluation/instruction_eval.py` - 指令评估模块
- `scripts/lesson_08_instruction_tuning.py` - CLI 工具
- `scripts/test_project_04.py` - 测试脚本
- `docs/PROJECT_04_COMPLETION_REPORT.md` - Project 4 完成报告

**技术亮点：**
- 参数高效：仅 0.79% 可训练参数（811K / 103M）
- 少样本学习：支持 0-shot 到 5-shot
- 思维链推理：鼓励逐步思考
- 灵活模板：Alpaca, Vicuna, Simple

**使用示例：**
```bash
# 创建指令数据集
python scripts/lesson_08_instruction_tuning.py --mode create_dataset \
    --translation_data data/dialect_parallel_train.json \
    --classification_data data/accent_train.json \
    --output_dir data/instruction_dataset \
    --template_style alpaca

# 训练
python scripts/lesson_08_instruction_tuning.py --mode train \
    --train_data data/instruction_dataset/train.json \
    --val_data data/instruction_dataset/val.json \
    --output_dir checkpoints/instruction_tuned

# 零样本推理
python scripts/lesson_08_instruction_tuning.py --mode inference \
    --model_path checkpoints/instruction_tuned/best \
    --instruction "请将以下方言翻译成普通话" \
    --input "侬好啊"

# 少样本推理
python scripts/lesson_08_instruction_tuning.py --mode few_shot \
    --model_path checkpoints/instruction_tuned/best \
    --instruction "请将以下方言翻译成普通话" \
    --examples_file data/few_shot_examples.json \
    --input "侬吃饭了伐"
```

**测试结果：**
- ✅ 指令构建器测试通过
- ✅ 数据集创建测试通过（9 个指令）
- ✅ 模型测试通过（loss=11.48, 0.79% 可训练参数）
- ✅ 评估器测试通过

---

### 2026-03-02 - Project 3: 多任务学习 - 方言翻译 + 口音分类

**新增功能：**
- 多任务数据集（结合翻译和分类任务）
- 多任务模型架构（共享编码器 + 任务特定头）
- 多任务训练器（损失加权、梯度裁剪）
- 完整的 CLI 工具（训练、评估、推理）

**新增文件：**
- `src/data_pipeline/multitask_dataset.py` - 多任务数据集
- `src/models/multitask_dialect_model.py` - 多任务模型
- `src/training/multitask_trainer.py` - 多任务训练器
- `scripts/lesson_08_09_multitask.py` - CLI 脚本
- `scripts/test_project_03.py` - 测试脚本
- `docs/PROJECT_03_COMPLETION_REPORT.md` - Project 3 完成报告

**技术亮点：**
- 参数高效：仅 1.07% 可训练参数（1.1M / 103M）
- 任务平衡：支持 4 种采样策略
- FP16 混合精度训练
- 自动设备管理

**使用示例：**
```bash
# 训练多任务模型
python scripts/lesson_08_09_multitask.py --mode train \
    --translation_data data/dialect_parallel_train.json \
    --classification_data data/accent_train.json \
    --output_dir checkpoints/multitask \
    --task_sampling balanced

# 评估
python scripts/lesson_08_09_multitask.py --mode evaluate \
    --model_path checkpoints/multitask/best \
    --translation_data data/dialect_parallel_test.json \
    --classification_data data/accent_test.json

# 推理 - 翻译
python scripts/lesson_08_09_multitask.py --mode inference \
    --model_path checkpoints/multitask/best \
    --task translation \
    --input "侬好啊"

# 推理 - 分类
python scripts/lesson_08_09_multitask.py --mode inference \
    --model_path checkpoints/multitask/best \
    --task classification \
    --input "侬好啊，今朝天气老好额"
```

**测试结果：**
- ✅ 数据集测试通过
- ✅ 翻译任务 loss: 10.96
- ✅ 分类任务 loss: 1.38
- ✅ 所有测试通过

---

### 2026-03-02 - Project 2: 超参数优化与 LoRA 架构搜索

**新增功能：**
- Optuna 贝叶斯优化集成
- LoRA 权重分析和可视化
- 帕累托前沿识别
- 完整的超参数搜索流程

**新增文件：**
- `src/training/hyperparameter_search.py` - Optuna 集成
- `src/evaluation/lora_analysis.py` - LoRA 分析
- `scripts/lesson_08_hp_search.py` - 超参数搜索 CLI
- `docs/PROJECT_02_COMPLETION_REPORT.md` - Project 2 完成报告

**优化结果：**
- 最佳配置：Rank=32, Alpha=128, LR=1.77e-4
- 模拟 BLEU: 40.00
- 帕累托前沿：17 个最优配置

**使用示例：**
```bash
# 运行超参数搜索（模拟模式）
python scripts/lesson_08_hp_search.py \
    --mode simulate \
    --n_trials 30 \
    --output_dir results/hp_search_demo

# 分析结果
python scripts/lesson_08_hp_search.py \
    --mode analyze \
    --study_dir results/hp_search_demo
```

---

### 2026-03-02 - Project 1: 增强方言翻译评估系统

**新增功能：**
- 机器翻译评估指标（BLEU、ROUGE、ChrF、METEOR）
- 数据增强模块（从 24 样本扩展到 500+ 样本）
- WandB 实验跟踪集成
- 完整的评估流程（evaluate 模式）

**新增文件：**
- `src/evaluation/mt_metrics.py` - MT 评估指标
- `src/data_pipeline/dialect_augmentation.py` - 数据增强
- `scripts/augment_dialect_data.py` - 数据增强脚本
- `docs/project_01_enhanced_evaluation.md` - Project 1 文档

**修改文件：**
- `requirements.txt` - 添加 peft, accelerate, bitsandbytes, evaluate, sacrebleu, rouge-score
- `src/training/dialect_translation_trainer.py` - 添加 WandB 日志
- `scripts/lesson_08_dialect_translation.py` - 添加评估模式

**使用示例：**
```bash
# 数据增强
python scripts/augment_dialect_data.py \
    --input material/lesson_8/dialect2mandarin.csv \
    --output_dir data/dialect_translation \
    --target_size 500

# 训练（带 WandB）
python scripts/lesson_08_dialect_translation.py \
    --mode train \
    --train_data data/dialect_translation/train.json \
    --val_data data/dialect_translation/val.json \
    --use_wandb

# 评估
python scripts/lesson_08_dialect_translation.py \
    --mode evaluate \
    --model_path checkpoints/dialect_translation_v2/best \
    --test_data data/dialect_translation/test.json
```

**目标指标：**
- BLEU-4 > 30
- ROUGE-L F1 > 50
- ChrF > 40

---

**项目维护者：** jengzang (不羁)
**最后更新：** 2026-03-02
**LLM 训练路径进度：** 4/7 项目完成 (57.1%)
