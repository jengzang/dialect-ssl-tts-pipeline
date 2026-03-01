# 使用示例与最佳实践

本文档提供已实现模块的详细使用示例和最佳实践。

---

## 环境准备

### 1. 创建虚拟环境

```bash
# 使用 Miniforge/Mamba（推荐）
mamba create -n dialect-speech python=3.10 -y
mamba activate dialect-speech

# 或使用 Conda
conda create -n dialect-speech python=3.10 -y
conda activate dialect-speech
```

### 2. 安装依赖

```bash
# 基础依赖（Lesson 4 需要）
pip install numpy pandas scipy
pip install librosa soundfile praat-parselmouth
pip install scikit-learn joblib
pip install matplotlib seaborn tqdm pyyaml

# 深度学习依赖（Lesson 6 需要）
pip install torch torchaudio

# Transformers 依赖（Lesson 7 需要）
pip install transformers datasets evaluate

# 或一次性安装
pip install -r requirements.txt
```

### 3. 验证安装

```bash
python scripts/test_infrastructure.py
```

---

## Lesson 4: SVM 元音分类器

### 场景1：训练新模型

```bash
# 完整的训练流程
python scripts/lesson_04_svm_vowel.py \
    --mode train \
    --audio_dir material/lesson_4/cantonese_v2 \
    --textgrid_dir material/lesson_4/cantonese_v2_out_TG \
    --target_vowels a e i o u \
    --model_path checkpoints/svm_vowel.pkl \
    --output_dir results/lesson_04
```

**输出文件**:
- `results/lesson_04/features.csv` - 提取的特征
- `results/lesson_04/confusion_matrix.png` - 混淆矩阵
- `results/lesson_04/tsne.png` - t-SNE 可视化
- `results/lesson_04/feature_distribution.png` - 特征分布
- `checkpoints/svm_vowel.pkl` - 训练好的模型
- `logs/lesson_04_svm_*.log` - 训练日志

### 场景2：使用已训练模型进行推理

```bash
python scripts/lesson_04_svm_vowel.py \
    --mode inference \
    --model_path checkpoints/svm_vowel.pkl \
    --test_data results/lesson_04/features.csv \
    --output_dir results/lesson_04_inference
```

### 场景3：自定义目标元音

```bash
# 只分类 a, e, i 三个元音
python scripts/lesson_04_svm_vowel.py \
    --mode train \
    --audio_dir material/lesson_4/cantonese_v2 \
    --textgrid_dir material/lesson_4/cantonese_v2_out_TG \
    --target_vowels a e i \
    --model_path checkpoints/svm_vowel_3class.pkl \
    --output_dir results/lesson_04_3class
```

### 最佳实践

1. **数据准备**:
   - 确保音频文件和 TextGrid 文件名匹配
   - TextGrid 应包含 words 和 phones 两层
   - 音频采样率建议 16kHz

2. **特征选择**:
   - 元音分类主要依赖共振峰（F1, F2, F3）
   - 可以添加时长特征提高准确率
   - 避免使用过多特征导致过拟合

3. **模型调优**:
   - 调整 SVM 核函数（rbf, linear, poly）
   - 调整正则化参数 C
   - 使用网格搜索找最优参数

---

## Lesson 6: LSTM 声调分类器

### 场景1：训练新模型

```bash
# 完整的训练流程
python scripts/lesson_06_lstm_tone.py \
    --mode train \
    --data_file material/lesson_6/vowel_with_tone.csv \
    --model_path checkpoints/lstm_tone.pth \
    --output_dir results/lesson_06 \
    --epochs 50 \
    --batch_size 32 \
    --test_size 0.2 \
    --val_size 0.1
```

**输出文件**:
- `results/lesson_06/confusion_matrix.png` - 混淆矩阵
- `results/lesson_06/training_curves.png` - 训练曲线
- `checkpoints/lstm_tone.pth` - 训练好的模型
- `logs/lesson_06_lstm_*.log` - 训练日志

### 场景2：使用 GPU 加速训练

```bash
python scripts/lesson_06_lstm_tone.py \
    --mode train \
    --data_file material/lesson_6/vowel_with_tone.csv \
    --model_path checkpoints/lstm_tone.pth \
    --output_dir results/lesson_06 \
    --device cuda \
    --epochs 100 \
    --batch_size 64
```

### 场景3：调整模型架构

修改 `config.yaml`:

```yaml
lstm:
  hidden_size: 256        # 增加隐藏层维度
  num_layers: 3           # 增加层数
  dropout: 0.5            # 增加 dropout
  learning_rate: 0.0005   # 降低学习率
  batch_size: 16          # 减小批次大小
  epochs: 100
  early_stopping_patience: 10
```

### 场景4：推理模式

```bash
python scripts/lesson_06_lstm_tone.py \
    --mode inference \
    --model_path checkpoints/lstm_tone.pth \
    --test_data material/lesson_6/vowel_with_tone.csv \
    --output_dir results/lesson_06_inference
```

### 最佳实践

1. **数据准备**:
   - CSV 文件应包含基频点序列（f0_point_0 到 f0_point_9）
   - 包含统计特征（f0_mean, f0_std 等）
   - 标签格式：phoneme 列包含声调信息（如 "a1", "e2"）

2. **训练技巧**:
   - 使用早停防止过拟合
   - 监控训练曲线，及时调整学习率
   - 如果验证集损失不下降，尝试降低学习率
   - GPU 训练可以使用更大的批次大小

3. **模型调优**:
   - 隐藏层维度：128-256
   - 层数：2-3 层
   - Dropout：0.3-0.5
   - 学习率：0.0001-0.001

---

## Lesson 7: wav2vec 2.0 IPA 识别器

### 场景1：从 CSV 训练

```bash
# 安装依赖
pip install transformers datasets evaluate

# 训练
python scripts/lesson_07_wav2vec_ipa.py \
    --mode train \
    --csv_file material/lesson_7/data.csv \
    --audio_column audio_path \
    --text_column ipa_text \
    --model_name facebook/wav2vec2-base \
    --output_dir checkpoints/wav2vec_ipa \
    --epochs 10
```

**CSV 格式**:
```csv
audio_path,ipa_text
/path/to/audio1.wav,ə l ə ʊ
/path/to/audio2.wav,h ɛ l ə ʊ
```

### 场景2：从目录训练

```bash
python scripts/lesson_07_wav2vec_ipa.py \
    --mode train \
    --data_dir material/lesson_7/audio \
    --transcript_file material/lesson_7/transcripts.txt \
    --model_name facebook/wav2vec2-base \
    --output_dir checkpoints/wav2vec_ipa \
    --epochs 10
```

**转录文件格式** (`transcripts.txt`):
```
audio1.wav|ə l ə ʊ
audio2.wav|h ɛ l ə ʊ
```

### 场景3：使用不同的预训练模型

```bash
# 使用 wav2vec2-large
python scripts/lesson_07_wav2vec_ipa.py \
    --mode train \
    --csv_file data.csv \
    --model_name facebook/wav2vec2-large \
    --output_dir checkpoints/wav2vec_ipa_large \
    --epochs 10

# 使用中文预训练模型
python scripts/lesson_07_wav2vec_ipa.py \
    --mode train \
    --csv_file data.csv \
    --model_name jonatasgrosman/wav2vec2-large-xlsr-53-chinese-zh-cn \
    --output_dir checkpoints/wav2vec_ipa_chinese \
    --epochs 10
```

### 场景4：推理模式

```bash
python scripts/lesson_07_wav2vec_ipa.py \
    --mode inference \
    --model_path checkpoints/wav2vec_ipa \
    --audio_file test.wav
```

### 场景5：调整训练参数

修改 `config.yaml`:

```yaml
wav2vec:
  model_name: facebook/wav2vec2-base
  learning_rate: 3e-4
  warmup_steps: 1000
  freeze_feature_encoder: true
  batch_size: 16
  epochs: 20
  gradient_accumulation_steps: 4
```

### 最佳实践

1. **数据准备**:
   - 音频必须是 16kHz 采样率（自动重采样）
   - 文本标注使用 IPA 符号
   - 最少需要 1 小时标注音频
   - 建议 10+ 小时获得更好效果

2. **模型选择**:
   - `wav2vec2-base`: 9500 万参数，适合快速实验
   - `wav2vec2-large`: 3.17 亿参数，更好的性能
   - `wav2vec2-large-xlsr-53`: 多语言预训练，适合低资源语言

3. **训练技巧**:
   - 冻结特征提取器可以加快训练
   - 使用梯度累积模拟大批次
   - 学习率预热提高稳定性
   - 监控 WER 指标

4. **显存优化**:
   - 减小批次大小
   - 增加梯度累积步数
   - 使用混合精度训练（自动启用）
   - 冻结特征提取器

---

## 常见问题

### Q1: 如何处理音频采样率不一致？

**A**: 所有模块都会自动处理采样率：
- SVM: Praat 自动处理
- LSTM: librosa 自动重采样
- wav2vec: HuggingFace datasets 自动重采样到 16kHz

### Q2: 训练时显存不足怎么办？

**A**:
```bash
# 减小批次大小
--batch_size 8

# 使用梯度累积
# 在 config.yaml 中设置
gradient_accumulation_steps: 4

# 冻结特征提取器（wav2vec）
freeze_feature_encoder: true
```

### Q3: 如何提高模型准确率？

**A**:
1. **增加数据量**: 更多标注数据
2. **数据增强**: 添加噪声、变速、变调
3. **调整模型**: 增加隐藏层维度、层数
4. **集成学习**: 训练多个模型投票
5. **超参数搜索**: 网格搜索或贝叶斯优化

### Q4: 如何导出模型用于生产环境？

**A**:
```python
# SVM: 已经是 pickle 格式，直接使用
import joblib
model = joblib.load('checkpoints/svm_vowel.pkl')

# LSTM: 导出为 TorchScript
import torch
model = torch.jit.script(lstm_model)
model.save('model.pt')

# wav2vec: 使用 HuggingFace 格式
from transformers import Wav2Vec2ForCTC
model = Wav2Vec2ForCTC.from_pretrained('checkpoints/wav2vec_ipa')
```

### Q5: 如何在 CPU 上训练？

**A**:
```bash
# 所有脚本都支持 CPU 训练
--device cpu

# 或在 config.yaml 中设置
training:
  device: cpu
```

---

## 性能优化建议

### 1. 数据加载优化

```yaml
# config.yaml
training:
  num_workers: 4        # 多进程加载数据
  pin_memory: true      # 固定内存（GPU）
```

### 2. 训练加速

```bash
# 使用混合精度训练（PyTorch）
# 在代码中启用 AMP

# 使用分布式训练
python -m torch.distributed.launch \
    --nproc_per_node=2 \
    scripts/lesson_06_lstm_tone.py \
    --mode train \
    ...
```

### 3. 模型压缩

```python
# 量化（PyTorch）
import torch.quantization as quantization

model_quantized = quantization.quantize_dynamic(
    model, {torch.nn.Linear}, dtype=torch.qint8
)

# 剪枝
import torch.nn.utils.prune as prune

prune.l1_unstructured(model.lstm, name='weight_ih_l0', amount=0.3)
```

---

## 调试技巧

### 1. 启用详细日志

```yaml
# config.yaml
logging:
  level: DEBUG
```

### 2. 可视化训练过程

```python
# 使用 TensorBoard
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter('runs/experiment_1')
writer.add_scalar('Loss/train', loss, epoch)
```

### 3. 检查数据

```python
# 打印数据集样本
for batch in train_loader:
    print(batch)
    break
```

---

## 总结

本文档提供了已实现模块的详细使用示例和最佳实践。通过这些示例，你可以：

1. 快速上手每个模块
2. 了解常见使用场景
3. 掌握调优技巧
4. 解决常见问题

建议按照 Lesson 4 → Lesson 6 → Lesson 7 的顺序学习，逐步掌握从传统机器学习到深度学习的完整技术栈。
