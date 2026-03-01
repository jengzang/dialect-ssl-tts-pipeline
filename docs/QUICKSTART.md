# 快速开始指南

本指南帮助你快速上手智能语音与方言大模型项目。

## 前置要求

- Python 3.10
- Miniforge 或 Conda
- 8GB+ RAM
- 20GB+ 可用磁盘空间

---

## 步骤 1: 环境配置

### 1.1 创建虚拟环境

```bash
# 使用 Miniforge/Mamba（推荐）
mamba create -n dialect-speech python=3.10 -y
mamba activate dialect-speech

# 或使用 Conda
conda create -n dialect-speech python=3.10 -y
conda activate dialect-speech
```

### 1.2 安装依赖

```bash
# 进入项目目录
cd JNU

# 安装基础依赖（Lesson 4 需要）
pip install numpy==1.24.3 pandas==2.0.3 scipy==1.10.1
pip install librosa==0.10.1 soundfile==0.12.1 praat-parselmouth==0.4.3
pip install scikit-learn==1.3.2 joblib==1.3.2
pip install matplotlib==3.7.3 seaborn==0.12.2
pip install tqdm==4.66.1 pyyaml==6.0.1

# 或使用清华镜像（国内推荐）
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

---

## 步骤 2: 验证安装

运行基础设施验证脚本：

```bash
python scripts/test_infrastructure.py
```

**预期输出：**
```
==================================================
基础设施验证
==================================================

[OK] 配置文件加载成功
[OK] 日志系统工作正常
[WARN] PyTorch 未安装，跳过设备管理器测试

==================================================
[OK] 所有测试通过！
==================================================
```

---

## 步骤 3: 运行第一个实验（Lesson 4: SVM 元音分类）

### 3.1 准备数据

课程材料已包含在 `material/lesson_4/` 目录中。

### 3.2 训练模型

```bash
python scripts/lesson_04_svm_vowel.py \
    --mode train \
    --audio_dir material/lesson_4/cantonese_v2 \
    --textgrid_dir material/lesson_4/cantonese_v2_out_TG \
    --target_vowels a e i o u \
    --model_path checkpoints/svm_vowel.pkl \
    --output_dir results/lesson_04
```

**注意：** 如果音频文件不在 `cantonese_v2` 目录，请根据实际情况调整路径。

### 3.3 查看结果

训练完成后，查看生成的文件：

```bash
# 查看输出目录
ls results/lesson_04/

# 预期文件：
# - features.csv              # 提取的特征
# - confusion_matrix.png      # 混淆矩阵
# - tsne.png                  # t-SNE 可视化
# - feature_distribution.png  # 特征分布
# - dataset/                  # 数据集文件

# 查看模型文件
ls checkpoints/svm_vowel.pkl
```

### 3.4 查看日志

```bash
# 查看最新的日志文件
ls -lt logs/ | head -5
```

---

## 步骤 4: 使用训练好的模型进行推理

```bash
python scripts/lesson_04_svm_vowel.py \
    --mode inference \
    --model_path checkpoints/svm_vowel.pkl \
    --test_data results/lesson_04/features.csv \
    --output_dir results/lesson_04_inference
```

---

## 常见问题排查

### 问题 1: ModuleNotFoundError

**错误信息：**
```
ModuleNotFoundError: No module named 'parselmouth'
```

**解决方案：**
```bash
pip install praat-parselmouth
```

### 问题 2: 音频文件不存在

**错误信息：**
```
FileNotFoundError: [Errno 2] No such file or directory: 'material/lesson_4/cantonese_v2'
```

**解决方案：**
1. 检查音频文件是否已解压
2. 确认音频文件的实际路径
3. 调整命令行参数中的 `--audio_dir`

### 问题 3: TextGrid 文件不存在

**错误信息：**
```
WARNING: TextGrid 文件不存在: ...
```

**解决方案：**
1. 确认 TextGrid 文件路径
2. 检查文件扩展名（应为 `.TextGrid`）
3. 确保音频文件和 TextGrid 文件名匹配

### 问题 4: 内存不足

**错误信息：**
```
MemoryError: Unable to allocate array
```

**解决方案：**
1. 减少处理的文件数量
2. 关闭其他占用内存的程序
3. 使用更小的数据集进行测试

---

## 下一步

### 学习路径

1. ✅ **Lesson 4: SVM 元音分类** - 你已经完成！
2. 🔜 **Lesson 6: LSTM 声调分类** - 即将推出
3. 🔜 **Lesson 7: wav2vec IPA 识别** - 即将推出

### 深入了解

- 阅读 `CLAUDE.md` 了解项目架构
- 阅读 `PROJECT_STATUS.md` 了解实施状态
- 查看 `src/` 目录了解代码结构
- 查看 `config.yaml` 了解配置选项

### 自定义实验

修改 `config.yaml` 中的参数：

```yaml
# SVM 配置
svm:
  kernel: rbf        # 核函数：rbf, linear, poly
  C: 1.0             # 正则化参数
  gamma: auto        # 核系数
  test_size: 0.25    # 测试集比例
  random_state: 42   # 随机种子
```

---

## 获取帮助

### 查看命令行帮助

```bash
python scripts/lesson_04_svm_vowel.py --help
```

### 查看文档

- `README.md` - 项目总览
- `CLAUDE.md` - 开发指南
- `PROJECT_STATUS.md` - 实施状态
- `skill.md` - 技术规范

### 常见问题

查看 `README.md` 中的"常见问题"章节。

---

**祝你实验顺利！** 🎉
