# 智能语音与方言大模型本地实验工程 (Intelligent Speech & Dialect LLM Local Research)

## 1. 架构师导言与核心愿景 (Architect's Vision)
你现在是一个顶尖的 AI 算法研究员。本项目的核心目标是在**完全本地化的环境**中，构建一套端到端的方言语音处理、模型微调与量化评估流水线。
本项目没有任何前端展示或后端服务部署的需求。你的任务是通过编写高效、模块化的 Python 脚本，结合 Claude Code CLI 进行自动化实验，输出严谨的实验日志、模型权重以及本地可视化分析图表。

---

## 2. 本地核心技术栈与工具链 (Local Tech Stack & Toolchain)
- **核心理论与方法论**：Self-Supervised Learning (SSL), Out-of-Distribution (OOD) Generalization, Zero-shot/Few-shot Inference, Sequence Modeling.
- **声学与语音模型**：wav2vec 2.0 (Hugging Face), GPT-SoVITS (本地克隆与运行)
- **机器学习与深度学习**：PyTorch (本地 GPU 加速), Scikit-learn
- **音频处理与标注**：MFA (Montreal Forced Aligner 本地环境), Praat (Parselmouth), Librosa
- **本地可视化与报告分析**：Matplotlib, Seaborn, Pandas, TensorBoard/Wandb (用于本地训练指标监控)

---

## 3. 核心实验模块与深度实现规范 (Core Modules & Implementation Details)

### 模块一：序列建模与本地深度误差分析 (Sequence Modeling & Error Analysis)
**技术边界**：基于 40k+ 数据集，在本地构建并调优声学分类器。
- **本地数据管线**：编写独立的 `data_pipeline.py`，串联 MFA 强制对齐，并使用 Librosa/Parselmouth 提取时序特征 (时长、F0、MFCCs)，清洗后将数据集保存为本地 `.h5` 或 `.csv` 文件。
- **模型训练与持久化**：使用 PyTorch 构建 LSTM/SVM，编写标准的本地训练循环 (Training Loop)，并将最优模型权重保存至本地 `checkpoints/` 目录。
- **本地误差分析 (Deep Error Analysis) [严格要求]**：
  - 编写独立的 `analyze_errors.py` 脚本。
  - **决策边界**：使用 t-SNE 降维，利用 Matplotlib 绘制决策边界散点图，保存为 `results/tsne_boundary.png`。
  - **混淆矩阵**：利用 Seaborn 绘制高分辨率的归一化混淆矩阵热力图，保存为本地图像。
  - **Failure Case 追踪**：将分类失败的样本索引、真实标签、预测标签及特征分布导出为 `results/failure_cases_report.csv`。

### 模块二：自监督预训练微调与 OOD 评估 (SSL Fine-tuning & OOD Generalization)
**技术边界**：在本地单机多卡/单卡环境下微调 wav2vec 2.0。
- **本地微调策略**：基于 Hugging Face `Trainer` API 编写微调脚本，冻结特征提取器，仅更新 Transformer 层，日志直接输出到本地 TensorBoard。
- **OOD 测试集构造与量化评估 [严格要求]**：
  - 编写 `ood_generator.py`，通过算法向干净音频中注入不同信噪比 (SNR) 的背景噪声和本地 RIR (房间混响) 文件，生成本地 OOD 音频集。
  - 编写 `evaluate_ood.py`，计算 ID 数据与 OOD 数据上的字错误率 (WER)。
  - **数学与量化**：严格应用 WER 公式 $WER = \frac{S + D + I}{N} \times 100\%$。脚本需最终生成一张 `results/wer_robustness_curve.pdf`，直观展示模型在不同噪声强度下的性能衰减。

### 模块三：生成式 TTS 受控实验与客观度量 (Generative TTS & Controlled Experiments)
**技术边界**：在本地打通 GPT-SoVITS 全管线并进行严谨的对照跑分。
- **自动化清洗与打标脚本**：编写 `prep_tts_data.py`，集成 VAD 切分和本地 Whisper 粗略打标，实现数据清洗的本地自动化。
- **受控对照实验 (Controlled Experiments)**：
  - 准备 Base Model 与 Fine-tuned Model 的本地权重。
  - 编写批量推理脚本 `batch_inference_tts.py`，在 Zero-shot 和 Few-shot 条件下各生成一批测试音频，保存至 `outputs/base_vs_finetuned/` 目录。
- **客观量化指标计算 [严格要求]**：
  - 编写 `calculate_metrics.py`。
  - **音色相似度 (SECS)**：调用本地声纹模型 (如 SpeechBrain 的 ECAPA-TDNN) 提取生成的音频与 Reference 音频的 Embedding，计算余弦相似度并输出均值。
  - **声学失真度**：计算 MCD (Mel-Cepstral Distortion)。
  - 实验结果需汇总为格式化的终端表格 (可用 `rich` 库) 或 `results/tts_evaluation_metrics.csv`。

---

## 4. Claude Code CLI 交互与代码生成指令 (CLI Execution Directives)

在后续的本地开发与复现中，请遵循以下工程规范：

1. **环境初始化**：首先生成 `requirements.txt` 或 `environment.yml`，明确 PyTorch 版本与 CUDA 对应关系，以及 MFA 的本地安装指令。
2. **纯脚本化项目结构**：按照 `src/` (核心逻辑), `scripts/` (执行脚本), `notebooks/` (探索性分析), `results/` (图表与报告) 规划目录。
3. **面向 CLI 的交互设计**：所有的核心脚本（如训练、微调、评估）必须使用 `argparse` 或 `click` 库封装，支持通过命令行参数（如 `--batch_size`, `--snr_level`, `--model_path`）灵活控制实验变量。
4. **日志记录 (Logging)**：摒弃 `print` 调试，强制使用 Python 内置的 `logging` 模块，将运行日志、警告和错误输出到本地 `logs/experiment.log` 文件中。

> **给 Claude Code 的系统指令：**
> “请详细阅读本 `skill.md`。本项目为纯本地算法研究与实验评估，不涉及任何 Web 服务或容器部署。在生成代码时，请侧重于高效的本地文件 I/O、命令行参数解析、以及高质量的本地 Matplotlib/Seaborn 图像生成。所有量化指标（如 WER, SECS）必须以可执行的 Python 脚本形式提供自动计算逻辑。”