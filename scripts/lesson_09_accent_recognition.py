"""Lesson 9: 方言口音识别器

命令行接口脚本，用于训练和测试方言口音分类器。

使用示例:
    # 训练模式
    python scripts/lesson_09_accent_recognition.py \\
        --mode train \\
        --data_dir material/lesson_9/audio \\
        --labels_file material/lesson_9/labels.csv \\
        --output_dir checkpoints/accent_classifier \\
        --epochs 20

    # 推理模式
    python scripts/lesson_09_accent_recognition.py \\
        --mode inference \\
        --model_path checkpoints/accent_classifier \\
        --audio_file test.wav
"""

import sys
from pathlib import Path
import argparse
import pandas as pd
import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.utils import get_config_loader, setup_logger, get_device
from src.models.accent_classifier import AccentClassifier
from src.training.accent_trainer import AccentTrainer
from src.evaluation.metrics import MetricsCalculator
from src.evaluation.visualizer import Visualizer


class AccentDataset(Dataset):
    """口音数据集"""

    def __init__(self, audio_paths, labels, processor, target_sr=16000):
        self.audio_paths = audio_paths
        self.labels = labels
        self.processor = processor
        self.target_sr = target_sr

    def __len__(self):
        return len(self.audio_paths)

    def __getitem__(self, idx):
        # 加载音频
        waveform, sr = torchaudio.load(self.audio_paths[idx])

        # 重采样
        if sr != self.target_sr:
            resampler = torchaudio.transforms.Resample(sr, self.target_sr)
            waveform = resampler(waveform)

        # 转为单声道
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        # 处理
        inputs = self.processor(
            waveform.squeeze().numpy(),
            sampling_rate=self.target_sr,
            return_tensors="pt",
            padding=True
        )

        return {
            'audio': inputs.input_values.squeeze(),
            'label': torch.tensor(self.labels[idx], dtype=torch.long)
        }


def train_mode(args, config, logger):
    """训练模式"""
    logger.info("=" * 60)
    logger.info("Lesson 9: 方言口音识别器 - 训练模式")
    logger.info("=" * 60)

    # 加载数据
    logger.info("\n步骤 1: 加载数据")
    logger.info("-" * 60)

    df = pd.read_csv(args.labels_file)
    logger.info(f"数据集大小: {len(df)}")

    # 编码标签
    label_encoder = LabelEncoder()
    labels = label_encoder.fit_transform(df['accent'])

    logger.info(f"口音类别: {label_encoder.classes_}")
    logger.info(f"类别数量: {len(label_encoder.classes_)}")

    # 划分数据集
    X_train, X_test, y_train, y_test = train_test_split(
        df['audio_path'].values,
        labels,
        test_size=0.2,
        random_state=42,
        stratify=labels
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train,
        test_size=0.1,
        random_state=42,
        stratify=y_train
    )

    logger.info(f"训练集: {len(X_train)}, 验证集: {len(X_val)}, 测试集: {len(X_test)}")

    # 构建模型
    logger.info("\n步骤 2: 构建模型")
    logger.info("-" * 60)

    device = get_device(args.device)
    if device is None:
        logger.error("PyTorch 未安装")
        sys.exit(1)

    model = AccentClassifier(config)
    model.build(num_classes=len(label_encoder.classes_))
    model.set_label_encoder(label_encoder)

    # 创建数据加载器
    logger.info("\n步骤 3: 创建数据加载器")
    logger.info("-" * 60)

    batch_size = args.batch_size or config.get('wav2vec', {}).get('batch_size', 8)

    train_dataset = AccentDataset(X_train, y_train, model.processor)
    val_dataset = AccentDataset(X_val, y_val, model.processor)
    test_dataset = AccentDataset(X_test, y_test, model.processor)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # 训练模型
    logger.info("\n步骤 4: 训练模型")
    logger.info("-" * 60)

    trainer = AccentTrainer(model.model, config, device, logger)
    history = trainer.train(train_loader, val_loader, epochs=args.epochs)

    # 评估模型
    logger.info("\n步骤 5: 评估模型")
    logger.info("-" * 60)

    eval_results = trainer.evaluate(test_loader)

    metrics_calc = MetricsCalculator()
    metrics_calc.print_metrics(
        eval_results['labels'],
        eval_results['predictions'],
        target_names=label_encoder.classes_
    )

    # 可视化
    logger.info("\n步骤 6: 生成可视化")
    logger.info("-" * 60)

    visualizer = Visualizer()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 混淆矩阵
    cm = metrics_calc.get_confusion_matrix(
        eval_results['labels'],
        eval_results['predictions']
    )
    visualizer.plot_confusion_matrix(
        cm,
        class_names=list(label_encoder.classes_),
        output_path=str(output_dir / "confusion_matrix.png"),
        title="方言口音识别混淆矩阵"
    )

    # 保存模型
    logger.info("\n步骤 7: 保存模型")
    logger.info("-" * 60)

    model.save(args.output_dir)

    logger.info("\n" + "=" * 60)
    logger.info("训练完成！")
    logger.info("=" * 60)
    logger.info(f"模型保存在: {args.output_dir}")
    logger.info(f"测试准确率: {eval_results['accuracy']:.4f}")


def inference_mode(args, config, logger):
    """推理模式"""
    logger.info("=" * 60)
    logger.info("Lesson 9: 方言口音识别器 - 推理模式")
    logger.info("=" * 60)

    # 加载模型
    logger.info("加载模型...")
    device = get_device(args.device)
    if device is None:
        logger.error("PyTorch 未安装")
        sys.exit(1)

    model = AccentClassifier(config)
    model.load(args.model_path, device)

    # 加载音频
    logger.info(f"加载音频: {args.audio_file}")
    waveform, sr = torchaudio.load(args.audio_file)

    # 重采样
    if sr != 16000:
        resampler = torchaudio.transforms.Resample(sr, 16000)
        waveform = resampler(waveform)

    # 转为单声道
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0)

    # 预测
    logger.info("开始识别...")
    predicted_class = model.predict(waveform)
    probs = model.predict_proba(waveform)

    # 输出结果
    accent_name = model.label_encoder.inverse_transform([predicted_class])[0]

    logger.info("\n" + "=" * 60)
    logger.info("识别结果")
    logger.info("=" * 60)
    logger.info(f"预测口音: {accent_name}")
    logger.info(f"置信度: {probs[predicted_class]:.4f}")
    logger.info("\n所有类别概率:")
    for i, accent in enumerate(model.label_encoder.classes_):
        logger.info(f"  {accent}: {probs[i]:.4f}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="Lesson 9: 方言口音识别器")

    parser.add_argument('--mode', type=str, choices=['train', 'inference'], required=True)
    parser.add_argument('--config', type=str, default='config.yaml')
    parser.add_argument('--device', type=str, default='auto')
    parser.add_argument('--output_dir', type=str, default='checkpoints/accent_classifier')

    # 训练参数
    parser.add_argument('--data_dir', type=str)
    parser.add_argument('--labels_file', type=str)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int)

    # 推理参数
    parser.add_argument('--model_path', type=str)
    parser.add_argument('--audio_file', type=str)

    args = parser.parse_args()

    # 加载配置
    config_loader = get_config_loader(args.config)
    config = config_loader.load()

    # 设置日志
    logging_config = config.get('logging', {})
    logging_config['log_dir'] = config.get('paths', {}).get('log_dir', 'logs')
    logger = setup_logger('lesson_09_accent', logging_config)

    try:
        if args.mode == 'train':
            if not args.labels_file:
                parser.error("训练模式需要 --labels_file 参数")
            train_mode(args, config, logger)
        elif args.mode == 'inference':
            if not args.model_path or not args.audio_file:
                parser.error("推理模式需要 --model_path 和 --audio_file 参数")
            inference_mode(args, config, logger)

    except Exception as e:
        logger.error(f"执行失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
