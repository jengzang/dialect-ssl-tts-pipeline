"""Lesson 6: LSTM 声调分类器

命令行接口脚本，用于训练和测试 LSTM 声调分类器。

使用示例:
    # 训练模式
    python scripts/lesson_06_lstm_tone.py \\
        --mode train \\
        --data_file material/lesson_6/vowel_with_tone.csv \\
        --model_path checkpoints/lstm_tone.pth \\
        --output_dir results/lesson_06 \\
        --epochs 50

    # 推理模式
    python scripts/lesson_06_lstm_tone.py \\
        --mode inference \\
        --model_path checkpoints/lstm_tone.pth \\
        --test_data material/lesson_6/vowel_with_tone.csv
"""

import sys
from pathlib import Path
import argparse
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from scipy import stats

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.utils import get_config_loader, setup_logger, get_device
from src.models.lstm_tone import LSTMToneClassifier, ToneLSTM
from src.training.lstm_trainer import LSTMTrainer
from src.evaluation.metrics import MetricsCalculator
from src.evaluation.visualizer import Visualizer


class ToneDataset(Dataset):
    """声调数据集

    用于 PyTorch DataLoader 的数据集类。
    """

    def __init__(self, features, labels, seq_lengths=None):
        """初始化数据集

        Args:
            features: 特征数组 (num_samples, max_seq_len, feature_dim)
            labels: 标签数组 (num_samples,)
            seq_lengths: 序列长度数组 (num_samples,)
        """
        self.features = torch.FloatTensor(features)
        self.labels = torch.LongTensor(labels)
        self.seq_lengths = seq_lengths if seq_lengths is not None else \
                          torch.LongTensor([len(f) for f in features])

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            'features': self.features[idx],
            'labels': self.labels[idx],
            'seq_lens': self.seq_lengths[idx]
        }


def collate_fn(batch):
    """数据整理函数

    处理变长序列的批次数据。

    Args:
        batch: 批次数据

    Returns:
        整理后的批次字典
    """
    features = torch.stack([item['features'] for item in batch])
    labels = torch.stack([item['labels'] for item in batch])
    seq_lens = torch.stack([item['seq_lens'] for item in batch])

    return {
        'features': features,
        'labels': labels,
        'seq_lens': seq_lens
    }


def load_data_from_csv(
    csv_file: str,
    test_size: float = 0.2,
    val_size: float = 0.1,
    random_state: int = 42,
    logger=None
):
    """从 CSV 加载数据

    Args:
        csv_file: CSV 文件路径
        test_size: 测试集比例
        val_size: 验证集比例
        random_state: 随机种子
        logger: 日志记录器

    Returns:
        (X_train, X_val, X_test, y_train, y_val, y_test,
         train_lengths, val_lengths, test_lengths, label_encoder)
    """
    if logger:
        logger.info(f"加载数据: {csv_file}")

    # 读取 CSV
    df = pd.read_csv(csv_file, na_values=['', ' ', 'nan', 'NaN', 'null', 'NA'])
    df.fillna(0, inplace=True)

    if logger:
        logger.info(f"数据集大小: {len(df)}")

    # 1. 提取基频点特征
    f0_point_cols = sorted(
        [col for col in df.columns if col.startswith('f0_point_')],
        key=lambda x: int(x.split('_')[-1])
    )

    # 2. 提取统计特征
    stat_cols = [
        'f0_mean', 'f0_std', 'f0_min', 'f0_max', 'f0_range',
        'f0_median', 'f0_skew', 'f0_kurtosis',
        'norm_f0_mean', 'norm_f0_std',
        'delta_mean', 'delta2_mean',
        'duration'
    ]

    # 检查缺失列
    missing_f0 = [col for col in f0_point_cols if col not in df.columns]
    missing_stat = [col for col in stat_cols if col not in df.columns]

    if missing_f0:
        raise ValueError(f"缺少基频点列: {missing_f0}")
    if missing_stat:
        raise ValueError(f"缺少统计特征列: {missing_stat}")

    # 3. 构建 3D 特征矩阵
    X_f0 = df[f0_point_cols].values.reshape(len(df), len(f0_point_cols), 1)
    X_stat = df[stat_cols].values.reshape(len(df), 1, len(stat_cols))
    X_stat_expanded = np.repeat(X_stat, len(f0_point_cols), axis=1)
    X = np.concatenate([X_f0, X_stat_expanded], axis=2)

    if logger:
        logger.info(f"特征形状: {X.shape}")

    # 4. 计算序列长度
    seq_lengths = df[f0_point_cols].notna().sum(axis=1).values

    # 5. 提取标签
    y = df['phoneme'].str.extract(r'(\d)')[0].astype(int).values - 1

    # 编码标签
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)

    if logger:
        logger.info(f"类别数量: {len(label_encoder.classes_)}")
        logger.info(f"类别: {label_encoder.classes_}")

    # 6. 划分数据集
    # 先划分训练集和测试集
    X_train, X_test, y_train, y_test, train_lengths, test_lengths = train_test_split(
        X, y, seq_lengths,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    # 再从训练集中划分验证集
    if val_size > 0:
        val_size_adjusted = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val, train_lengths, val_lengths = train_test_split(
            X_train, y_train, train_lengths,
            test_size=val_size_adjusted,
            random_state=random_state,
            stratify=y_train
        )
    else:
        X_val = y_val = val_lengths = None

    if logger:
        logger.info(f"训练集大小: {len(X_train)}")
        if X_val is not None:
            logger.info(f"验证集大小: {len(X_val)}")
        logger.info(f"测试集大小: {len(X_test)}")

    return (X_train, X_val, X_test, y_train, y_val, y_test,
            train_lengths, val_lengths, test_lengths, label_encoder)


def train_mode(args, config, logger):
    """训练模式

    Args:
        args: 命令行参数
        config: 配置字典
        logger: 日志记录器
    """
    logger.info("=" * 60)
    logger.info("Lesson 6: LSTM 声调分类器 - 训练模式")
    logger.info("=" * 60)

    # 1. 加载数据
    logger.info("\n步骤 1: 加载数据")
    logger.info("-" * 60)

    (X_train, X_val, X_test, y_train, y_val, y_test,
     train_lengths, val_lengths, test_lengths, label_encoder) = load_data_from_csv(
        args.data_file,
        test_size=args.test_size,
        val_size=args.val_size,
        random_state=config.get('training', {}).get('seed', 42),
        logger=logger
    )

    # 2. 创建数据加载器
    logger.info("\n步骤 2: 创建数据加载器")
    logger.info("-" * 60)

    batch_size = args.batch_size or config.get('lstm', {}).get('batch_size', 32)

    train_dataset = ToneDataset(X_train, y_train, train_lengths)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn
    )

    if X_val is not None:
        val_dataset = ToneDataset(X_val, y_val, val_lengths)
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_fn
        )
    else:
        val_loader = None

    test_dataset = ToneDataset(X_test, y_test, test_lengths)
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn
    )

    logger.info(f"批次大小: {batch_size}")
    logger.info(f"训练批次数: {len(train_loader)}")
    if val_loader:
        logger.info(f"验证批次数: {len(val_loader)}")
    logger.info(f"测试批次数: {len(test_loader)}")

    # 3. 构建模型
    logger.info("\n步骤 3: 构建模型")
    logger.info("-" * 60)

    device = get_device(args.device)
    if device is None:
        logger.error("PyTorch 未安装，无法训练 LSTM 模型")
        sys.exit(1)

    input_dim = X_train.shape[2]
    num_classes = len(label_encoder.classes_)

    model = ToneLSTM(
        input_dim=input_dim,
        hidden_dim=config.get('lstm', {}).get('hidden_size', 128),
        num_layers=config.get('lstm', {}).get('num_layers', 2),
        num_classes=num_classes,
        dropout=config.get('lstm', {}).get('dropout', 0.3),
        bidirectional=True
    )

    logger.info(f"模型参数数量: {sum(p.numel() for p in model.parameters())}")

    # 4. 训练模型
    logger.info("\n步骤 4: 训练模型")
    logger.info("-" * 60)

    trainer = LSTMTrainer(model, config, device, logger)
    history = trainer.train(
        train_loader,
        val_loader,
        epochs=args.epochs
    )

    # 5. 评估模型
    logger.info("\n步骤 5: 评估模型")
    logger.info("-" * 60)

    eval_results = trainer.evaluate(test_loader)

    metrics_calc = MetricsCalculator()
    metrics_calc.print_metrics(
        np.array(eval_results['labels']),
        np.array(eval_results['predictions']),
        target_names=[str(c) for c in label_encoder.classes_]
    )

    # 6. 可视化
    logger.info("\n步骤 6: 生成可视化")
    logger.info("-" * 60)

    visualizer = Visualizer()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 混淆矩阵
    cm = metrics_calc.get_confusion_matrix(
        np.array(eval_results['labels']),
        np.array(eval_results['predictions'])
    )
    visualizer.plot_confusion_matrix(
        cm,
        class_names=[str(c) for c in label_encoder.classes_],
        output_path=str(output_dir / "confusion_matrix.png"),
        title="LSTM 声调分类混淆矩阵"
    )

    # 训练曲线
    plot_training_curves(history, str(output_dir / "training_curves.png"))

    # 7. 保存模型
    logger.info("\n步骤 7: 保存模型")
    logger.info("-" * 60)

    model_path = Path(args.model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)

    trainer.save_checkpoint(
        str(model_path),
        label_encoder=label_encoder,
        input_dim=input_dim,
        num_classes=num_classes
    )

    logger.info("\n" + "=" * 60)
    logger.info("训练完成！")
    logger.info("=" * 60)
    logger.info(f"模型保存在: {model_path}")
    logger.info(f"结果保存在: {args.output_dir}")
    logger.info(f"最终测试准确率: {eval_results['accuracy']:.4f}")


def plot_training_curves(history, output_path):
    """绘制训练曲线

    Args:
        history: 训练历史
        output_path: 输出路径
    """
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    # 损失曲线
    ax1.plot(history['train_loss'], label='Train Loss')
    if 'val_loss' in history and history['val_loss']:
        ax1.plot(history['val_loss'], label='Val Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True)

    # 准确率曲线
    ax2.plot(history['train_acc'], label='Train Acc')
    if 'val_acc' in history and history['val_acc']:
        ax2.plot(history['val_acc'], label='Val Acc')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Training and Validation Accuracy')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def inference_mode(args, config, logger):
    """推理模式

    Args:
        args: 命令行参数
        config: 配置字典
        logger: 日志记录器
    """
    logger.info("=" * 60)
    logger.info("Lesson 6: LSTM 声调分类器 - 推理模式")
    logger.info("=" * 60)

    # 加载模型
    logger.info("加载模型...")
    device = get_device(args.device)
    if device is None:
        logger.error("PyTorch 未安装")
        sys.exit(1)

    checkpoint = torch.load(args.model_path, map_location=device)

    # 重建模型
    model = ToneLSTM(
        input_dim=checkpoint['input_dim'],
        hidden_dim=config.get('lstm', {}).get('hidden_size', 128),
        num_layers=config.get('lstm', {}).get('num_layers', 2),
        num_classes=checkpoint['num_classes'],
        dropout=config.get('lstm', {}).get('dropout', 0.3),
        bidirectional=True
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    label_encoder = checkpoint['label_encoder']

    logger.info("模型加载完成")

    # 加载测试数据
    logger.info(f"加载测试数据: {args.test_data}")

    # 这里简化处理，实际应该使用完整的数据加载流程
    logger.info("推理功能待完善")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="Lesson 6: LSTM 声调分类器"
    )

    # 通用参数
    parser.add_argument(
        '--mode',
        type=str,
        choices=['train', 'inference'],
        required=True,
        help='运行模式'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='config.yaml',
        help='配置文件路径'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='auto',
        help='计算设备 (auto, cpu, cuda)'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='results/lesson_06',
        help='输出目录'
    )

    # 训练模式参数
    parser.add_argument(
        '--data_file',
        type=str,
        help='训练数据 CSV 文件'
    )
    parser.add_argument(
        '--model_path',
        type=str,
        default='checkpoints/lstm_tone.pth',
        help='模型保存/加载路径'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        help='训练轮数'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        help='批次大小'
    )
    parser.add_argument(
        '--test_size',
        type=float,
        default=0.2,
        help='测试集比例'
    )
    parser.add_argument(
        '--val_size',
        type=float,
        default=0.1,
        help='验证集比例'
    )

    # 推理模式参数
    parser.add_argument(
        '--test_data',
        type=str,
        help='测试数据文件'
    )

    args = parser.parse_args()

    # 加载配置
    config_loader = get_config_loader(args.config)
    config = config_loader.load()

    # 设置日志
    logging_config = config.get('logging', {})
    logging_config['log_dir'] = config.get('paths', {}).get('log_dir', 'logs')
    logger = setup_logger('lesson_06_lstm', logging_config)

    try:
        if args.mode == 'train':
            if not args.data_file:
                parser.error("训练模式需要 --data_file 参数")
            train_mode(args, config, logger)
        elif args.mode == 'inference':
            if not args.test_data:
                parser.error("推理模式需要 --test_data 参数")
            inference_mode(args, config, logger)

    except Exception as e:
        logger.error(f"执行失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
