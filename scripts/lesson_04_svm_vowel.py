"""Lesson 4: SVM 元音分类器

命令行接口脚本，用于训练和测试 SVM 元音分类器。

使用示例:
    # 训练模式
    python scripts/lesson_04_svm_vowel.py \\
        --mode train \\
        --audio_dir material/lesson_4/cantonese_v2 \\
        --textgrid_dir material/lesson_4/cantonese_v2_out_TG \\
        --target_vowels a e i o u \\
        --output_dir results/lesson_04

    # 推理模式
    python scripts/lesson_04_svm_vowel.py \\
        --mode inference \\
        --model_path checkpoints/svm_vowel.pkl \\
        --test_data data/lesson_04_test.csv
"""

import sys
from pathlib import Path
import argparse
import pandas as pd

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.utils import get_config_loader, setup_logger
from src.data_pipeline.feature_extractor import PraatFeatureExtractor
from src.data_pipeline.dataset_builder import SVMDatasetBuilder
from src.models.svm_classifier import SVMClassifier
from src.evaluation.metrics import MetricsCalculator
from src.evaluation.visualizer import Visualizer


def train_mode(args, config, logger):
    """训练模式

    Args:
        args: 命令行参数
        config: 配置字典
        logger: 日志记录器
    """
    logger.info("=" * 60)
    logger.info("Lesson 4: SVM 元音分类器 - 训练模式")
    logger.info("=" * 60)

    # 1. 特征提取
    logger.info("\n步骤 1: 提取特征")
    logger.info("-" * 60)

    extractor = PraatFeatureExtractor(config)

    # 提取特征并保存到 CSV
    features_file = Path(args.output_dir) / "features.csv"
    df = extractor.extract_batch(
        audio_dir=args.audio_dir,
        textgrid_dir=args.textgrid_dir,
        output_file=str(features_file),
        target_phonemes=args.target_vowels,
        audio_ext=args.audio_ext
    )

    logger.info(f"特征提取完成，共 {len(df)} 条记录")

    # 2. 构建数据集
    logger.info("\n步骤 2: 构建数据集")
    logger.info("-" * 60)

    # 定义特征列
    feature_columns = ['f1', 'f2', 'f3', 'duration']
    label_column = 'phoneme'

    dataset_builder = SVMDatasetBuilder(config)
    X_train, X_test, y_train, y_test = dataset_builder.build(
        data=df,
        feature_columns=feature_columns,
        label_column=label_column,
        test_size=config.get('svm', {}).get('test_size', 0.25),
        random_state=config.get('svm', {}).get('random_state', 42)
    )

    # 保存数据集
    dataset_dir = Path(args.output_dir) / "dataset"
    dataset_builder.save_dataset(
        (X_train, X_test, y_train, y_test),
        str(dataset_dir)
    )

    # 3. 训练模型
    logger.info("\n步骤 3: 训练 SVM 模型")
    logger.info("-" * 60)

    classifier = SVMClassifier(config)
    classifier.set_preprocessors(
        dataset_builder.scaler,
        dataset_builder.label_encoder
    )
    classifier.build()
    classifier.train(X_train, y_train)

    # 4. 评估模型
    logger.info("\n步骤 4: 评估模型")
    logger.info("-" * 60)

    metrics_calc = MetricsCalculator()

    # 预测
    y_pred = classifier.predict(X_test)

    # 打印指标
    metrics_calc.print_metrics(
        y_test, y_pred,
        target_names=dataset_builder.label_encoder.classes_
    )

    # 5. 可视化
    logger.info("\n步骤 5: 生成可视化")
    logger.info("-" * 60)

    visualizer = Visualizer()

    # 混淆矩阵
    cm = metrics_calc.get_confusion_matrix(y_test, y_pred)
    visualizer.plot_confusion_matrix(
        cm,
        class_names=list(dataset_builder.label_encoder.classes_),
        output_path=str(Path(args.output_dir) / "confusion_matrix.png"),
        title="SVM 元音分类混淆矩阵"
    )

    # t-SNE 可视化
    visualizer.plot_tsne(
        X_test, y_test,
        class_names=list(dataset_builder.label_encoder.classes_),
        output_path=str(Path(args.output_dir) / "tsne.png"),
        title="SVM 元音分类 t-SNE 可视化"
    )

    # 特征分布
    visualizer.plot_feature_distribution(
        X_test, y_test,
        feature_names=feature_columns,
        class_names=list(dataset_builder.label_encoder.classes_),
        output_path=str(Path(args.output_dir) / "feature_distribution.png")
    )

    # 6. 保存模型
    logger.info("\n步骤 6: 保存模型")
    logger.info("-" * 60)

    model_path = Path(args.model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    classifier.save(str(model_path))

    logger.info("\n" + "=" * 60)
    logger.info("训练完成！")
    logger.info("=" * 60)
    logger.info(f"模型保存在: {model_path}")
    logger.info(f"结果保存在: {args.output_dir}")


def inference_mode(args, config, logger):
    """推理模式

    Args:
        args: 命令行参数
        config: 配置字典
        logger: 日志记录器
    """
    logger.info("=" * 60)
    logger.info("Lesson 4: SVM 元音分类器 - 推理模式")
    logger.info("=" * 60)

    # 加载模型
    logger.info("加载模型...")
    classifier = SVMClassifier(config)
    classifier.load(args.model_path)

    # 加载测试数据
    logger.info(f"加载测试数据: {args.test_data}")
    df = pd.read_csv(args.test_data)

    # 提取特征
    feature_columns = ['f1', 'f2', 'f3', 'duration']
    X = df[feature_columns].values

    # 预测
    logger.info("开始预测...")
    y_pred = classifier.predict(X)

    # 如果有标签，计算准确率
    if 'phoneme' in df.columns:
        y_true_labels = df['phoneme'].values
        y_true = classifier.label_encoder.transform(y_true_labels)

        metrics_calc = MetricsCalculator()
        metrics_calc.print_metrics(
            y_true, y_pred,
            target_names=classifier.label_encoder.classes_
        )

    # 保存预测结果
    df['predicted_phoneme'] = classifier.label_encoder.inverse_transform(y_pred)
    output_file = Path(args.output_dir) / "predictions.csv"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_file, index=False)

    logger.info(f"预测结果已保存到: {output_file}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="Lesson 4: SVM 元音分类器"
    )

    # 通用参数
    parser.add_argument(
        '--mode',
        type=str,
        choices=['train', 'inference'],
        required=True,
        help='运行模式: train 或 inference'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='config.yaml',
        help='配置文件路径'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='results/lesson_04',
        help='输出目录'
    )

    # 训练模式参数
    parser.add_argument(
        '--audio_dir',
        type=str,
        help='音频文件目录'
    )
    parser.add_argument(
        '--textgrid_dir',
        type=str,
        help='TextGrid 文件目录'
    )
    parser.add_argument(
        '--target_vowels',
        nargs='+',
        help='目标元音列表'
    )
    parser.add_argument(
        '--audio_ext',
        type=str,
        default='.mp3',
        help='音频文件扩展名'
    )
    parser.add_argument(
        '--model_path',
        type=str,
        default='checkpoints/svm_vowel.pkl',
        help='模型保存/加载路径'
    )

    # 推理模式参数
    parser.add_argument(
        '--test_data',
        type=str,
        help='测试数据 CSV 文件路径'
    )

    args = parser.parse_args()

    # 加载配置
    config_loader = get_config_loader(args.config)
    config = config_loader.load()

    # 设置日志
    logging_config = config.get('logging', {})
    logging_config['log_dir'] = config.get('paths', {}).get('log_dir', 'logs')
    logger = setup_logger('lesson_04_svm', logging_config)

    try:
        if args.mode == 'train':
            if not args.audio_dir or not args.textgrid_dir:
                parser.error("训练模式需要 --audio_dir 和 --textgrid_dir 参数")
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
