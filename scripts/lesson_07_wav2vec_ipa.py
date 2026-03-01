"""Lesson 7: wav2vec 2.0 IPA 识别器

命令行接口脚本，用于训练和测试 wav2vec 2.0 IPA 识别模型。

使用示例:
    # 训练模式
    python scripts/lesson_07_wav2vec_ipa.py \\
        --mode train \\
        --data_dir material/lesson_7/data \\
        --model_name facebook/wav2vec2-base \\
        --output_dir checkpoints/wav2vec_ipa \\
        --epochs 10

    # 推理模式
    python scripts/lesson_07_wav2vec_ipa.py \\
        --mode inference \\
        --model_path checkpoints/wav2vec_ipa \\
        --audio_file test.wav
"""

import sys
from pathlib import Path
import argparse

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.utils import get_config_loader, setup_logger, get_device


def check_dependencies():
    """检查依赖是否安装"""
    try:
        import transformers
        import datasets
        import evaluate
        return True
    except ImportError as e:
        print(f"错误: 缺少必需的依赖库")
        print(f"请安装: pip install transformers datasets evaluate")
        print(f"详细错误: {e}")
        return False


def train_mode(args, config, logger):
    """训练模式

    Args:
        args: 命令行参数
        config: 配置字典
        logger: 日志记录器
    """
    logger.info("=" * 60)
    logger.info("Lesson 7: wav2vec 2.0 IPA 识别器 - 训练模式")
    logger.info("=" * 60)

    # 导入必需的库
    from src.data_pipeline.wav2vec_dataset import Wav2VecDatasetBuilder, DataCollatorCTCWithPadding
    from src.models.wav2vec_ipa import Wav2VecIPAModel
    from src.training.wav2vec_trainer import Wav2VecTrainer, prepare_dataset

    # 1. 构建数据集
    logger.info("\n步骤 1: 构建数据集")
    logger.info("-" * 60)

    dataset_builder = Wav2VecDatasetBuilder(config)

    # 从 CSV 或目录加载数据
    if args.csv_file:
        dataset_dict = dataset_builder.build_from_csv(
            args.csv_file,
            audio_column=args.audio_column,
            text_column=args.text_column
        )
    elif args.data_dir and args.transcript_file:
        dataset_dict = dataset_builder.build_from_directory(
            args.data_dir,
            args.transcript_file
        )
    else:
        logger.error("请提供 --csv_file 或 (--data_dir 和 --transcript_file)")
        sys.exit(1)

    # 2. 创建词汇表
    logger.info("\n步骤 2: 创建词汇表")
    logger.info("-" * 60)

    vocab = Wav2VecIPAModel.create_vocab_from_dataset(
        dataset_dict['train'],
        text_column='text'
    )

    logger.info(f"词汇表大小: {len(vocab)}")
    logger.info(f"词汇表: {vocab[:20]}...")  # 显示前20个

    # 3. 构建模型
    logger.info("\n步骤 3: 构建模型")
    logger.info("-" * 60)

    model = Wav2VecIPAModel(config)
    model.build(
        vocab=vocab,
        freeze_feature_encoder=config.get('wav2vec', {}).get('freeze_feature_encoder', True)
    )

    # 4. 准备数据集
    logger.info("\n步骤 4: 准备数据集")
    logger.info("-" * 60)

    # 应用预处理
    dataset_dict = dataset_dict.map(
        lambda batch: prepare_dataset(batch, model.processor),
        remove_columns=dataset_dict["train"].column_names,
        num_proc=1
    )

    # 5. 创建数据整理器
    data_collator = DataCollatorCTCWithPadding(processor=model.processor, padding=True)

    # 6. 训练模型
    logger.info("\n步骤 6: 训练模型")
    logger.info("-" * 60)

    trainer = Wav2VecTrainer(
        model=model.model,
        config=config,
        processor=model.processor,
        data_collator=data_collator,
        logger=logger
    )

    history = trainer.train(
        train_dataset=dataset_dict['train'],
        val_dataset=dataset_dict['validation'],
        output_dir=args.output_dir
    )

    # 7. 评估模型
    logger.info("\n步骤 7: 评估模型")
    logger.info("-" * 60)

    metrics = trainer.evaluate(dataset_dict['test'])

    logger.info("\n" + "=" * 60)
    logger.info("训练完成！")
    logger.info("=" * 60)
    logger.info(f"模型保存在: {args.output_dir}")
    logger.info(f"测试 WER: {metrics.get('eval_wer', 'N/A'):.4f}")


def inference_mode(args, config, logger):
    """推理模式

    Args:
        args: 命令行参数
        config: 配置字典
        logger: 日志记录器
    """
    logger.info("=" * 60)
    logger.info("Lesson 7: wav2vec 2.0 IPA 识别器 - 推理模式")
    logger.info("=" * 60)

    from src.models.wav2vec_ipa import Wav2VecIPAModel
    import torchaudio

    # 加载模型
    logger.info("加载模型...")
    model = Wav2VecIPAModel(config)
    model.load(args.model_path)

    # 加载音频
    logger.info(f"加载音频: {args.audio_file}")
    waveform, sample_rate = torchaudio.load(args.audio_file)

    # 重采样到 16kHz
    if sample_rate != 16000:
        resampler = torchaudio.transforms.Resample(sample_rate, 16000)
        waveform = resampler(waveform)

    # 预测
    logger.info("开始识别...")
    transcription = model.predict(waveform.squeeze())

    logger.info("\n" + "=" * 60)
    logger.info("识别结果")
    logger.info("=" * 60)
    logger.info(f"转录: {transcription}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="Lesson 7: wav2vec 2.0 IPA 识别器"
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
        '--output_dir',
        type=str,
        default='checkpoints/wav2vec_ipa',
        help='输出目录'
    )

    # 训练模式参数
    parser.add_argument(
        '--csv_file',
        type=str,
        help='训练数据 CSV 文件'
    )
    parser.add_argument(
        '--data_dir',
        type=str,
        help='音频文件目录'
    )
    parser.add_argument(
        '--transcript_file',
        type=str,
        help='转录文件'
    )
    parser.add_argument(
        '--audio_column',
        type=str,
        default='audio_path',
        help='CSV 中的音频路径列名'
    )
    parser.add_argument(
        '--text_column',
        type=str,
        default='text',
        help='CSV 中的文本列名'
    )
    parser.add_argument(
        '--model_name',
        type=str,
        default='facebook/wav2vec2-base',
        help='预训练模型名称'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        help='训练轮数'
    )

    # 推理模式参数
    parser.add_argument(
        '--model_path',
        type=str,
        help='模型路径'
    )
    parser.add_argument(
        '--audio_file',
        type=str,
        help='音频文件路径'
    )

    args = parser.parse_args()

    # 检查依赖
    if not check_dependencies():
        sys.exit(1)

    # 加载配置
    config_loader = get_config_loader(args.config)
    config = config_loader.load()

    # 更新配置
    if args.model_name:
        config.setdefault('wav2vec', {})['model_name'] = args.model_name
    if args.epochs:
        config.setdefault('wav2vec', {})['epochs'] = args.epochs

    # 设置日志
    logging_config = config.get('logging', {})
    logging_config['log_dir'] = config.get('paths', {}).get('log_dir', 'logs')
    logger = setup_logger('lesson_07_wav2vec', logging_config)

    try:
        if args.mode == 'train':
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
