"""
Lesson 10: 方言虚拟人搭建

整合 GPT-SoVITS 语音合成和 Sadtalker 虚拟人技术，
构建完整的方言虚拟人系统。

功能：
- 训练 TTS 模型
- 语音合成
- 创建虚拟人视频
- 批量生成虚拟人

使用示例：
    # 训练 TTS 模型
    python scripts/lesson_10_virtual_human.py --mode train \\
        --train_data_dir data/dialect_corpus \\
        --output_dir checkpoints/tts_model \\
        --epochs 10

    # 合成语音
    python scripts/lesson_10_virtual_human.py --mode synthesize \\
        --text "我今日好开心啊" \\
        --ref_audio data/ref.wav \\
        --ref_text "参考文本" \\
        --output results/synthesized.wav

    # 创建虚拟人
    python scripts/lesson_10_virtual_human.py --mode create_video \\
        --text "我今日好开心啊" \\
        --ref_audio data/ref.wav \\
        --ref_text "参考文本" \\
        --avatar_image data/avatar.jpg \\
        --output results/virtual_human.mp4

    # 批量创建虚拟人
    python scripts/lesson_10_virtual_human.py --mode batch \\
        --input_file data/texts.txt \\
        --ref_audio data/ref.wav \\
        --ref_text "参考文本" \\
        --avatar_image data/avatar.jpg \\
        --output_dir results/virtual_humans
"""

import argparse
import logging
from pathlib import Path
import sys

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.config_loader import ConfigLoader
from src.utils.logger import setup_logger
from src.training.virtual_human_trainer import DialectVirtualHumanTrainer


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description='Lesson 10: Dialect Virtual Human'
    )

    parser.add_argument(
        '--mode',
        type=str,
        required=True,
        choices=['train', 'synthesize', 'create_video', 'batch'],
        help='运行模式'
    )

    parser.add_argument(
        '--train_data_dir',
        type=str,
        help='训练数据目录（用于 train 模式）'
    )

    parser.add_argument(
        '--text',
        type=str,
        help='合成文本'
    )

    parser.add_argument(
        '--ref_audio',
        type=str,
        help='参考音频路径'
    )

    parser.add_argument(
        '--ref_text',
        type=str,
        help='参考文本'
    )

    parser.add_argument(
        '--avatar_image',
        type=str,
        help='虚拟人图像路径'
    )

    parser.add_argument(
        '--output',
        type=str,
        help='输出文件路径'
    )

    parser.add_argument(
        '--output_dir',
        type=str,
        default='results/virtual_human',
        help='输出目录'
    )

    parser.add_argument(
        '--input_file',
        type=str,
        help='输入文件（用于批量模式）'
    )

    parser.add_argument(
        '--epochs',
        type=int,
        default=10,
        help='训练轮数'
    )

    parser.add_argument(
        '--gpt_sovits_dir',
        type=str,
        default='./GPT-SoVITS',
        help='GPT-SoVITS 目录'
    )

    parser.add_argument(
        '--sadtalker_dir',
        type=str,
        default='./SadTalker',
        help='Sadtalker 目录'
    )

    parser.add_argument(
        '--gpt_model_path',
        type=str,
        help='GPT 模型路径'
    )

    parser.add_argument(
        '--sovits_model_path',
        type=str,
        help='SoVITS 模型路径'
    )

    parser.add_argument(
        '--config',
        type=str,
        default='config.yaml',
        help='配置文件路径'
    )

    return parser.parse_args()


def train_mode(args, logger):
    """训练模式"""
    logger.info("=== Training Mode ===")

    if not args.train_data_dir:
        logger.error("--train_data_dir is required")
        return False

    config = {
        'output_dir': args.output_dir,
        'gpt_sovits': {
            'gpt_sovits_dir': args.gpt_sovits_dir,
            'gpt_model_path': args.gpt_model_path,
            'sovits_model_path': args.sovits_model_path
        }
    }

    trainer = DialectVirtualHumanTrainer(config)

    # 训练 TTS 模型
    model_path = trainer.train_tts_model(
        train_data_dir=args.train_data_dir,
        epochs=args.epochs
    )

    if model_path:
        logger.info(f"[OK] Training completed: {model_path}")
        return True
    else:
        logger.error("[ERROR] Training failed")
        return False


def synthesize_mode(args, logger):
    """语音合成模式"""
    logger.info("=== Synthesize Mode ===")

    if not all([args.text, args.ref_audio, args.ref_text]):
        logger.error("--text, --ref_audio, --ref_text are required")
        return False

    config = {
        'output_dir': args.output_dir,
        'gpt_sovits': {
            'gpt_sovits_dir': args.gpt_sovits_dir,
            'gpt_model_path': args.gpt_model_path,
            'sovits_model_path': args.sovits_model_path
        }
    }

    trainer = DialectVirtualHumanTrainer(config)

    # 合成语音
    output_path = trainer.synthesize_speech(
        text=args.text,
        ref_audio=args.ref_audio,
        ref_text=args.ref_text,
        output_path=args.output
    )

    if output_path:
        logger.info(f"[OK] Speech synthesized: {output_path}")
        return True
    else:
        logger.error("[ERROR] Synthesis failed")
        return False


def create_video_mode(args, logger):
    """创建虚拟人视频模式"""
    logger.info("=== Create Video Mode ===")

    if not all([args.text, args.ref_audio, args.ref_text, args.avatar_image]):
        logger.error("--text, --ref_audio, --ref_text, --avatar_image are required")
        return False

    if not args.output:
        args.output = str(Path(args.output_dir) / 'virtual_human.mp4')

    config = {
        'output_dir': args.output_dir,
        'gpt_sovits': {
            'gpt_sovits_dir': args.gpt_sovits_dir,
            'gpt_model_path': args.gpt_model_path,
            'sovits_model_path': args.sovits_model_path
        },
        'sadtalker_dir': args.sadtalker_dir
    }

    trainer = DialectVirtualHumanTrainer(config)

    # 创建虚拟人
    success = trainer.create_dialect_virtual_human(
        dialect_text=args.text,
        ref_audio=args.ref_audio,
        ref_text=args.ref_text,
        avatar_image=args.avatar_image,
        output_video_path=args.output
    )

    if success:
        logger.info(f"[OK] Virtual human created: {args.output}")
        return True
    else:
        logger.error("[ERROR] Virtual human creation failed")
        return False


def batch_mode(args, logger):
    """批量创建模式"""
    logger.info("=== Batch Mode ===")

    if not all([args.input_file, args.ref_audio, args.ref_text, args.avatar_image]):
        logger.error("--input_file, --ref_audio, --ref_text, --avatar_image required")
        return False

    config = {
        'output_dir': args.output_dir,
        'gpt_sovits': {
            'gpt_sovits_dir': args.gpt_sovits_dir,
            'gpt_model_path': args.gpt_model_path,
            'sovits_model_path': args.sovits_model_path
        },
        'sadtalker_dir': args.sadtalker_dir
    }

    trainer = DialectVirtualHumanTrainer(config)

    # 读取输入文件
    try:
        with open(args.input_file, 'r', encoding='utf-8') as f:
            texts = [line.strip() for line in f if line.strip()]
    except Exception as e:
        logger.error(f"Failed to read input file: {e}")
        return False

    logger.info(f"Loaded {len(texts)} texts")

    # 批量创建
    results = trainer.batch_create_virtual_humans(
        texts=texts,
        ref_audio=args.ref_audio,
        ref_text=args.ref_text,
        avatar_image=args.avatar_image,
        output_dir=args.output_dir
    )

    success_count = sum(results)
    logger.info(f"[OK] Created {success_count} / {len(texts)} virtual humans")

    return success_count > 0


def main():
    """主函数"""
    args = parse_args()

    # 设置日志
    log_config = {
        'level': 'INFO',
        'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        'log_dir': 'logs'
    }
    logger = setup_logger('lesson_10', log_config)

    logger.info("Starting Lesson 10: Dialect Virtual Human")
    logger.info(f"Mode: {args.mode}")

    # 根据模式执行
    if args.mode == 'train':
        success = train_mode(args, logger)
    elif args.mode == 'synthesize':
        success = synthesize_mode(args, logger)
    elif args.mode == 'create_video':
        success = create_video_mode(args, logger)
    elif args.mode == 'batch':
        success = batch_mode(args, logger)
    else:
        logger.error(f"Unknown mode: {args.mode}")
        success = False

    if success:
        logger.info("=== Completed Successfully ===")
        sys.exit(0)
    else:
        logger.error("=== Failed ===")
        sys.exit(1)


if __name__ == '__main__':
    main()
