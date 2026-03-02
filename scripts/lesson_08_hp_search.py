"""
Lesson 8 超参数搜索脚本

使用 Optuna 进行 LoRA 超参数优化。

使用示例：
    # 运行超参数搜索（模拟模式）
    python scripts/lesson_08_hp_search.py \
        --mode simulate \
        --n_trials 50 \
        --study_name dialect_translation_hp \
        --output_dir results/hp_search

    # 运行实际训练的超参数搜索
    python scripts/lesson_08_hp_search.py \
        --mode train \
        --n_trials 20 \
        --train_data data/dialect_translation/train.json \
        --val_data data/dialect_translation/val.json \
        --study_name dialect_translation_hp_real \
        --output_dir results/hp_search_real
"""

import argparse
import logging
import sys
from pathlib import Path
import json

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.logger import setup_logger
from src.training.hyperparameter_search import (
    LoRAHyperparameterSearch,
    create_mock_train_function
)
from src.evaluation.lora_analysis import analyze_lora_efficiency


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description='LoRA Hyperparameter Search'
    )

    parser.add_argument(
        '--mode',
        type=str,
        required=True,
        choices=['simulate', 'train', 'analyze'],
        help='运行模式'
    )

    parser.add_argument(
        '--study_name',
        type=str,
        default='dialect_translation_hp',
        help='Optuna 研究名称'
    )

    parser.add_argument(
        '--n_trials',
        type=int,
        default=50,
        help='试验次数'
    )

    parser.add_argument(
        '--timeout',
        type=int,
        help='超时时间（秒）'
    )

    parser.add_argument(
        '--output_dir',
        type=str,
        default='results/hp_search',
        help='输出目录'
    )

    parser.add_argument(
        '--train_data',
        type=str,
        help='训练数据路径（train 模式）'
    )

    parser.add_argument(
        '--val_data',
        type=str,
        help='验证数据路径（train 模式）'
    )

    parser.add_argument(
        '--model_name',
        type=str,
        default='uer/gpt2-chinese-cluecorpussmall',
        help='基础模型名称'
    )

    parser.add_argument(
        '--direction',
        type=str,
        default='maximize',
        choices=['maximize', 'minimize'],
        help='优化方向'
    )

    return parser.parse_args()


def simulate_mode(args, logger):
    """模拟模式"""
    logger.info("=== Simulate Mode ===")
    logger.info("Running hyperparameter search with mock training function")

    # 创建超参数搜索器
    searcher = LoRAHyperparameterSearch(
        study_name=args.study_name,
        direction=args.direction
    )

    # 创建模拟训练函数
    train_fn = create_mock_train_function()

    # 执行优化
    logger.info(f"Starting optimization: {args.n_trials} trials")
    searcher.optimize(
        train_fn=train_fn,
        n_trials=args.n_trials,
        timeout=args.timeout
    )

    # 保存结果
    logger.info("Saving results...")
    searcher.save_results(args.output_dir)

    # 获取最佳参数
    best_params = searcher.get_best_params()
    logger.info("\n=== Best Parameters ===")
    for key, value in best_params.items():
        logger.info(f"{key}: {value}")

    # 分析效率
    logger.info("\nAnalyzing efficiency...")
    trials_df = searcher.study.trials_dataframe()

    if len(trials_df) > 0:
        configs = []
        scores = []

        for _, row in trials_df.iterrows():
            if row['state'] == 'COMPLETE':
                config = {
                    'lora_r': row['params_lora_r'],
                    'lora_alpha': row['params_lora_alpha'],
                    'learning_rate': row['params_learning_rate'],
                    'batch_size': row['params_batch_size'],
                    'lora_dropout': row['params_lora_dropout'],
                    'target_modules': row['params_target_modules']
                }
                configs.append(config)
                scores.append(row['value'])

        if configs:
            efficiency_results = analyze_lora_efficiency(
                configs,
                scores,
                args.output_dir
            )

            logger.info("\n=== Efficiency Analysis ===")
            logger.info(f"Best score: {efficiency_results['best_score']:.4f}")
            logger.info(f"Most efficient score: {efficiency_results['most_efficient_score']:.4f}")
            logger.info(f"Pareto front size: {efficiency_results['pareto_front_size']}")

    logger.info(f"\n[OK] Results saved to: {args.output_dir}")
    return True


def train_mode(args, logger):
    """训练模式"""
    logger.info("=== Train Mode ===")
    logger.info("Running hyperparameter search with actual training")

    if not args.train_data:
        logger.error("--train_data is required for train mode")
        return False

    # TODO: 实现实际训练的超参数搜索
    # 这需要集成 dialect_translation_trainer

    logger.warning("Train mode not yet implemented")
    logger.info("Please use simulate mode for now")

    return False


def analyze_mode(args, logger):
    """分析模式"""
    logger.info("=== Analyze Mode ===")
    logger.info("Analyzing existing hyperparameter search results")

    # 加载已有的研究
    try:
        searcher = LoRAHyperparameterSearch(
            study_name=args.study_name,
            direction=args.direction,
            load_if_exists=True
        )

        # 打印结果
        searcher._print_results()

        # 保存结果
        searcher.save_results(args.output_dir)

        logger.info(f"\n[OK] Analysis saved to: {args.output_dir}")
        return True

    except Exception as e:
        logger.error(f"Failed to analyze: {e}")
        return False


def main():
    """主函数"""
    args = parse_args()

    # 设置日志
    log_config = {
        'level': 'INFO',
        'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        'log_dir': 'logs'
    }
    logger = setup_logger('hp_search', log_config)

    logger.info("Starting LoRA Hyperparameter Search")
    logger.info(f"Mode: {args.mode}")
    logger.info(f"Study name: {args.study_name}")

    # 根据模式执行
    if args.mode == 'simulate':
        success = simulate_mode(args, logger)
    elif args.mode == 'train':
        success = train_mode(args, logger)
    elif args.mode == 'analyze':
        success = analyze_mode(args, logger)
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
