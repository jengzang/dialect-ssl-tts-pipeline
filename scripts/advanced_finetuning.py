#!/usr/bin/env python3
"""
高级微调方法比较工具

比较不同参数高效微调方法的性能和效率
"""

import argparse
import logging
from pathlib import Path
import sys
import json
import time
from typing import Dict, List, Any

import torch
from transformers import GPT2Config, GPT2LMHeadModel

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.training.advanced_trainer import AdvancedFinetuner

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def compare_methods(
    model_name: str,
    methods: List[str],
    device: str = "cuda",
    output_dir: str = "results/advanced_finetuning"
) -> Dict[str, Any]:
    """
    比较不同微调方法

    Args:
        model_name: 模型名称
        methods: 微调方法列表
        device: 设备
        output_dir: 输出目录

    Returns:
        比较结果
    """
    logger.info(f"Comparing methods: {methods}")
    logger.info(f"Model: {model_name}")

    results = {}

    # 方法配置
    method_configs = {
        "lora": {"r": 8, "alpha": 16, "dropout": 0.1},
        "lora_r4": {"r": 4, "alpha": 8, "dropout": 0.1},
        "lora_r16": {"r": 16, "alpha": 32, "dropout": 0.1},
        "prefix": {"prefix_length": 10, "prefix_hidden_size": 512},
        "adapter": {"adapter_size": 64, "adapter_activation": "gelu"},
        "full": {}
    }

    for method in methods:
        logger.info(f"\n{'='*60}")
        logger.info(f"Testing method: {method}")
        logger.info(f"{'='*60}")

        try:
            # 创建训练器
            config = method_configs.get(method, {})
            trainer = AdvancedFinetuner(
                model_name=model_name,
                method=method.split("_")[0],  # 去掉后缀（如 lora_r4 -> lora）
                method_config=config,
                device=device,
                mixed_precision=True
            )

            # 获取统计信息
            stats = trainer.get_stats()

            # 基准测试推理速度
            logger.info("Benchmarking inference speed...")
            inference_time = benchmark_inference(trainer.model, device)
            stats["inference_time_ms"] = inference_time

            # 计算效率指标
            stats["param_efficiency"] = stats["trainable_params"] / stats["total_params"]
            stats["memory_efficiency_mb"] = stats["peak_memory_mb"] if stats["peak_memory_mb"] > 0 else estimate_memory(stats["trainable_params"])

            results[method] = stats

            logger.info(f"Results for {method}:")
            logger.info(f"  Trainable params: {stats['trainable_params']:,}")
            logger.info(f"  Trainable ratio: {stats['param_efficiency']*100:.2f}%")
            logger.info(f"  Inference time: {stats['inference_time_ms']:.2f} ms")

        except Exception as e:
            logger.error(f"Error testing {method}: {e}", exc_info=True)
            results[method] = {"error": str(e)}

    # 保存结果
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    with open(output_path / "comparison_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    logger.info(f"\nResults saved to {output_path / 'comparison_results.json'}")

    # 生成比较报告
    generate_comparison_report(results, output_path)

    return results


def benchmark_inference(model: torch.nn.Module, device: str, num_runs: int = 10) -> float:
    """
    基准测试推理速度

    Args:
        model: 模型
        device: 设备
        num_runs: 运行次数

    Returns:
        平均推理时间（毫秒）
    """
    model.eval()

    # 准备输入
    input_ids = torch.randint(0, 1000, (1, 50), device=device)

    # 预热
    with torch.no_grad():
        for _ in range(3):
            _ = model(input_ids)

    # 基准测试
    if device == "cuda":
        torch.cuda.synchronize()

    start_time = time.time()

    with torch.no_grad():
        for _ in range(num_runs):
            _ = model(input_ids)

    if device == "cuda":
        torch.cuda.synchronize()

    elapsed_time = (time.time() - start_time) / num_runs * 1000  # 转换为毫秒

    return elapsed_time


def estimate_memory(num_params: int) -> float:
    """
    估算内存使用（MB）

    Args:
        num_params: 参数数量

    Returns:
        估算的内存使用（MB）
    """
    # 假设 FP16，每个参数 2 字节
    bytes_per_param = 2
    memory_bytes = num_params * bytes_per_param
    memory_mb = memory_bytes / (1024 * 1024)
    return memory_mb


def generate_comparison_report(results: Dict[str, Any], output_dir: Path):
    """
    生成比较报告

    Args:
        results: 比较结果
        output_dir: 输出目录
    """
    logger.info("\n" + "="*80)
    logger.info("COMPARISON REPORT")
    logger.info("="*80)

    # 表头
    logger.info(f"\n{'Method':<15} {'Trainable':<15} {'Ratio':<10} {'Inference':<15} {'Memory':<15}")
    logger.info("-" * 80)

    # 排序：按可训练参数数量
    sorted_methods = sorted(
        [(k, v) for k, v in results.items() if "error" not in v],
        key=lambda x: x[1]["trainable_params"]
    )

    for method, stats in sorted_methods:
        trainable = f"{stats['trainable_params']:,}"
        ratio = f"{stats['param_efficiency']*100:.2f}%"
        inference = f"{stats.get('inference_time_ms', 0):.2f} ms"
        memory = f"{stats.get('memory_efficiency_mb', 0):.1f} MB"

        logger.info(f"{method:<15} {trainable:<15} {ratio:<10} {inference:<15} {memory:<15}")

    # 找出最佳方法
    logger.info("\n" + "="*80)
    logger.info("BEST METHODS")
    logger.info("="*80)

    if sorted_methods:
        # 最少参数
        best_params = min(sorted_methods, key=lambda x: x[1]["trainable_params"])
        logger.info(f"Fewest parameters: {best_params[0]} ({best_params[1]['trainable_params']:,})")

        # 最快推理
        best_speed = min(sorted_methods, key=lambda x: x[1].get("inference_time_ms", float("inf")))
        logger.info(f"Fastest inference: {best_speed[0]} ({best_speed[1].get('inference_time_ms', 0):.2f} ms)")

        # 最佳效率（参数/性能比）
        best_efficiency = min(sorted_methods, key=lambda x: x[1]["param_efficiency"])
        logger.info(f"Best efficiency: {best_efficiency[0]} ({best_efficiency[1]['param_efficiency']*100:.2f}%)")

    # 保存文本报告
    with open(output_dir / "comparison_report.txt", "w", encoding="utf-8") as f:
        f.write("="*80 + "\n")
        f.write("ADVANCED FINETUNING METHODS COMPARISON\n")
        f.write("="*80 + "\n\n")

        f.write(f"{'Method':<15} {'Trainable':<15} {'Ratio':<10} {'Inference':<15} {'Memory':<15}\n")
        f.write("-" * 80 + "\n")

        for method, stats in sorted_methods:
            trainable = f"{stats['trainable_params']:,}"
            ratio = f"{stats['param_efficiency']*100:.2f}%"
            inference = f"{stats.get('inference_time_ms', 0):.2f} ms"
            memory = f"{stats.get('memory_efficiency_mb', 0):.1f} MB"

            f.write(f"{method:<15} {trainable:<15} {ratio:<10} {inference:<15} {memory:<15}\n")

    logger.info(f"\nText report saved to {output_dir / 'comparison_report.txt'}")


def create_custom_model(device: str = "cpu") -> GPT2LMHeadModel:
    """
    创建自定义小模型用于测试

    Args:
        device: 设备

    Returns:
        GPT2 模型
    """
    logger.info("Creating custom GPT-2 model for testing...")

    config = GPT2Config(
        vocab_size=5000,
        n_positions=512,
        n_embd=384,
        n_layer=6,
        n_head=6
    )

    model = GPT2LMHeadModel(config)
    model = model.to(device)

    logger.info(f"Model created: {sum(p.numel() for p in model.parameters()):,} parameters")

    return model


def main():
    parser = argparse.ArgumentParser(description="高级微调方法比较")

    parser.add_argument(
        "--model_name",
        type=str,
        default="custom",
        help="模型名称（custom 表示创建自定义小模型）"
    )

    parser.add_argument(
        "--methods",
        type=str,
        nargs="+",
        default=["lora", "lora_r4", "lora_r16", "prefix", "adapter"],
        help="要比较的方法列表"
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="设备"
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/advanced_finetuning",
        help="输出目录"
    )

    args = parser.parse_args()

    logger.info("="*80)
    logger.info("ADVANCED FINETUNING METHODS COMPARISON")
    logger.info("="*80)
    logger.info(f"Model: {args.model_name}")
    logger.info(f"Methods: {args.methods}")
    logger.info(f"Device: {args.device}")
    logger.info(f"Output: {args.output_dir}")

    # 如果使用自定义模型，先创建并保存
    if args.model_name == "custom":
        model = create_custom_model(args.device)
        # 注意：这里简化处理，实际应该保存模型
        logger.info("Using custom model (in-memory)")

    # 比较方法
    results = compare_methods(
        model_name=args.model_name,
        methods=args.methods,
        device=args.device,
        output_dir=args.output_dir
    )

    logger.info("\n" + "="*80)
    logger.info("COMPARISON COMPLETED")
    logger.info("="*80)


if __name__ == "__main__":
    main()
