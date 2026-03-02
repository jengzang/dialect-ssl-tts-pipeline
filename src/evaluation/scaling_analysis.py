"""
缩放定律分析

分析模型大小、数据量与性能之间的关系。

分析内容：
1. 参数数量 vs. 性能
2. 训练数据量 vs. 性能
3. 计算量 vs. 性能
4. 推理效率分析
"""

import logging
from typing import Dict, Any, List, Optional, Tuple
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


class ScalingAnalyzer:
    """缩放定律分析器"""

    def __init__(self):
        """初始化分析器"""
        self.results = []

    def add_result(
        self,
        model_name: str,
        num_params: int,
        train_samples: int,
        performance: float,
        inference_time: float,
        memory_mb: float,
        **kwargs
    ):
        """
        添加实验结果

        Args:
            model_name: 模型名称
            num_params: 参数数量
            train_samples: 训练样本数
            performance: 性能指标（如 BLEU）
            inference_time: 推理时间（秒）
            memory_mb: 内存占用（MB）
            **kwargs: 其他指标
        """
        result = {
            "model_name": model_name,
            "num_params": num_params,
            "train_samples": train_samples,
            "performance": performance,
            "inference_time": inference_time,
            "memory_mb": memory_mb,
            **kwargs
        }

        self.results.append(result)
        logger.info(f"Added result for {model_name}: {num_params:,} params, perf={performance:.2f}")

    def analyze_param_scaling(self) -> Dict[str, Any]:
        """
        分析参数缩放定律

        Returns:
            分析结果
        """
        if not self.results:
            return {}

        # 按参数数量排序
        sorted_results = sorted(self.results, key=lambda x: x["num_params"])

        params = [r["num_params"] for r in sorted_results]
        performance = [r["performance"] for r in sorted_results]

        # 对数拟合
        log_params = np.log10(params)
        log_perf = np.log10(performance)

        # 线性拟合（对数空间）
        coeffs = np.polyfit(log_params, log_perf, 1)
        slope = coeffs[0]
        intercept = coeffs[1]

        # 计算 R²
        predicted = np.polyval(coeffs, log_params)
        ss_res = np.sum((log_perf - predicted) ** 2)
        ss_tot = np.sum((log_perf - np.mean(log_perf)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

        analysis = {
            "scaling_law": f"Performance ∝ Params^{slope:.3f}",
            "slope": slope,
            "intercept": intercept,
            "r_squared": r_squared,
            "data_points": len(params)
        }

        logger.info(f"Parameter scaling law: {analysis['scaling_law']}")
        logger.info(f"R² = {r_squared:.4f}")

        return analysis

    def analyze_efficiency(self) -> Dict[str, Any]:
        """
        分析效率指标

        Returns:
            效率分析
        """
        if not self.results:
            return {}

        efficiency_metrics = []

        for result in self.results:
            # 计算效率指标
            perf_per_param = result["performance"] / (result["num_params"] / 1e6)  # 每百万参数的性能
            perf_per_mb = result["performance"] / result["memory_mb"]  # 每 MB 的性能
            perf_per_second = result["performance"] / result["inference_time"]  # 每秒的性能

            efficiency_metrics.append({
                "model_name": result["model_name"],
                "perf_per_param": perf_per_param,
                "perf_per_mb": perf_per_mb,
                "perf_per_second": perf_per_second
            })

        # 找出最高效的模型
        best_param_eff = max(efficiency_metrics, key=lambda x: x["perf_per_param"])
        best_memory_eff = max(efficiency_metrics, key=lambda x: x["perf_per_mb"])
        best_speed_eff = max(efficiency_metrics, key=lambda x: x["perf_per_second"])

        analysis = {
            "efficiency_metrics": efficiency_metrics,
            "best_param_efficiency": best_param_eff,
            "best_memory_efficiency": best_memory_eff,
            "best_speed_efficiency": best_speed_eff
        }

        logger.info(f"Most parameter-efficient: {best_param_eff['model_name']}")
        logger.info(f"Most memory-efficient: {best_memory_eff['model_name']}")
        logger.info(f"Fastest: {best_speed_eff['model_name']}")

        return analysis

    def plot_scaling_curves(self, output_dir: str):
        """
        绘制缩放曲线

        Args:
            output_dir: 输出目录
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        if not self.results:
            logger.warning("No results to plot")
            return

        # 按参数数量排序
        sorted_results = sorted(self.results, key=lambda x: x["num_params"])

        params = [r["num_params"] / 1e6 for r in sorted_results]  # 转换为百万
        performance = [r["performance"] for r in sorted_results]
        memory = [r["memory_mb"] for r in sorted_results]
        inference_time = [r["inference_time"] * 1000 for r in sorted_results]  # 转换为毫秒
        model_names = [r["model_name"] for r in sorted_results]

        # 1. 参数 vs. 性能
        plt.figure(figsize=(10, 6))
        plt.plot(params, performance, 'o-', linewidth=2, markersize=8)
        for i, name in enumerate(model_names):
            plt.annotate(name, (params[i], performance[i]),
                        textcoords="offset points", xytext=(0,10), ha='center')
        plt.xlabel('Parameters (M)', fontsize=12)
        plt.ylabel('Performance (BLEU)', fontsize=12)
        plt.title('Scaling Law: Parameters vs. Performance', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / 'params_vs_performance.png', dpi=300)
        plt.close()

        # 2. 参数 vs. 内存
        plt.figure(figsize=(10, 6))
        plt.plot(params, memory, 's-', linewidth=2, markersize=8, color='orange')
        for i, name in enumerate(model_names):
            plt.annotate(name, (params[i], memory[i]),
                        textcoords="offset points", xytext=(0,10), ha='center')
        plt.xlabel('Parameters (M)', fontsize=12)
        plt.ylabel('Memory (MB)', fontsize=12)
        plt.title('Model Size vs. Memory Usage', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / 'params_vs_memory.png', dpi=300)
        plt.close()

        # 3. 参数 vs. 推理时间
        plt.figure(figsize=(10, 6))
        plt.plot(params, inference_time, '^-', linewidth=2, markersize=8, color='green')
        for i, name in enumerate(model_names):
            plt.annotate(name, (params[i], inference_time[i]),
                        textcoords="offset points", xytext=(0,10), ha='center')
        plt.xlabel('Parameters (M)', fontsize=12)
        plt.ylabel('Inference Time (ms)', fontsize=12)
        plt.title('Model Size vs. Inference Speed', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / 'params_vs_inference_time.png', dpi=300)
        plt.close()

        # 4. 效率对比（雷达图）
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))

        # 计算归一化的效率指标
        perf_per_param = [p / (pr / 1e6) for p, pr in zip(performance, [r["num_params"] for r in sorted_results])]
        perf_per_mb = [p / m for p, m in zip(performance, memory)]
        perf_per_ms = [p / t for p, t in zip(performance, inference_time)]

        # 归一化到 0-1
        def normalize(values):
            min_val, max_val = min(values), max(values)
            return [(v - min_val) / (max_val - min_val) if max_val > min_val else 0.5 for v in values]

        perf_per_param_norm = normalize(perf_per_param)
        perf_per_mb_norm = normalize(perf_per_mb)
        perf_per_ms_norm = normalize(perf_per_ms)

        # 绘制雷达图
        categories = ['Param\nEfficiency', 'Memory\nEfficiency', 'Speed\nEfficiency']
        angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
        angles += angles[:1]

        for i, name in enumerate(model_names):
            values = [perf_per_param_norm[i], perf_per_mb_norm[i], perf_per_ms_norm[i]]
            values += values[:1]
            ax.plot(angles, values, 'o-', linewidth=2, label=name)
            ax.fill(angles, values, alpha=0.15)

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories)
        ax.set_ylim(0, 1)
        ax.set_title('Model Efficiency Comparison', fontsize=14, pad=20)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
        ax.grid(True)
        plt.tight_layout()
        plt.savefig(output_dir / 'efficiency_radar.png', dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"Saved 4 scaling curve plots to {output_dir}")

    def save_analysis(self, output_path: str):
        """
        保存分析结果

        Args:
            output_path: 输出路径
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # 执行分析
        param_scaling = self.analyze_param_scaling()
        efficiency = self.analyze_efficiency()

        # 保存
        analysis = {
            "param_scaling": param_scaling,
            "efficiency": efficiency,
            "raw_results": self.results
        }

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(analysis, f, ensure_ascii=False, indent=2)

        logger.info(f"Saved scaling analysis to {output_path}")

    def print_summary(self):
        """打印分析摘要"""
        logger.info("=== Scaling Analysis Summary ===")
        logger.info(f"Total models analyzed: {len(self.results)}")

        if self.results:
            param_scaling = self.analyze_param_scaling()
            logger.info(f"\nScaling Law: {param_scaling.get('scaling_law', 'N/A')}")
            logger.info(f"R² = {param_scaling.get('r_squared', 0):.4f}")

            efficiency = self.analyze_efficiency()
            best_param = efficiency.get('best_param_efficiency', {})
            logger.info(f"\nMost parameter-efficient: {best_param.get('model_name', 'N/A')}")


def simulate_scaling_law(
    base_performance: float = 30.0,
    base_params: int = 100_000_000,
    scaling_exponent: float = 0.15,
    num_models: int = 5
) -> List[Dict[str, Any]]:
    """
    模拟缩放定律数据

    Args:
        base_performance: 基准性能
        base_params: 基准参数数量
        scaling_exponent: 缩放指数
        num_models: 模型数量

    Returns:
        模拟结果列表
    """
    results = []

    param_multipliers = np.logspace(np.log10(0.25), np.log10(4), num_models)

    for i, mult in enumerate(param_multipliers):
        num_params = int(base_params * mult)
        performance = base_performance * (mult ** scaling_exponent)

        # 添加一些随机噪声
        performance += np.random.normal(0, 0.5)

        # 模拟内存和推理时间
        memory_mb = num_params * 4 / (1024 * 1024)  # 假设 FP32
        inference_time = 0.01 * mult  # 线性增长

        results.append({
            "model_name": f"model-{i+1}",
            "num_params": num_params,
            "train_samples": 1000,
            "performance": performance,
            "inference_time": inference_time,
            "memory_mb": memory_mb
        })

    logger.info(f"Simulated {num_models} models with scaling exponent {scaling_exponent}")
    return results
