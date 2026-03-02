"""
LoRA 权重分析模块

分析 LoRA 权重的分布、重要性和效率。
"""

import logging
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt

try:
    import seaborn as sns
    sns.set_style("whitegrid")
except ImportError:
    pass  # seaborn 是可选的

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logging.warning("PyTorch not available. Some features will be disabled.")

logger = logging.getLogger(__name__)


class LoRAAnalyzer:
    """LoRA 权重分析器"""

    def __init__(self, model_path: Optional[str] = None):
        """
        初始化分析器

        Args:
            model_path: LoRA 模型路径
        """
        self.model_path = model_path
        self.lora_weights = {}

        if model_path and TORCH_AVAILABLE:
            self.load_lora_weights(model_path)

    def load_lora_weights(self, model_path: str):
        """
        加载 LoRA 权重

        Args:
            model_path: 模型路径
        """
        if not TORCH_AVAILABLE:
            logger.error("PyTorch not available")
            return

        model_path = Path(model_path)

        # 查找 LoRA 权重文件
        lora_files = list(model_path.glob("adapter_*.bin"))

        if not lora_files:
            logger.warning(f"No LoRA weights found in {model_path}")
            return

        for lora_file in lora_files:
            try:
                weights = torch.load(lora_file, map_location='cpu')
                self.lora_weights[lora_file.name] = weights
                logger.info(f"Loaded LoRA weights from {lora_file}")
            except Exception as e:
                logger.error(f"Failed to load {lora_file}: {e}")

    def analyze_weight_distribution(
        self,
        output_dir: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        分析权重分布

        Args:
            output_dir: 输出目录（用于保存图表）

        Returns:
            分析结果字典
        """
        if not self.lora_weights:
            logger.warning("No LoRA weights loaded")
            return {}

        results = {}

        for name, weights in self.lora_weights.items():
            layer_stats = {}

            for key, tensor in weights.items():
                if isinstance(tensor, torch.Tensor):
                    tensor_np = tensor.detach().cpu().numpy()

                    stats = {
                        'shape': list(tensor.shape),
                        'mean': float(np.mean(tensor_np)),
                        'std': float(np.std(tensor_np)),
                        'min': float(np.min(tensor_np)),
                        'max': float(np.max(tensor_np)),
                        'norm': float(np.linalg.norm(tensor_np))
                    }

                    layer_stats[key] = stats

            results[name] = layer_stats

        # 保存结果
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(exist_ok=True, parents=True)

            results_path = output_dir / "weight_distribution.json"
            with open(results_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2)
            logger.info(f"Weight distribution saved to: {results_path}")

        return results

    def plot_weight_distribution(
        self,
        output_dir: str,
        layer_name: Optional[str] = None
    ):
        """
        绘制权重分布图

        Args:
            output_dir: 输出目录
            layer_name: 指定层名称（可选）
        """
        if not self.lora_weights:
            logger.warning("No LoRA weights loaded")
            return

        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)

        for name, weights in self.lora_weights.items():
            for key, tensor in weights.items():
                if layer_name and layer_name not in key:
                    continue

                if isinstance(tensor, torch.Tensor):
                    tensor_np = tensor.detach().cpu().numpy().flatten()

                    plt.figure(figsize=(10, 6))
                    plt.hist(tensor_np, bins=50, alpha=0.7, edgecolor='black')
                    plt.xlabel('Weight Value')
                    plt.ylabel('Frequency')
                    plt.title(f'Weight Distribution: {key}')
                    plt.grid(True, alpha=0.3)

                    # 保存图表
                    safe_key = key.replace('/', '_').replace('.', '_')
                    plot_path = output_dir / f"dist_{safe_key}.png"
                    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
                    plt.close()

                    logger.info(f"Saved plot: {plot_path}")

    def compare_lora_configs(
        self,
        configs: List[Dict[str, Any]],
        scores: List[float],
        output_dir: str
    ):
        """
        比较不同 LoRA 配置

        Args:
            configs: 配置列表
            scores: 对应的分数列表
            output_dir: 输出目录
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)

        # 提取关键参数
        ranks = [c.get('lora_r', 0) for c in configs]
        alphas = [c.get('lora_alpha', 0) for c in configs]
        lrs = [c.get('learning_rate', 0) for c in configs]

        # 1. Rank vs Score
        plt.figure(figsize=(10, 6))
        plt.scatter(ranks, scores, alpha=0.6, s=100)
        plt.xlabel('LoRA Rank')
        plt.ylabel('BLEU Score')
        plt.title('LoRA Rank vs Performance')
        plt.grid(True, alpha=0.3)
        plt.savefig(output_dir / 'rank_vs_score.png', dpi=150, bbox_inches='tight')
        plt.close()

        # 2. Alpha vs Score
        plt.figure(figsize=(10, 6))
        plt.scatter(alphas, scores, alpha=0.6, s=100)
        plt.xlabel('LoRA Alpha')
        plt.ylabel('BLEU Score')
        plt.title('LoRA Alpha vs Performance')
        plt.grid(True, alpha=0.3)
        plt.savefig(output_dir / 'alpha_vs_score.png', dpi=150, bbox_inches='tight')
        plt.close()

        # 3. Learning Rate vs Score
        plt.figure(figsize=(10, 6))
        plt.scatter(lrs, scores, alpha=0.6, s=100)
        plt.xscale('log')
        plt.xlabel('Learning Rate')
        plt.ylabel('BLEU Score')
        plt.title('Learning Rate vs Performance')
        plt.grid(True, alpha=0.3)
        plt.savefig(output_dir / 'lr_vs_score.png', dpi=150, bbox_inches='tight')
        plt.close()

        # 4. Rank vs Alpha (颜色表示分数)
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(ranks, alphas, c=scores, cmap='viridis', s=100, alpha=0.6)
        plt.colorbar(scatter, label='BLEU Score')
        plt.xlabel('LoRA Rank')
        plt.ylabel('LoRA Alpha')
        plt.title('LoRA Configuration Space')
        plt.grid(True, alpha=0.3)
        plt.savefig(output_dir / 'config_space.png', dpi=150, bbox_inches='tight')
        plt.close()

        logger.info(f"Comparison plots saved to: {output_dir}")

    def plot_pareto_front(
        self,
        params: List[int],
        scores: List[float],
        output_dir: str,
        param_name: str = "Trainable Parameters"
    ):
        """
        绘制帕累托前沿

        Args:
            params: 参数数量列表
            scores: 分数列表
            output_dir: 输出目录
            param_name: 参数名称
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)

        # 找到帕累托前沿
        pareto_indices = []
        for i in range(len(params)):
            is_pareto = True
            for j in range(len(params)):
                if i != j:
                    # 如果存在另一个点，参数更少且性能更好
                    if params[j] < params[i] and scores[j] > scores[i]:
                        is_pareto = False
                        break
            if is_pareto:
                pareto_indices.append(i)

        # 绘制
        plt.figure(figsize=(10, 6))

        # 所有点
        plt.scatter(params, scores, alpha=0.3, s=50, label='All configurations')

        # 帕累托前沿
        pareto_params = [params[i] for i in pareto_indices]
        pareto_scores = [scores[i] for i in pareto_indices]
        plt.scatter(pareto_params, pareto_scores, color='red', s=100,
                   label='Pareto front', zorder=5)

        # 连接帕累托前沿点
        sorted_pareto = sorted(zip(pareto_params, pareto_scores))
        if sorted_pareto:
            px, py = zip(*sorted_pareto)
            plt.plot(px, py, 'r--', alpha=0.5, zorder=4)

        plt.xlabel(param_name)
        plt.ylabel('BLEU Score')
        plt.title('Performance vs Efficiency (Pareto Front)')
        plt.legend()
        plt.grid(True, alpha=0.3)

        plot_path = output_dir / 'pareto_front.png'
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()

        logger.info(f"Pareto front plot saved to: {plot_path}")

        return pareto_indices

    def estimate_trainable_params(self, lora_config: Dict[str, Any]) -> int:
        """
        估算可训练参数数量

        Args:
            lora_config: LoRA 配置

        Returns:
            估算的参数数量
        """
        r = lora_config.get('lora_r', 8)
        d_model = lora_config.get('d_model', 4096)  # 假设值
        num_layers = lora_config.get('num_layers', 32)  # 假设值

        # 简化估算：每个 LoRA 层有两个矩阵 A 和 B
        # A: d_model x r
        # B: r x d_model
        # 总参数: 2 * d_model * r * num_layers * num_target_modules

        target_modules = lora_config.get('target_modules', 'q_proj,v_proj')
        num_target = len(target_modules.split(','))

        params = 2 * d_model * r * num_layers * num_target

        return params


def analyze_lora_efficiency(
    configs: List[Dict[str, Any]],
    scores: List[float],
    output_dir: str
) -> Dict[str, Any]:
    """
    分析 LoRA 效率的便捷函数

    Args:
        configs: 配置列表
        scores: 分数列表
        output_dir: 输出目录

    Returns:
        分析结果
    """
    analyzer = LoRAAnalyzer()

    # 比较配置
    analyzer.compare_lora_configs(configs, scores, output_dir)

    # 估算参数数量
    params = [analyzer.estimate_trainable_params(c) for c in configs]

    # 绘制帕累托前沿
    pareto_indices = analyzer.plot_pareto_front(params, scores, output_dir)

    # 找到最佳配置
    best_idx = np.argmax(scores)
    best_config = configs[best_idx]
    best_score = scores[best_idx]

    # 找到最高效配置（帕累托前沿上参数最少的）
    if pareto_indices:
        pareto_params = [params[i] for i in pareto_indices]
        most_efficient_idx = pareto_indices[np.argmin(pareto_params)]
        most_efficient_config = configs[most_efficient_idx]
        most_efficient_score = scores[most_efficient_idx]
    else:
        most_efficient_config = best_config
        most_efficient_score = best_score

    results = {
        'best_config': best_config,
        'best_score': best_score,
        'most_efficient_config': most_efficient_config,
        'most_efficient_score': most_efficient_score,
        'pareto_front_size': len(pareto_indices)
    }

    # 保存结果
    output_dir = Path(output_dir)
    results_path = output_dir / 'efficiency_analysis.json'
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)

    logger.info(f"Efficiency analysis saved to: {results_path}")

    return results
