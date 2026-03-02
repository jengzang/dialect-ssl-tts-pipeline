"""
超参数搜索模块

使用 Optuna 进行贝叶斯优化，系统性探索 LoRA 超参数空间。

支持的超参数：
- LoRA rank (r)
- LoRA alpha
- 学习率
- 批次大小
- 目标模块
"""

import logging
from typing import Dict, Any, List, Optional, Callable
from pathlib import Path
import json
import optuna
from optuna.trial import Trial
from optuna.visualization import (
    plot_optimization_history,
    plot_param_importances,
    plot_parallel_coordinate
)
import pandas as pd

logger = logging.getLogger(__name__)


class LoRAHyperparameterSearch:
    """LoRA 超参数搜索器"""

    def __init__(
        self,
        study_name: str,
        storage: Optional[str] = None,
        direction: str = "maximize",
        load_if_exists: bool = True
    ):
        """
        初始化超参数搜索器

        Args:
            study_name: 研究名称
            storage: Optuna 存储路径（SQLite）
            direction: 优化方向（maximize 或 minimize）
            load_if_exists: 是否加载已存在的研究
        """
        self.study_name = study_name
        self.direction = direction

        # 创建或加载研究
        if storage is None:
            storage = f"sqlite:///optuna_studies/{study_name}.db"

        # 确保目录存在
        Path("optuna_studies").mkdir(exist_ok=True)

        self.study = optuna.create_study(
            study_name=study_name,
            storage=storage,
            direction=direction,
            load_if_exists=load_if_exists
        )

        logger.info(f"Initialized study: {study_name}")
        logger.info(f"Direction: {direction}")
        logger.info(f"Storage: {storage}")

    def suggest_lora_params(self, trial: Trial) -> Dict[str, Any]:
        """
        建议 LoRA 超参数

        Args:
            trial: Optuna trial 对象

        Returns:
            超参数字典
        """
        params = {
            # LoRA rank: 2^n 形式
            'lora_r': trial.suggest_categorical(
                'lora_r',
                [4, 8, 16, 32, 64]
            ),

            # LoRA alpha: 通常是 rank 的倍数
            'lora_alpha': trial.suggest_categorical(
                'lora_alpha',
                [16, 32, 64, 128]
            ),

            # 学习率: 对数空间
            'learning_rate': trial.suggest_float(
                'learning_rate',
                1e-5, 5e-4,
                log=True
            ),

            # 批次大小
            'batch_size': trial.suggest_categorical(
                'batch_size',
                [1, 2, 4, 8]
            ),

            # Dropout
            'lora_dropout': trial.suggest_float(
                'lora_dropout',
                0.0, 0.3
            ),

            # 目标模块（简化版）
            'target_modules': trial.suggest_categorical(
                'target_modules',
                ['q_proj,v_proj', 'q_proj,k_proj,v_proj', 'all']
            )
        }

        return params

    def objective_function(
        self,
        trial: Trial,
        train_fn: Callable[[Dict[str, Any]], float]
    ) -> float:
        """
        目标函数

        Args:
            trial: Optuna trial 对象
            train_fn: 训练函数，接受超参数字典，返回评估指标

        Returns:
            评估指标值
        """
        # 建议超参数
        params = self.suggest_lora_params(trial)

        logger.info(f"Trial {trial.number}: {params}")

        try:
            # 执行训练并获取评估指标
            score = train_fn(params)

            # 记录中间结果
            trial.set_user_attr('params', params)
            trial.set_user_attr('score', score)

            logger.info(f"Trial {trial.number} completed: score={score:.4f}")

            return score

        except Exception as e:
            logger.error(f"Trial {trial.number} failed: {e}")
            raise optuna.TrialPruned()

    def optimize(
        self,
        train_fn: Callable[[Dict[str, Any]], float],
        n_trials: int = 50,
        timeout: Optional[int] = None,
        n_jobs: int = 1
    ):
        """
        执行优化

        Args:
            train_fn: 训练函数
            n_trials: 试验次数
            timeout: 超时时间（秒）
            n_jobs: 并行任务数
        """
        logger.info(f"Starting optimization: {n_trials} trials")

        self.study.optimize(
            lambda trial: self.objective_function(trial, train_fn),
            n_trials=n_trials,
            timeout=timeout,
            n_jobs=n_jobs,
            show_progress_bar=True
        )

        logger.info("Optimization completed")
        self._print_results()

    def _print_results(self):
        """打印优化结果"""
        logger.info("=" * 60)
        logger.info("Optimization Results")
        logger.info("=" * 60)

        # 最佳试验
        best_trial = self.study.best_trial
        logger.info(f"\nBest trial: {best_trial.number}")
        logger.info(f"Best value: {best_trial.value:.4f}")
        logger.info(f"Best params:")
        for key, value in best_trial.params.items():
            logger.info(f"  {key}: {value}")

        # 统计信息
        logger.info(f"\nTotal trials: {len(self.study.trials)}")
        logger.info(f"Completed trials: {len([t for t in self.study.trials if t.state == optuna.trial.TrialState.COMPLETE])}")
        logger.info(f"Pruned trials: {len([t for t in self.study.trials if t.state == optuna.trial.TrialState.PRUNED])}")

    def get_best_params(self) -> Dict[str, Any]:
        """
        获取最佳超参数

        Returns:
            最佳超参数字典
        """
        return self.study.best_params

    def save_results(self, output_dir: str):
        """
        保存优化结果

        Args:
            output_dir: 输出目录
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)

        # 保存最佳参数
        best_params_path = output_dir / "best_params.json"
        with open(best_params_path, 'w', encoding='utf-8') as f:
            json.dump(self.study.best_params, f, indent=2)
        logger.info(f"Best params saved to: {best_params_path}")

        # 保存所有试验结果
        trials_df = self.study.trials_dataframe()
        trials_path = output_dir / "trials.csv"
        trials_df.to_csv(trials_path, index=False)
        logger.info(f"Trials saved to: {trials_path}")

        # 保存可视化
        try:
            # 优化历史
            fig = plot_optimization_history(self.study)
            fig.write_html(str(output_dir / "optimization_history.html"))

            # 参数重要性
            fig = plot_param_importances(self.study)
            fig.write_html(str(output_dir / "param_importances.html"))

            # 平行坐标图
            fig = plot_parallel_coordinate(self.study)
            fig.write_html(str(output_dir / "parallel_coordinate.html"))

            logger.info("Visualizations saved")

        except Exception as e:
            logger.warning(f"Failed to save visualizations: {e}")

    def analyze_pareto_front(self) -> pd.DataFrame:
        """
        分析帕累托前沿（性能 vs 参数数量）

        Returns:
            帕累托前沿数据
        """
        trials_df = self.study.trials_dataframe()

        # 计算可训练参数数量（简化估算）
        def estimate_params(row):
            r = row['params_lora_r']
            # 假设模型有 N 个注意力层，每层有 q, k, v 投影
            # 简化估算：params ≈ 2 * d_model * r * num_layers * num_projections
            # 这里使用 rank 作为相对指标
            return r

        trials_df['trainable_params'] = trials_df.apply(estimate_params, axis=1)

        # 找到帕累托前沿
        pareto_front = []
        for idx, row in trials_df.iterrows():
            is_pareto = True
            for _, other_row in trials_df.iterrows():
                # 如果存在另一个点，参数更少且性能更好
                if (other_row['trainable_params'] < row['trainable_params'] and
                    other_row['value'] > row['value']):
                    is_pareto = False
                    break
            if is_pareto:
                pareto_front.append(row)

        pareto_df = pd.DataFrame(pareto_front)
        logger.info(f"Pareto front: {len(pareto_df)} points")

        return pareto_df


def create_mock_train_function() -> Callable[[Dict[str, Any]], float]:
    """
    创建模拟训练函数（用于测试）

    Returns:
        模拟训练函数
    """
    def mock_train(params: Dict[str, Any]) -> float:
        """
        模拟训练函数

        根据超参数返回模拟的 BLEU 分数
        """
        import random
        import time

        # 模拟训练时间
        time.sleep(0.1)

        # 基于超参数计算模拟分数
        r = params['lora_r']
        alpha = params['lora_alpha']
        lr = params['learning_rate']

        # 简单的启发式规则
        base_score = 30.0

        # rank 越大，性能越好（但收益递减）
        rank_bonus = min(r / 10, 5.0)

        # alpha/rank 比例接近 2-4 时效果较好
        ratio = alpha / r
        ratio_bonus = 3.0 if 2 <= ratio <= 4 else 0.0

        # 学习率在 1e-4 附近较好
        lr_bonus = 2.0 if 5e-5 <= lr <= 2e-4 else 0.0

        # 添加随机噪声
        noise = random.gauss(0, 1.0)

        score = base_score + rank_bonus + ratio_bonus + lr_bonus + noise

        return max(0, min(100, score))  # 限制在 [0, 100]

    return mock_train
