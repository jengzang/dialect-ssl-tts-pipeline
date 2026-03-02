"""
机器翻译评估指标

实现常用的机器翻译评估指标：
- BLEU (Bilingual Evaluation Understudy)
- ROUGE (Recall-Oriented Understudy for Gisting Evaluation)
- ChrF (Character n-gram F-score)
- METEOR (Metric for Evaluation of Translation with Explicit ORdering)
"""

import logging
from typing import List, Dict, Any, Optional
import numpy as np
from pathlib import Path
import json

try:
    from sacrebleu import corpus_bleu, sentence_bleu, CHRF
    from rouge_score import rouge_scorer
    import evaluate
    METRICS_AVAILABLE = True
except ImportError:
    METRICS_AVAILABLE = False
    logging.warning(
        "MT metrics not available. Install: "
        "pip install sacrebleu rouge-score evaluate"
    )

logger = logging.getLogger(__name__)


class MTMetrics:
    """机器翻译评估指标计算器"""

    def __init__(self):
        """初始化评估指标"""
        if not METRICS_AVAILABLE:
            raise ImportError(
                "Required packages not installed. Run: "
                "pip install sacrebleu rouge-score evaluate"
            )

        # 初始化 ROUGE scorer
        self.rouge_scorer = rouge_scorer.RougeScorer(
            ['rouge1', 'rouge2', 'rougeL'],
            use_stemmer=False
        )

        # 初始化 ChrF
        self.chrf = CHRF()

        # 尝试加载 METEOR（可能不可用）
        try:
            self.meteor = evaluate.load('meteor')
            self.meteor_available = True
        except Exception as e:
            logger.warning(f"METEOR not available: {e}")
            self.meteor_available = False

    def compute_bleu(
        self,
        predictions: List[str],
        references: List[str],
        max_order: int = 4
    ) -> Dict[str, float]:
        """
        计算 BLEU 分数

        Args:
            predictions: 预测翻译列表
            references: 参考翻译列表
            max_order: 最大 n-gram 阶数（默认 4）

        Returns:
            包含 BLEU 分数的字典
        """
        # 确保输入格式正确
        refs = [[ref] for ref in references]

        # 计算 corpus BLEU
        bleu = corpus_bleu(predictions, refs)

        results = {
            'bleu': bleu.score,
            'bleu_1': bleu.precisions[0],
            'bleu_2': bleu.precisions[1],
            'bleu_3': bleu.precisions[2],
            'bleu_4': bleu.precisions[3],
            'bp': bleu.bp,  # Brevity penalty
            'sys_len': bleu.sys_len,
            'ref_len': bleu.ref_len
        }

        logger.info(f"BLEU-4: {results['bleu']:.2f}")
        return results

    def compute_rouge(
        self,
        predictions: List[str],
        references: List[str]
    ) -> Dict[str, float]:
        """
        计算 ROUGE 分数

        Args:
            predictions: 预测翻译列表
            references: 参考翻译列表

        Returns:
            包含 ROUGE 分数的字典
        """
        rouge_scores = {
            'rouge1_precision': [],
            'rouge1_recall': [],
            'rouge1_fmeasure': [],
            'rouge2_precision': [],
            'rouge2_recall': [],
            'rouge2_fmeasure': [],
            'rougeL_precision': [],
            'rougeL_recall': [],
            'rougeL_fmeasure': []
        }

        for pred, ref in zip(predictions, references):
            scores = self.rouge_scorer.score(ref, pred)

            for metric in ['rouge1', 'rouge2', 'rougeL']:
                rouge_scores[f'{metric}_precision'].append(
                    scores[metric].precision
                )
                rouge_scores[f'{metric}_recall'].append(
                    scores[metric].recall
                )
                rouge_scores[f'{metric}_fmeasure'].append(
                    scores[metric].fmeasure
                )

        # 计算平均值
        results = {
            key: np.mean(values) * 100
            for key, values in rouge_scores.items()
        }

        logger.info(f"ROUGE-L F1: {results['rougeL_fmeasure']:.2f}")
        return results

    def compute_chrf(
        self,
        predictions: List[str],
        references: List[str]
    ) -> Dict[str, float]:
        """
        计算 ChrF 分数

        Args:
            predictions: 预测翻译列表
            references: 参考翻译列表

        Returns:
            包含 ChrF 分数的字典
        """
        refs = [[ref] for ref in references]
        chrf_score = self.chrf.corpus_score(predictions, refs)

        results = {
            'chrf': chrf_score.score
        }

        logger.info(f"ChrF: {results['chrf']:.2f}")
        return results

    def compute_meteor(
        self,
        predictions: List[str],
        references: List[str]
    ) -> Optional[Dict[str, float]]:
        """
        计算 METEOR 分数

        Args:
            predictions: 预测翻译列表
            references: 参考翻译列表

        Returns:
            包含 METEOR 分数的字典，如果不可用则返回 None
        """
        if not self.meteor_available:
            logger.warning("METEOR not available")
            return None

        try:
            meteor_score = self.meteor.compute(
                predictions=predictions,
                references=references
            )

            results = {
                'meteor': meteor_score['meteor'] * 100
            }

            logger.info(f"METEOR: {results['meteor']:.2f}")
            return results
        except Exception as e:
            logger.error(f"METEOR computation failed: {e}")
            return None

    def compute_all(
        self,
        predictions: List[str],
        references: List[str]
    ) -> Dict[str, Any]:
        """
        计算所有可用的评估指标

        Args:
            predictions: 预测翻译列表
            references: 参考翻译列表

        Returns:
            包含所有指标的字典
        """
        logger.info(f"Computing metrics for {len(predictions)} samples...")

        results = {}

        # BLEU
        try:
            bleu_results = self.compute_bleu(predictions, references)
            results.update(bleu_results)
        except Exception as e:
            logger.error(f"BLEU computation failed: {e}")

        # ROUGE
        try:
            rouge_results = self.compute_rouge(predictions, references)
            results.update(rouge_results)
        except Exception as e:
            logger.error(f"ROUGE computation failed: {e}")

        # ChrF
        try:
            chrf_results = self.compute_chrf(predictions, references)
            results.update(chrf_results)
        except Exception as e:
            logger.error(f"ChrF computation failed: {e}")

        # METEOR
        try:
            meteor_results = self.compute_meteor(predictions, references)
            if meteor_results:
                results.update(meteor_results)
        except Exception as e:
            logger.error(f"METEOR computation failed: {e}")

        return results

    def save_results(
        self,
        results: Dict[str, Any],
        output_path: str
    ):
        """
        保存评估结果到 JSON 文件

        Args:
            results: 评估结果字典
            output_path: 输出文件路径
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(exist_ok=True, parents=True)

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        logger.info(f"Results saved to: {output_path}")


def evaluate_translation(
    predictions: List[str],
    references: List[str],
    output_path: Optional[str] = None
) -> Dict[str, Any]:
    """
    评估翻译质量的便捷函数

    Args:
        predictions: 预测翻译列表
        references: 参考翻译列表
        output_path: 可选的输出文件路径

    Returns:
        包含所有指标的字典
    """
    metrics = MTMetrics()
    results = metrics.compute_all(predictions, references)

    if output_path:
        metrics.save_results(results, output_path)

    return results

