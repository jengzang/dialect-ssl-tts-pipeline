"""
指令遵循评估模块

评估模型遵循指令的能力。

评估指标：
1. 指令遵循准确率
2. 任务完成率
3. 输出格式正确性
4. 零样本性能
5. 少样本性能
"""

import logging
from typing import Dict, Any, List, Optional
import json
import re
from pathlib import Path

logger = logging.getLogger(__name__)


class InstructionEvaluator:
    """指令遵循评估器"""

    def __init__(self):
        """初始化评估器"""
        self.results = []

    def evaluate_translation(
        self,
        predicted: str,
        reference: str
    ) -> Dict[str, float]:
        """
        评估翻译任务

        Args:
            predicted: 预测文本
            reference: 参考文本

        Returns:
            评估指标
        """
        # 简单的字符级匹配
        predicted = predicted.strip()
        reference = reference.strip()

        # 完全匹配
        exact_match = 1.0 if predicted == reference else 0.0

        # 字符级 F1
        pred_chars = set(predicted)
        ref_chars = set(reference)

        if len(pred_chars) == 0 and len(ref_chars) == 0:
            char_f1 = 1.0
        elif len(pred_chars) == 0 or len(ref_chars) == 0:
            char_f1 = 0.0
        else:
            precision = len(pred_chars & ref_chars) / len(pred_chars)
            recall = len(pred_chars & ref_chars) / len(ref_chars)
            char_f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        # 长度比率
        length_ratio = min(len(predicted), len(reference)) / max(len(predicted), len(reference)) if max(len(predicted), len(reference)) > 0 else 0.0

        return {
            "exact_match": exact_match,
            "char_f1": char_f1,
            "length_ratio": length_ratio
        }

    def evaluate_classification(
        self,
        predicted: str,
        reference: str
    ) -> Dict[str, float]:
        """
        评估分类任务

        Args:
            predicted: 预测文本
            reference: 参考文本

        Returns:
            评估指标
        """
        predicted = predicted.strip().lower()
        reference = reference.strip().lower()

        # 完全匹配
        exact_match = 1.0 if predicted == reference else 0.0

        # 包含匹配（预测中包含参考）
        contains_match = 1.0 if reference in predicted else 0.0

        return {
            "exact_match": exact_match,
            "contains_match": contains_match
        }

    def evaluate_format_correctness(
        self,
        predicted: str,
        expected_format: str = "text"
    ) -> float:
        """
        评估输出格式正确性

        Args:
            predicted: 预测文本
            expected_format: 期望格式

        Returns:
            格式正确性分数
        """
        if expected_format == "text":
            # 检查是否为有效文本（非空，无特殊字符）
            if len(predicted.strip()) == 0:
                return 0.0
            if len(predicted) > 500:  # 过长
                return 0.5
            return 1.0

        elif expected_format == "label":
            # 检查是否为单个标签
            if len(predicted.split()) <= 3:  # 短标签
                return 1.0
            return 0.5

        return 1.0

    def evaluate_instruction_following(
        self,
        instruction: str,
        input_text: str,
        predicted: str,
        reference: str,
        task_type: str
    ) -> Dict[str, Any]:
        """
        评估指令遵循能力

        Args:
            instruction: 指令
            input_text: 输入文本
            predicted: 预测文本
            reference: 参考文本
            task_type: 任务类型

        Returns:
            评估结果
        """
        result = {
            "instruction": instruction,
            "input": input_text,
            "predicted": predicted,
            "reference": reference,
            "task_type": task_type
        }

        # 任务特定评估
        if task_type == "translation":
            metrics = self.evaluate_translation(predicted, reference)
            format_score = self.evaluate_format_correctness(predicted, "text")
        elif task_type == "classification":
            metrics = self.evaluate_classification(predicted, reference)
            format_score = self.evaluate_format_correctness(predicted, "label")
        else:
            metrics = {"exact_match": 0.0}
            format_score = 1.0

        result.update(metrics)
        result["format_correctness"] = format_score

        # 计算总分
        result["overall_score"] = sum(metrics.values()) / len(metrics) if metrics else 0.0

        self.results.append(result)

        return result

    def compute_aggregate_metrics(self) -> Dict[str, float]:
        """
        计算聚合指标

        Returns:
            聚合指标
        """
        if not self.results:
            return {}

        # 按任务类型分组
        by_task = {}
        for result in self.results:
            task_type = result["task_type"]
            if task_type not in by_task:
                by_task[task_type] = []
            by_task[task_type].append(result)

        # 计算每个任务的平均指标
        aggregate = {}

        for task_type, results in by_task.items():
            task_metrics = {}

            # 收集所有指标
            all_metrics = set()
            for result in results:
                for key in result.keys():
                    if isinstance(result[key], (int, float)) and key not in ["instruction", "input", "predicted", "reference", "task_type"]:
                        all_metrics.add(key)

            # 计算平均值
            for metric in all_metrics:
                values = [r[metric] for r in results if metric in r]
                if values:
                    task_metrics[f"{task_type}_{metric}"] = sum(values) / len(values)

            aggregate.update(task_metrics)

        # 计算总体指标
        if self.results:
            aggregate["overall_accuracy"] = sum(r.get("exact_match", 0) for r in self.results) / len(self.results)
            aggregate["overall_score"] = sum(r.get("overall_score", 0) for r in self.results) / len(self.results)
            aggregate["format_correctness"] = sum(r.get("format_correctness", 0) for r in self.results) / len(self.results)

        return aggregate

    def save_results(self, output_path: str):
        """
        保存评估结果

        Args:
            output_path: 输出路径
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # 计算聚合指标
        aggregate_metrics = self.compute_aggregate_metrics()

        # 保存
        output = {
            "aggregate_metrics": aggregate_metrics,
            "detailed_results": self.results,
            "num_samples": len(self.results)
        }

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output, f, ensure_ascii=False, indent=2)

        logger.info(f"Saved evaluation results to {output_path}")

    def print_summary(self):
        """打印评估摘要"""
        aggregate = self.compute_aggregate_metrics()

        logger.info("=== Instruction Following Evaluation Summary ===")
        logger.info(f"Total samples: {len(self.results)}")

        for key, value in aggregate.items():
            logger.info(f"{key}: {value:.4f}")


def evaluate_few_shot_performance(
    model,
    tokenizer,
    test_data: List[Dict[str, Any]],
    num_shots: List[int] = [0, 1, 3, 5],
    device: str = "cuda"
) -> Dict[str, Dict[str, float]]:
    """
    评估少样本性能

    Args:
        model: 模型
        tokenizer: Tokenizer
        test_data: 测试数据
        num_shots: 少样本数量列表
        device: 设备

    Returns:
        少样本性能结果
    """
    results = {}

    for n_shot in num_shots:
        logger.info(f"Evaluating {n_shot}-shot performance...")

        evaluator = InstructionEvaluator()

        for i, test_item in enumerate(test_data):
            # 选择示例
            if n_shot > 0:
                # 从训练数据中随机选择示例（这里简化为使用测试数据的前 n_shot 个）
                examples = test_data[:n_shot]
                examples = [{"input": ex["input"], "output": ex["output"]} for ex in examples]
            else:
                examples = []

            # 生成
            if n_shot == 0:
                # 零样本
                prompt = f"{test_item['instruction']}\n输入：{test_item['input']}\n输出："
                inputs = tokenizer(prompt, return_tensors="pt", max_length=256, truncation=True).to(device)

                with torch.no_grad():
                    outputs = model.generate(
                        input_ids=inputs['input_ids'],
                        attention_mask=inputs['attention_mask'],
                        max_length=256
                    )

                predicted = tokenizer.decode(outputs[0], skip_special_tokens=True)
                if "输出：" in predicted:
                    predicted = predicted.split("输出：")[-1].strip()
            else:
                # 少样本
                predicted = model.generate_with_few_shot(
                    tokenizer=tokenizer,
                    instruction=test_item['instruction'],
                    examples=examples,
                    input_text=test_item['input']
                )

            # 评估
            evaluator.evaluate_instruction_following(
                instruction=test_item['instruction'],
                input_text=test_item['input'],
                predicted=predicted,
                reference=test_item['output'],
                task_type=test_item.get('task_type', 'translation')
            )

        # 计算聚合指标
        aggregate = evaluator.compute_aggregate_metrics()
        results[f"{n_shot}-shot"] = aggregate

        logger.info(f"{n_shot}-shot accuracy: {aggregate.get('overall_accuracy', 0):.4f}")

    return results
