"""wav2vec 训练器模块

基于 HuggingFace Trainer 的 wav2vec 2.0 训练器。
"""

import torch
from transformers import Trainer, TrainingArguments, TrainerCallback
from typing import Dict, Any, Optional
import logging
from pathlib import Path
import numpy as np

from .base_trainer import BaseTrainer


class Wav2VecTrainer(BaseTrainer):
    """wav2vec 2.0 训练器

    封装 HuggingFace Trainer 进行 wav2vec 2.0 微调。
    """

    def __init__(
        self,
        model: any,
        config: Dict[str, Any],
        processor: any,
        data_collator: any,
        logger: Optional[logging.Logger] = None
    ):
        """初始化训练器

        Args:
            model: wav2vec 2.0 模型
            config: 配置字典
            processor: Wav2Vec2Processor
            data_collator: 数据整理器
            logger: 日志记录器
        """
        super().__init__(model, config, logger)
        self.processor = processor
        self.data_collator = data_collator

        # 获取配置
        self.wav2vec_config = config.get('wav2vec', {})
        self.training_config = config.get('training', {})

    def train(
        self,
        train_dataset: any,
        val_dataset: Optional[any] = None,
        output_dir: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """训练模型

        Args:
            train_dataset: 训练数据集
            val_dataset: 验证数据集
            output_dir: 输出目录
            **kwargs: 其他参数

        Returns:
            训练历史字典
        """
        if output_dir is None:
            output_dir = self.config.get('paths', {}).get('checkpoint_dir', 'checkpoints')

        output_path = Path(output_dir) / "wav2vec_ipa"
        output_path.mkdir(parents=True, exist_ok=True)

        self.logger.info("=" * 60)
        self.logger.info("开始训练 wav2vec 2.0 模型")
        self.logger.info("=" * 60)

        # 创建训练参数
        training_args = TrainingArguments(
            output_dir=str(output_path),
            group_by_length=True,
            per_device_train_batch_size=self.wav2vec_config.get('batch_size', 8),
            per_device_eval_batch_size=self.wav2vec_config.get('batch_size', 8),
            gradient_accumulation_steps=self.wav2vec_config.get('gradient_accumulation_steps', 2),
            evaluation_strategy="steps",
            num_train_epochs=self.wav2vec_config.get('epochs', 10),
            fp16=torch.cuda.is_available(),
            save_steps=500,
            eval_steps=500,
            logging_steps=100,
            learning_rate=self.wav2vec_config.get('learning_rate', 1e-4),
            warmup_steps=self.wav2vec_config.get('warmup_steps', 500),
            save_total_limit=2,
            push_to_hub=False,
            load_best_model_at_end=True,
            metric_for_best_model="wer",
            greater_is_better=False,
        )

        # 创建 Trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            data_collator=self.data_collator,
            tokenizer=self.processor.feature_extractor,
            compute_metrics=self._compute_metrics,
        )

        # 训练
        self.logger.info("开始训练...")
        train_result = trainer.train()

        # 保存模型
        self.logger.info("保存模型...")
        trainer.save_model()

        # 保存训练指标
        metrics = train_result.metrics
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)

        self.logger.info("=" * 60)
        self.logger.info("训练完成")
        self.logger.info("=" * 60)

        return {
            'train_metrics': metrics,
            'trainer': trainer
        }

    def _compute_metrics(self, pred):
        """计算评估指标

        Args:
            pred: 预测结果

        Returns:
            指标字典
        """
        from evaluate import load

        wer_metric = load("wer")

        pred_logits = pred.predictions
        pred_ids = np.argmax(pred_logits, axis=-1)

        pred.label_ids[pred.label_ids == -100] = self.processor.tokenizer.pad_token_id

        pred_str = self.processor.batch_decode(pred_ids)
        label_str = self.processor.batch_decode(pred.label_ids, group_tokens=False)

        wer = wer_metric.compute(predictions=pred_str, references=label_str)

        return {"wer": wer}

    def evaluate(self, test_dataset: any) -> Dict[str, float]:
        """评估模型

        Args:
            test_dataset: 测试数据集

        Returns:
            评估指标字典
        """
        self.logger.info("评估模型...")

        # 创建临时 Trainer 用于评估
        training_args = TrainingArguments(
            output_dir="./tmp",
            per_device_eval_batch_size=self.wav2vec_config.get('batch_size', 8),
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            data_collator=self.data_collator,
            compute_metrics=self._compute_metrics,
            tokenizer=self.processor.feature_extractor,
        )

        # 评估
        metrics = trainer.evaluate(test_dataset)

        self.logger.info(f"评估结果: {metrics}")

        return metrics

    def save_checkpoint(self, path: str, **kwargs):
        """保存检查点

        Args:
            path: 保存路径
            **kwargs: 其他参数
        """
        output_dir = Path(path)
        output_dir.mkdir(parents=True, exist_ok=True)

        # 保存模型
        self.model.save_pretrained(output_dir)

        # 保存 processor
        self.processor.save_pretrained(output_dir)

        self.logger.info(f"检查点已保存到: {output_dir}")

    def load_checkpoint(self, path: str):
        """加载检查点

        Args:
            path: 检查点路径
        """
        from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

        model_dir = Path(path)

        # 加载 processor
        self.processor = Wav2Vec2Processor.from_pretrained(model_dir)

        # 加载模型
        self.model = Wav2Vec2ForCTC.from_pretrained(model_dir)

        self.logger.info(f"检查点已加载: {model_dir}")


def prepare_dataset(batch, processor):
    """准备数据集

    将音频和文本转换为模型输入格式。

    Args:
        batch: 批次数据
        processor: Wav2Vec2Processor

    Returns:
        处理后的批次
    """
    # 处理音频
    audio = batch["audio"]

    # 提取输入特征
    batch["input_values"] = processor(
        audio["array"],
        sampling_rate=audio["sampling_rate"]
    ).input_values[0]

    # 编码标签
    with processor.as_target_processor():
        batch["labels"] = processor(batch["text"]).input_ids

    return batch
