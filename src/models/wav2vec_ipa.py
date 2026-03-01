"""wav2vec 2.0 模型模块

基于 HuggingFace Transformers 的 wav2vec 2.0 微调模型。
"""

import torch
from transformers import (
    Wav2Vec2CTCTokenizer,
    Wav2Vec2FeatureExtractor,
    Wav2Vec2Processor,
    Wav2Vec2ForCTC
)
from typing import Dict, Any, Optional, List
import logging
from pathlib import Path

from .base import BaseModel


class Wav2VecIPAModel(BaseModel):
    """wav2vec 2.0 IPA 识别模型

    用于音素（IPA）识别的 wav2vec 2.0 微调模型。
    """

    def __init__(self, config: Dict[str, Any]):
        """初始化模型

        Args:
            config: 配置字典
        """
        super().__init__(config)
        self.logger = logging.getLogger(__name__)

        self.model_name = config.get('wav2vec', {}).get(
            'model_name', 'facebook/wav2vec2-base'
        )
        self.processor = None
        self.tokenizer = None
        self.feature_extractor = None

    def build(
        self,
        vocab: List[str],
        freeze_feature_encoder: bool = True
    ):
        """构建模型

        Args:
            vocab: 词汇表（音素列表）
            freeze_feature_encoder: 是否冻结特征提取器
        """
        self.logger.info(f"构建 wav2vec 2.0 模型: {self.model_name}")

        # 创建 tokenizer
        vocab_dict = {v: k for k, v in enumerate(vocab)}
        vocab_dict["[UNK]"] = len(vocab_dict)
        vocab_dict["[PAD]"] = len(vocab_dict)

        self.tokenizer = Wav2Vec2CTCTokenizer(
            vocab_dict,
            unk_token="[UNK]",
            pad_token="[PAD]",
            word_delimiter_token="|"
        )

        # 创建 feature extractor
        self.feature_extractor = Wav2Vec2FeatureExtractor(
            feature_size=1,
            sampling_rate=16000,
            padding_value=0.0,
            do_normalize=True,
            return_attention_mask=True
        )

        # 创建 processor
        self.processor = Wav2Vec2Processor(
            feature_extractor=self.feature_extractor,
            tokenizer=self.tokenizer
        )

        # 加载预训练模型
        self.model = Wav2Vec2ForCTC.from_pretrained(
            self.model_name,
            ctc_loss_reduction="mean",
            pad_token_id=self.processor.tokenizer.pad_token_id,
            vocab_size=len(vocab_dict)
        )

        # 冻结特征提取器
        if freeze_feature_encoder:
            self.model.freeze_feature_encoder()
            self.logger.info("特征提取器已冻结")

        self.logger.info(f"模型已构建，词汇表大小: {len(vocab_dict)}")

    def train(self, *args, **kwargs):
        """训练模型

        注意：wav2vec 的训练逻辑在 Wav2VecTrainer 中实现
        """
        raise NotImplementedError("请使用 Wav2VecTrainer 进行训练")

    def predict(
        self,
        audio_input: torch.Tensor,
        return_logits: bool = False
    ) -> Union[str, torch.Tensor]:
        """预测

        Args:
            audio_input: 音频输入张量
            return_logits: 是否返回 logits

        Returns:
            预测的文本或 logits
        """
        self.model.eval()

        with torch.no_grad():
            # 如果输入是原始音频，需要先处理
            if audio_input.dim() == 1:
                inputs = self.processor(
                    audio_input,
                    sampling_rate=16000,
                    return_tensors="pt",
                    padding=True
                )
                audio_input = inputs.input_values

            # 前向传播
            logits = self.model(audio_input).logits

            if return_logits:
                return logits

            # 解码
            predicted_ids = torch.argmax(logits, dim=-1)
            transcription = self.processor.batch_decode(predicted_ids)[0]

        return transcription

    def evaluate(self, test_dataset, metric_fn=None) -> Dict[str, float]:
        """评估模型

        Args:
            test_dataset: 测试数据集
            metric_fn: 评估指标函数

        Returns:
            评估指标字典
        """
        self.model.eval()

        predictions = []
        references = []

        for example in test_dataset:
            # 预测
            pred = self.predict(example['audio']['array'])
            predictions.append(pred)

            # 参考
            references.append(example['text'])

        # 计算指标
        if metric_fn is not None:
            metrics = metric_fn(predictions, references)
        else:
            # 默认计算 WER
            from evaluate import load
            wer_metric = load("wer")
            wer = wer_metric.compute(predictions=predictions, references=references)
            metrics = {'wer': wer}

        return metrics

    def save(self, path: str):
        """保存模型

        Args:
            path: 保存路径
        """
        output_dir = Path(path)
        output_dir.mkdir(parents=True, exist_ok=True)

        # 保存模型
        self.model.save_pretrained(output_dir)

        # 保存 processor
        self.processor.save_pretrained(output_dir)

        self.logger.info(f"模型已保存到: {output_dir}")

    def load(self, path: str):
        """加载模型

        Args:
            path: 模型路径
        """
        model_dir = Path(path)

        # 加载 processor
        self.processor = Wav2Vec2Processor.from_pretrained(model_dir)
        self.tokenizer = self.processor.tokenizer
        self.feature_extractor = self.processor.feature_extractor

        # 加载模型
        self.model = Wav2Vec2ForCTC.from_pretrained(model_dir)

        self.logger.info(f"模型已加载: {model_dir}")

    def get_vocab(self) -> Dict[str, int]:
        """获取词汇表

        Returns:
            词汇表字典
        """
        if self.tokenizer is None:
            return {}

        return self.tokenizer.get_vocab()

    @staticmethod
    def create_vocab_from_dataset(dataset, text_column: str = 'text') -> List[str]:
        """从数据集创建词汇表

        Args:
            dataset: 数据集
            text_column: 文本列名

        Returns:
            词汇表列表
        """
        vocab_set = set()

        for example in dataset:
            text = example[text_column]
            # 提取所有字符
            for char in text:
                if char not in [' ', '|']:  # 排除空格和分隔符
                    vocab_set.add(char)

        vocab_list = sorted(list(vocab_set))

        return vocab_list
