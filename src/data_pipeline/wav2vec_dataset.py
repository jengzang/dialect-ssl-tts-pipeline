"""wav2vec 数据集模块

提供 HuggingFace 格式的数据集构建功能。
"""

import torch
import torchaudio
from datasets import Dataset, DatasetDict, Audio
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import pandas as pd
import logging
from dataclasses import dataclass


class Wav2VecDatasetBuilder:
    """wav2vec 数据集构建器

    将音频文件和标注构建为 HuggingFace Dataset 格式。
    """

    def __init__(self, config: Dict):
        """初始化数据集构建器

        Args:
            config: 配置字典
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.target_sr = 16000  # wav2vec 要求 16kHz

    def build_from_csv(
        self,
        csv_file: str,
        audio_column: str = 'audio_path',
        text_column: str = 'text',
        split_ratios: Optional[Dict[str, float]] = None
    ) -> DatasetDict:
        """从 CSV 文件构建数据集

        Args:
            csv_file: CSV 文件路径
            audio_column: 音频路径列名
            text_column: 文本标注列名
            split_ratios: 数据集划分比例 {'train': 0.8, 'val': 0.1, 'test': 0.1}

        Returns:
            DatasetDict 包含 train/val/test 数据集
        """
        self.logger.info(f"从 CSV 构建数据集: {csv_file}")

        # 读取 CSV
        df = pd.read_csv(csv_file)
        self.logger.info(f"数据集大小: {len(df)}")

        # 检查必需列
        if audio_column not in df.columns:
            raise ValueError(f"CSV 缺少音频路径列: {audio_column}")
        if text_column not in df.columns:
            raise ValueError(f"CSV 缺少文本列: {text_column}")

        # 构建数据字典
        data_dict = {
            'audio': df[audio_column].tolist(),
            'text': df[text_column].tolist()
        }

        # 创建 Dataset
        dataset = Dataset.from_dict(data_dict)

        # 添加音频列（自动处理采样率）
        dataset = dataset.cast_column('audio', Audio(sampling_rate=self.target_sr))

        # 划分数据集
        if split_ratios is None:
            split_ratios = {'train': 0.8, 'val': 0.1, 'test': 0.1}

        dataset_dict = self._split_dataset(dataset, split_ratios)

        self.logger.info(f"数据集划分完成:")
        for split, ds in dataset_dict.items():
            self.logger.info(f"  - {split}: {len(ds)} 样本")

        return dataset_dict

    def build_from_directory(
        self,
        audio_dir: str,
        transcript_file: str,
        split_ratios: Optional[Dict[str, float]] = None
    ) -> DatasetDict:
        """从目录构建数据集

        Args:
            audio_dir: 音频文件目录
            transcript_file: 转录文件（每行格式：audio_filename|text）
            split_ratios: 数据集划分比例

        Returns:
            DatasetDict
        """
        self.logger.info(f"从目录构建数据集: {audio_dir}")

        audio_path = Path(audio_dir)

        # 读取转录文件
        audio_files = []
        texts = []

        with open(transcript_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                parts = line.split('|')
                if len(parts) != 2:
                    self.logger.warning(f"跳过格式错误的行: {line}")
                    continue

                audio_file, text = parts
                audio_full_path = audio_path / audio_file

                if not audio_full_path.exists():
                    self.logger.warning(f"音频文件不存在: {audio_full_path}")
                    continue

                audio_files.append(str(audio_full_path))
                texts.append(text)

        self.logger.info(f"加载了 {len(audio_files)} 个音频文件")

        # 构建数据字典
        data_dict = {
            'audio': audio_files,
            'text': texts
        }

        # 创建 Dataset
        dataset = Dataset.from_dict(data_dict)
        dataset = dataset.cast_column('audio', Audio(sampling_rate=self.target_sr))

        # 划分数据集
        if split_ratios is None:
            split_ratios = {'train': 0.8, 'val': 0.1, 'test': 0.1}

        dataset_dict = self._split_dataset(dataset, split_ratios)

        return dataset_dict

    def _split_dataset(
        self,
        dataset: Dataset,
        split_ratios: Dict[str, float]
    ) -> DatasetDict:
        """划分数据集

        Args:
            dataset: 原始数据集
            split_ratios: 划分比例

        Returns:
            DatasetDict
        """
        # 验证比例
        total_ratio = sum(split_ratios.values())
        if abs(total_ratio - 1.0) > 0.01:
            raise ValueError(f"划分比例之和必须为 1.0，当前为 {total_ratio}")

        # 计算划分点
        n = len(dataset)
        train_size = int(n * split_ratios.get('train', 0.8))
        val_size = int(n * split_ratios.get('val', 0.1))

        # 划分
        train_dataset = dataset.select(range(train_size))
        val_dataset = dataset.select(range(train_size, train_size + val_size))
        test_dataset = dataset.select(range(train_size + val_size, n))

        return DatasetDict({
            'train': train_dataset,
            'validation': val_dataset,
            'test': test_dataset
        })

    def save_dataset(self, dataset_dict: DatasetDict, output_dir: str):
        """保存数据集

        Args:
            dataset_dict: 数据集字典
            output_dir: 输出目录
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        dataset_dict.save_to_disk(str(output_path))
        self.logger.info(f"数据集已保存到: {output_path}")

    def load_dataset(self, dataset_dir: str) -> DatasetDict:
        """加载数据集

        Args:
            dataset_dir: 数据集目录

        Returns:
            DatasetDict
        """
        from datasets import load_from_disk

        dataset_dict = load_from_disk(dataset_dir)
        self.logger.info(f"数据集已加载: {dataset_dir}")

        return dataset_dict


@dataclass
class DataCollatorCTCWithPadding:
    """CTC 数据整理器

    动态填充输入和标签。
    """

    processor: any
    padding: Union[bool, str] = True

    def __call__(
        self,
        features: List[Dict[str, Union[List[int], torch.Tensor]]]
    ) -> Dict[str, torch.Tensor]:
        """整理批次数据

        Args:
            features: 特征列表

        Returns:
            批次字典
        """
        # 分离输入和标签
        input_features = [{"input_values": feature["input_values"]} for feature in features]
        label_features = [{"input_ids": feature["labels"]} for feature in features]

        # 填充输入
        batch = self.processor.pad(
            input_features,
            padding=self.padding,
            return_tensors="pt",
        )

        # 填充标签
        labels_batch = self.processor.tokenizer.pad(
            label_features,
            padding=self.padding,
            return_tensors="pt",
        )

        # 将填充位置替换为 -100（忽略损失）
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )

        batch["labels"] = labels

        return batch
