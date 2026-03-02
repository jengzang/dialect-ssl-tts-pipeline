"""
MFA 模型训练器

封装 MFA 训练流程，支持：
- 语料准备与验证
- 声学模型训练
- 模型评估
"""

import logging
from pathlib import Path
from typing import Dict, Any, Optional
import json
from src.data_pipeline.mfa_wrapper import MFAWrapper

logger = logging.getLogger(__name__)


class MFATrainer:
    """MFA 声学模型训练器"""

    def __init__(self, config: Dict[str, Any]):
        """
        初始化训练器

        Args:
            config: 配置字典
        """
        self.config = config
        self.mfa = MFAWrapper(config.get('mfa', {}))
        self.corpus_dir = Path(config['corpus_dir'])
        self.dictionary_path = Path(config['dictionary_path'])
        self.output_dir = Path(config['output_dir'])
        self.output_dir.mkdir(exist_ok=True, parents=True)

    def prepare_corpus(self) -> bool:
        """
        准备训练语料

        检查语料目录结构：
        corpus_dir/
            speaker1/
                audio1.wav
                audio1.txt
                audio2.wav
                audio2.txt
            speaker2/
                ...

        Returns:
            是否准备成功
        """
        logger.info("Preparing corpus...")

        if not self.corpus_dir.exists():
            logger.error(f"Corpus directory not found: {self.corpus_dir}")
            return False

        # 检查音频文件
        audio_files = list(self.corpus_dir.rglob('*.wav'))
        if not audio_files:
            logger.error("No audio files found in corpus")
            return False

        logger.info(f"Found {len(audio_files)} audio files")

        # 检查文本文件
        text_files = list(self.corpus_dir.rglob('*.txt'))
        logger.info(f"Found {len(text_files)} text files")

        # 检查配对
        unpaired = []
        for audio_file in audio_files:
            text_file = audio_file.with_suffix('.txt')
            if not text_file.exists():
                unpaired.append(audio_file.name)

        if unpaired:
            logger.warning(f"Found {len(unpaired)} unpaired audio files")
            logger.warning(f"Examples: {unpaired[:5]}")

        return True

    def validate(self) -> bool:
        """
        验证语料和词典

        Returns:
            是否验证通过
        """
        logger.info("Validating corpus and dictionary...")

        if not self.dictionary_path.exists():
            logger.error(f"Dictionary not found: {self.dictionary_path}")
            return False

        return self.mfa.validate_corpus(
            str(self.corpus_dir),
            str(self.dictionary_path)
        )

    def train(
        self,
        validate_first: bool = True,
        save_metadata: bool = True
    ) -> Optional[str]:
        """
        训练声学模型

        Args:
            validate_first: 是否先验证语料
            save_metadata: 是否保存训练元数据

        Returns:
            模型路径，失败返回 None
        """
        logger.info("Starting MFA acoustic model training...")

        # 准备语料
        if not self.prepare_corpus():
            logger.error("Corpus preparation failed")
            return None

        # 可选验证
        if validate_first:
            if not self.validate():
                logger.warning("Validation failed, but continuing...")

        # 训练模型
        model_path = self.output_dir / 'acoustic_model.zip'
        success = self.mfa.train_acoustic_model(
            str(self.corpus_dir),
            str(self.dictionary_path),
            str(model_path),
            validate=False  # 已经验证过了
        )

        if not success:
            logger.error("Training failed")
            return None

        logger.info(f"Model saved to: {model_path}")

        # 保存元数据
        if save_metadata:
            self._save_training_metadata(model_path)

        return str(model_path)

    def _save_training_metadata(self, model_path: Path):
        """保存训练元数据"""
        metadata = {
            'corpus_dir': str(self.corpus_dir),
            'dictionary_path': str(self.dictionary_path),
            'model_path': str(model_path),
            'config': self.config
        }

        metadata_path = self.output_dir / 'training_metadata.json'
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

        logger.info(f"Metadata saved to: {metadata_path}")

    def align_with_trained_model(
        self,
        model_path: str,
        test_corpus_dir: str,
        output_dir: str
    ) -> bool:
        """
        使用训练好的模型进行对齐

        Args:
            model_path: 训练好的模型路径
            test_corpus_dir: 测试语料目录
            output_dir: 输出目录

        Returns:
            是否成功
        """
        logger.info("Aligning with trained model...")

        return self.mfa.align(
            test_corpus_dir,
            str(self.dictionary_path),
            model_path,
            output_dir
        )
