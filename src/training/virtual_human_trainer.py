"""
方言虚拟人训练器

整合 GPT-SoVITS 语音合成和 Sadtalker 虚拟人技术，
构建完整的方言虚拟人系统。
"""

import logging
from pathlib import Path
from typing import Dict, Any, Optional, List
import json
from tqdm import tqdm

from src.models.gpt_sovits_model import GPTSoVITSModel

logger = logging.getLogger(__name__)


class DialectVirtualHumanTrainer:
    """方言虚拟人训练器"""

    def __init__(self, config: Dict[str, Any]):
        """
        初始化训练器

        Args:
            config: 配置字典
        """
        self.config = config

        # 初始化 GPT-SoVITS 模型
        self.tts_model = GPTSoVITSModel(config.get('gpt_sovits', {}))

        # 输出目录
        self.output_dir = Path(config.get('output_dir', 'results/virtual_human'))
        self.output_dir.mkdir(exist_ok=True, parents=True)

    def prepare_training_data(
        self,
        audio_dir: str,
        transcript_file: str
    ) -> bool:
        """
        准备训练数据

        Args:
            audio_dir: 音频目录
            transcript_file: 转录文件（JSON 格式）

        Returns:
            是否成功
        """
        logger.info("Preparing training data...")

        audio_dir = Path(audio_dir)
        if not audio_dir.exists():
            logger.error(f"Audio directory not found: {audio_dir}")
            return False

        # 加载转录
        try:
            with open(transcript_file, 'r', encoding='utf-8') as f:
                transcripts = json.load(f)
        except Exception as e:
            logger.error(f"Failed to load transcripts: {e}")
            return False

        # 检查音频文件
        audio_files = list(audio_dir.glob('*.wav'))
        logger.info(f"Found {len(audio_files)} audio files")
        logger.info(f"Found {len(transcripts)} transcripts")

        # 验证配对
        missing = []
        for audio_file in audio_files:
            if audio_file.stem not in transcripts:
                missing.append(audio_file.name)

        if missing:
            logger.warning(f"Found {len(missing)} unpaired audio files")

        return True

    def train_tts_model(
        self,
        train_data_dir: str,
        epochs: int = 10
    ) -> Optional[str]:
        """
        训练 TTS 模型

        Args:
            train_data_dir: 训练数据目录
            epochs: 训练轮数

        Returns:
            模型路径，失败返回 None
        """
        logger.info("Training TTS model...")

        output_model_dir = self.output_dir / 'tts_model'
        output_model_dir.mkdir(exist_ok=True, parents=True)

        success = self.tts_model.fine_tune(
            train_data_dir=train_data_dir,
            output_model_dir=str(output_model_dir),
            epochs=epochs
        )

        if success:
            logger.info(f"TTS model saved to: {output_model_dir}")
            return str(output_model_dir)
        else:
            logger.error("TTS training failed")
            return None

    def synthesize_speech(
        self,
        text: str,
        ref_audio: str,
        ref_text: str,
        output_path: Optional[str] = None
    ) -> Optional[str]:
        """
        合成语音

        Args:
            text: 合成文本
            ref_audio: 参考音频
            ref_text: 参考文本
            output_path: 输出路径

        Returns:
            输出音频路径
        """
        logger.info(f"Synthesizing speech: {text}")

        return self.tts_model.synthesize(
            text=text,
            output_path=output_path,
            ref_audio=ref_audio,
            ref_text=ref_text
        )

    def batch_synthesize_speech(
        self,
        texts: List[str],
        ref_audio: str,
        ref_text: str,
        output_dir: Optional[str] = None
    ) -> List[Optional[str]]:
        """
        批量合成语音

        Args:
            texts: 文本列表
            ref_audio: 参考音频
            ref_text: 参考文本
            output_dir: 输出目录

        Returns:
            输出音频路径列表
        """
        logger.info(f"Batch synthesizing {len(texts)} texts")

        if output_dir is None:
            output_dir = self.output_dir / 'synthesized_speech'

        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)

        results = []
        for i, text in enumerate(tqdm(texts, desc="Synthesizing")):
            output_path = output_dir / f"speech_{i:04d}.wav"
            result = self.synthesize_speech(
                text=text,
                ref_audio=ref_audio,
                ref_text=ref_text,
                output_path=str(output_path)
            )
            results.append(result)

        logger.info(f"Synthesized {len([r for r in results if r])} / {len(texts)} speeches")
        return results

    def create_virtual_human_video(
        self,
        audio_path: str,
        image_path: str,
        output_path: str,
        sadtalker_dir: Optional[str] = None
    ) -> bool:
        """
        创建虚拟人视频

        使用 Sadtalker 从音频和图像生成虚拟人视频。

        Args:
            audio_path: 音频路径
            image_path: 人物图像路径
            output_path: 输出视频路径
            sadtalker_dir: Sadtalker 目录

        Returns:
            是否成功
        """
        logger.info("Creating virtual human video...")

        # 注意：这里需要 Sadtalker 的实际实现
        # 以下是示例代码框架

        if sadtalker_dir is None:
            sadtalker_dir = self.config.get('sadtalker_dir', './SadTalker')

        sadtalker_dir = Path(sadtalker_dir)
        if not sadtalker_dir.exists():
            logger.error(f"Sadtalker directory not found: {sadtalker_dir}")
            return False

        # 调用 Sadtalker
        import subprocess

        cmd = [
            'python',
            str(sadtalker_dir / 'inference.py'),
            '--driven_audio', audio_path,
            '--source_image', image_path,
            '--result_dir', str(Path(output_path).parent),
            '--still',
            '--preprocess', 'full'
        ]

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=600  # 10 minutes
            )

            if result.returncode == 0:
                logger.info(f"Video created: {output_path}")
                return True
            else:
                logger.error(f"Video creation failed: {result.stderr}")
                return False

        except subprocess.TimeoutExpired:
            logger.error("Video creation timeout")
            return False
        except Exception as e:
            logger.error(f"Error creating video: {e}")
            return False

    def create_dialect_virtual_human(
        self,
        dialect_text: str,
        ref_audio: str,
        ref_text: str,
        avatar_image: str,
        output_video_path: str
    ) -> bool:
        """
        创建方言虚拟人

        完整流程：文本 -> 语音合成 -> 虚拟人视频

        Args:
            dialect_text: 方言文本
            ref_audio: 参考音频
            ref_text: 参考文本
            avatar_image: 虚拟人图像
            output_video_path: 输出视频路径

        Returns:
            是否成功
        """
        logger.info("Creating dialect virtual human...")

        # 步骤 1: 合成语音
        audio_dir = self.output_dir / 'temp_audio'
        audio_dir.mkdir(exist_ok=True, parents=True)
        audio_path = audio_dir / 'synthesized.wav'

        synthesized_audio = self.synthesize_speech(
            text=dialect_text,
            ref_audio=ref_audio,
            ref_text=ref_text,
            output_path=str(audio_path)
        )

        if not synthesized_audio:
            logger.error("Speech synthesis failed")
            return False

        # 步骤 2: 创建虚拟人视频
        success = self.create_virtual_human_video(
            audio_path=synthesized_audio,
            image_path=avatar_image,
            output_path=output_video_path
        )

        if success:
            logger.info(f"Dialect virtual human created: {output_video_path}")
        else:
            logger.error("Virtual human creation failed")

        return success

    def batch_create_virtual_humans(
        self,
        texts: List[str],
        ref_audio: str,
        ref_text: str,
        avatar_image: str,
        output_dir: str
    ) -> List[bool]:
        """
        批量创建虚拟人

        Args:
            texts: 文本列表
            ref_audio: 参考音频
            ref_text: 参考文本
            avatar_image: 虚拟人图像
            output_dir: 输出目录

        Returns:
            成功状态列表
        """
        logger.info(f"Batch creating {len(texts)} virtual humans")

        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)

        results = []
        for i, text in enumerate(tqdm(texts, desc="Creating virtual humans")):
            output_video = output_dir / f"virtual_human_{i:04d}.mp4"
            success = self.create_dialect_virtual_human(
                dialect_text=text,
                ref_audio=ref_audio,
                ref_text=ref_text,
                avatar_image=avatar_image,
                output_video_path=str(output_video)
            )
            results.append(success)

        success_count = sum(results)
        logger.info(f"Created {success_count} / {len(texts)} virtual humans")

        return results
