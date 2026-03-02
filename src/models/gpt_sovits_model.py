"""
GPT-SoVITS 方言语音合成模型

封装 GPT-SoVITS 接口，实现方言语音合成功能。

GPT-SoVITS 是一个强大的少样本语音合成系统，支持：
- 零样本/少样本语音克隆
- 多语言支持
- 高质量语音合成
"""

import torch
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List
import subprocess
import json

logger = logging.getLogger(__name__)


class GPTSoVITSModel:
    """GPT-SoVITS 语音合成模型"""

    def __init__(self, config: Dict[str, Any]):
        """
        初始化模型

        Args:
            config: 配置字典
        """
        self.config = config
        self.device = config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')

        # GPT-SoVITS 路径
        self.gpt_sovits_dir = Path(config.get('gpt_sovits_dir', './GPT-SoVITS'))
        self.gpt_model_path = config.get('gpt_model_path')
        self.sovits_model_path = config.get('sovits_model_path')

        # 参考音频
        self.ref_audio_path = config.get('ref_audio_path')
        self.ref_text = config.get('ref_text')

        # 输出配置
        self.output_dir = Path(config.get('output_dir', 'results/tts'))
        self.output_dir.mkdir(exist_ok=True, parents=True)

    def check_installation(self) -> bool:
        """检查 GPT-SoVITS 是否已安装"""
        if not self.gpt_sovits_dir.exists():
            logger.error(f"GPT-SoVITS directory not found: {self.gpt_sovits_dir}")
            return False

        # 检查关键文件
        inference_script = self.gpt_sovits_dir / 'inference_webui.py'
        if not inference_script.exists():
            logger.error("GPT-SoVITS inference script not found")
            return False

        logger.info("GPT-SoVITS installation verified")
        return True

    def synthesize(
        self,
        text: str,
        output_path: Optional[str] = None,
        ref_audio: Optional[str] = None,
        ref_text: Optional[str] = None
    ) -> Optional[str]:
        """
        合成语音

        Args:
            text: 要合成的文本
            output_path: 输出音频路径
            ref_audio: 参考音频路径（可选，覆盖配置）
            ref_text: 参考文本（可选，覆盖配置）

        Returns:
            输出音频路径，失败返回 None
        """
        logger.info(f"Synthesizing: {text}")

        # 使用参数或配置中的参考音频
        ref_audio = ref_audio or self.ref_audio_path
        ref_text = ref_text or self.ref_text

        if not ref_audio or not ref_text:
            logger.error("Reference audio and text are required")
            return None

        # 生成输出路径
        if output_path is None:
            output_path = self.output_dir / f"output_{hash(text)}.wav"
        else:
            output_path = Path(output_path)

        output_path.parent.mkdir(exist_ok=True, parents=True)

        # 调用 GPT-SoVITS API
        try:
            success = self._call_gpt_sovits_api(
                text=text,
                ref_audio=ref_audio,
                ref_text=ref_text,
                output_path=str(output_path)
            )

            if success and output_path.exists():
                logger.info(f"Synthesis completed: {output_path}")
                return str(output_path)
            else:
                logger.error("Synthesis failed")
                return None

        except Exception as e:
            logger.error(f"Error during synthesis: {e}")
            return None

    def _call_gpt_sovits_api(
        self,
        text: str,
        ref_audio: str,
        ref_text: str,
        output_path: str
    ) -> bool:
        """
        调用 GPT-SoVITS API

        Args:
            text: 合成文本
            ref_audio: 参考音频
            ref_text: 参考文本
            output_path: 输出路径

        Returns:
            是否成功
        """
        # 构建 API 调用命令
        # 注意：这里假设 GPT-SoVITS 提供了命令行接口
        # 实际使用时可能需要根据 GPT-SoVITS 的具体 API 调整

        cmd = [
            'python',
            str(self.gpt_sovits_dir / 'inference_cli.py'),
            '--gpt_model', self.gpt_model_path,
            '--sovits_model', self.sovits_model_path,
            '--ref_audio', ref_audio,
            '--ref_text', ref_text,
            '--text', text,
            '--output', output_path,
            '--device', self.device
        ]

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300  # 5 minutes timeout
            )

            if result.returncode == 0:
                logger.info("API call successful")
                return True
            else:
                logger.error(f"API call failed: {result.stderr}")
                return False

        except subprocess.TimeoutExpired:
            logger.error("API call timeout")
            return False
        except Exception as e:
            logger.error(f"Error calling API: {e}")
            return False

    def batch_synthesize(
        self,
        texts: List[str],
        output_dir: Optional[str] = None
    ) -> List[Optional[str]]:
        """
        批量合成

        Args:
            texts: 文本列表
            output_dir: 输出目录

        Returns:
            输出音频路径列表
        """
        logger.info(f"Batch synthesizing {len(texts)} texts")

        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(exist_ok=True, parents=True)
        else:
            output_dir = self.output_dir

        results = []
        for i, text in enumerate(texts):
            output_path = output_dir / f"output_{i:04d}.wav"
            result = self.synthesize(text, str(output_path))
            results.append(result)

            logger.info(f"Processed {i + 1}/{len(texts)}")

        return results

    def clone_voice(
        self,
        ref_audio: str,
        ref_text: str,
        target_texts: List[str],
        output_dir: str
    ) -> List[Optional[str]]:
        """
        语音克隆

        使用参考音频克隆声音，并合成目标文本。

        Args:
            ref_audio: 参考音频路径
            ref_text: 参考文本
            target_texts: 目标文本列表
            output_dir: 输出目录

        Returns:
            输出音频路径列表
        """
        logger.info(f"Voice cloning with reference: {ref_audio}")

        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)

        results = []
        for i, text in enumerate(target_texts):
            output_path = output_dir / f"cloned_{i:04d}.wav"
            result = self.synthesize(
                text=text,
                output_path=str(output_path),
                ref_audio=ref_audio,
                ref_text=ref_text
            )
            results.append(result)

            logger.info(f"Cloned {i + 1}/{len(target_texts)}")

        return results

    def fine_tune(
        self,
        train_data_dir: str,
        output_model_dir: str,
        epochs: int = 10
    ) -> bool:
        """
        微调模型

        Args:
            train_data_dir: 训练数据目录
            output_model_dir: 输出模型目录
            epochs: 训练轮数

        Returns:
            是否成功
        """
        logger.info("Fine-tuning GPT-SoVITS model...")

        # 构建微调命令
        cmd = [
            'python',
            str(self.gpt_sovits_dir / 'train.py'),
            '--data_dir', train_data_dir,
            '--output_dir', output_model_dir,
            '--epochs', str(epochs),
            '--device', self.device
        ]

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=7200  # 2 hours timeout
            )

            if result.returncode == 0:
                logger.info("Fine-tuning completed")
                return True
            else:
                logger.error(f"Fine-tuning failed: {result.stderr}")
                return False

        except subprocess.TimeoutExpired:
            logger.error("Fine-tuning timeout")
            return False
        except Exception as e:
            logger.error(f"Error during fine-tuning: {e}")
            return False
