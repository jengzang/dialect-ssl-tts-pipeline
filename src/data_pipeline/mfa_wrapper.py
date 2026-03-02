"""
Montreal Forced Aligner (MFA) 封装模块

提供 MFA 命令行工具的 Python 接口，支持：
- 语音对齐 (align)
- 声学模型训练 (train)
- 发音词典验证 (validate)
"""

import subprocess
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any
import shutil

logger = logging.getLogger(__name__)


class MFAWrapper:
    """MFA 命令行工具封装类"""

    def __init__(self, config: Dict[str, Any]):
        """
        初始化 MFA 封装器

        Args:
            config: 配置字典，包含 MFA 相关参数
        """
        self.config = config
        self.mfa_command = config.get('mfa_command', 'mfa')
        self.num_jobs = config.get('num_jobs', 4)
        self.temp_dir = Path(config.get('temp_dir', './mfa_temp'))
        self.temp_dir.mkdir(exist_ok=True, parents=True)

    def check_mfa_installed(self) -> bool:
        """检查 MFA 是否已安装"""
        try:
            result = subprocess.run(
                [self.mfa_command, '--version'],
                capture_output=True,
                text=True,
                timeout=10
            )
            if result.returncode == 0:
                logger.info(f"MFA version: {result.stdout.strip()}")
                return True
            else:
                logger.error("MFA command failed")
                return False
        except FileNotFoundError:
            logger.error(f"MFA command '{self.mfa_command}' not found")
            return False
        except Exception as e:
            logger.error(f"Error checking MFA installation: {e}")
            return False

    def align(
        self,
        corpus_dir: str,
        dictionary_path: str,
        acoustic_model_path: str,
        output_dir: str,
        clean: bool = True
    ) -> bool:
        """
        执行强制对齐

        Args:
            corpus_dir: 语料目录（包含音频和文本文件）
            dictionary_path: 发音词典路径
            acoustic_model_path: 声学模型路径
            output_dir: 输出目录（TextGrid 文件）
            clean: 是否清理临时文件

        Returns:
            是否成功
        """
        logger.info("Starting MFA alignment...")
        logger.info(f"Corpus: {corpus_dir}")
        logger.info(f"Dictionary: {dictionary_path}")
        logger.info(f"Acoustic model: {acoustic_model_path}")
        logger.info(f"Output: {output_dir}")

        # 构建命令
        cmd = [
            self.mfa_command,
            'align',
            corpus_dir,
            dictionary_path,
            acoustic_model_path,
            output_dir,
            '--num_jobs', str(self.num_jobs),
            '--clean' if clean else '--no_clean'
        ]

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=3600  # 1 hour timeout
            )

            if result.returncode == 0:
                logger.info("Alignment completed successfully")
                logger.info(result.stdout)
                return True
            else:
                logger.error("Alignment failed")
                logger.error(result.stderr)
                return False

        except subprocess.TimeoutExpired:
            logger.error("Alignment timeout (>1 hour)")
            return False
        except Exception as e:
            logger.error(f"Error during alignment: {e}")
            return False

    def train_acoustic_model(
        self,
        corpus_dir: str,
        dictionary_path: str,
        output_model_path: str,
        validate: bool = True
    ) -> bool:
        """
        训练声学模型

        Args:
            corpus_dir: 训练语料目录
            dictionary_path: 发音词典路径
            output_model_path: 输出模型路径
            validate: 是否先验证语料

        Returns:
            是否成功
        """
        logger.info("Starting acoustic model training...")
        logger.info(f"Corpus: {corpus_dir}")
        logger.info(f"Dictionary: {dictionary_path}")
        logger.info(f"Output model: {output_model_path}")

        # 可选：先验证语料
        if validate:
            logger.info("Validating corpus...")
            if not self.validate_corpus(corpus_dir, dictionary_path):
                logger.warning("Corpus validation failed, but continuing...")

        # 构建训练命令
        cmd = [
            self.mfa_command,
            'train',
            corpus_dir,
            dictionary_path,
            output_model_path,
            '--num_jobs', str(self.num_jobs)
        ]

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=7200  # 2 hours timeout
            )

            if result.returncode == 0:
                logger.info("Training completed successfully")
                logger.info(result.stdout)
                return True
            else:
                logger.error("Training failed")
                logger.error(result.stderr)
                return False

        except subprocess.TimeoutExpired:
            logger.error("Training timeout (>2 hours)")
            return False
        except Exception as e:
            logger.error(f"Error during training: {e}")
            return False

    def validate_corpus(
        self,
        corpus_dir: str,
        dictionary_path: str
    ) -> bool:
        """
        验证语料和词典

        Args:
            corpus_dir: 语料目录
            dictionary_path: 发音词典路径

        Returns:
            是否验证通过
        """
        logger.info("Validating corpus and dictionary...")

        cmd = [
            self.mfa_command,
            'validate',
            corpus_dir,
            dictionary_path,
            '--num_jobs', str(self.num_jobs)
        ]

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=600  # 10 minutes timeout
            )

            if result.returncode == 0:
                logger.info("Validation passed")
                logger.info(result.stdout)
                return True
            else:
                logger.warning("Validation found issues")
                logger.warning(result.stderr)
                return False

        except subprocess.TimeoutExpired:
            logger.error("Validation timeout")
            return False
        except Exception as e:
            logger.error(f"Error during validation: {e}")
            return False

    def download_model(
        self,
        model_name: str,
        model_type: str = 'acoustic'
    ) -> Optional[str]:
        """
        下载预训练模型

        Args:
            model_name: 模型名称（如 'english_us_arpa'）
            model_type: 模型类型（'acoustic' 或 'dictionary'）

        Returns:
            模型路径，失败返回 None
        """
        logger.info(f"Downloading {model_type} model: {model_name}")

        cmd = [
            self.mfa_command,
            'model',
            'download',
            model_type,
            model_name
        ]

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=600
            )

            if result.returncode == 0:
                logger.info(f"Model downloaded: {model_name}")
                # 提取模型路径（从输出中解析）
                return model_name
            else:
                logger.error(f"Failed to download model: {model_name}")
                logger.error(result.stderr)
                return None

        except Exception as e:
            logger.error(f"Error downloading model: {e}")
            return None

    def list_available_models(self, model_type: str = 'acoustic') -> List[str]:
        """
        列出可用的预训练模型

        Args:
            model_type: 模型类型（'acoustic' 或 'dictionary'）

        Returns:
            模型名称列表
        """
        cmd = [
            self.mfa_command,
            'model',
            'list',
            model_type
        ]

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=30
            )

            if result.returncode == 0:
                # 解析输出获取模型列表
                models = [line.strip() for line in result.stdout.split('\n') if line.strip()]
                return models
            else:
                logger.error("Failed to list models")
                return []

        except Exception as e:
            logger.error(f"Error listing models: {e}")
            return []

    def cleanup_temp_files(self):
        """清理临时文件"""
        if self.temp_dir.exists():
            try:
                shutil.rmtree(self.temp_dir)
                logger.info(f"Cleaned up temp directory: {self.temp_dir}")
            except Exception as e:
                logger.error(f"Error cleaning temp directory: {e}")
