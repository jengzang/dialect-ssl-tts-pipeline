"""音频处理工具模块

提供音频加载、预处理和基频提取功能。
"""

import numpy as np
import librosa
import parselmouth
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging


class AudioProcessor:
    """音频处理器

    提供音频加载、重采样、基频提取等功能。
    """

    def __init__(self, config: Dict):
        """初始化音频处理器

        Args:
            config: 配置字典
        """
        self.config = config
        self.logger = logging.getLogger(__name__)

        # 获取配置参数
        self.pitch_config = config.get('feature_extraction', {}).get('pitch', {})

    def load_audio(
        self,
        audio_path: str,
        sr: Optional[int] = None
    ) -> Tuple[np.ndarray, int]:
        """加载音频文件

        Args:
            audio_path: 音频文件路径
            sr: 目标采样率，如果为 None 则使用原始采样率

        Returns:
            (audio, sample_rate)
        """
        audio, sample_rate = librosa.load(audio_path, sr=sr)
        return audio, sample_rate

    def extract_pitch_sequence(
        self,
        audio_path: str,
        time_step: Optional[float] = None,
        pitch_floor: Optional[float] = None,
        pitch_ceiling: Optional[float] = None
    ) -> np.ndarray:
        """提取基频序列

        Args:
            audio_path: 音频文件路径
            time_step: 时间步长（秒）
            pitch_floor: 基频下限（Hz）
            pitch_ceiling: 基频上限（Hz）

        Returns:
            基频序列（Hz），无声段为 0
        """
        # 使用配置或默认值
        time_step = time_step or self.pitch_config.get('time_step', 0.01)
        pitch_floor = pitch_floor or self.pitch_config.get('pitch_floor', 75)
        pitch_ceiling = pitch_ceiling or self.pitch_config.get('pitch_ceiling', 600)

        # 加载音频
        sound = parselmouth.Sound(audio_path)

        # 提取基频
        pitch = sound.to_pitch(
            time_step=time_step,
            pitch_floor=pitch_floor,
            pitch_ceiling=pitch_ceiling
        )

        # 获取基频值
        pitch_values = pitch.selected_array['frequency']

        # 将无声段（0 Hz）保留为 0
        return pitch_values

    def extract_pitch_from_segment(
        self,
        audio_path: str,
        start_time: float,
        end_time: float,
        num_points: int = 10
    ) -> np.ndarray:
        """从音频片段提取固定数量的基频点

        Args:
            audio_path: 音频文件路径
            start_time: 起始时间（秒）
            end_time: 结束时间（秒）
            num_points: 提取的基频点数量

        Returns:
            基频点数组，长度为 num_points
        """
        # 加载音频
        sound = parselmouth.Sound(audio_path)

        # 提取片段
        segment = sound.extract_part(
            from_time=start_time,
            to_time=end_time,
            preserve_times=False
        )

        # 提取基频
        pitch = segment.to_pitch(
            time_step=self.pitch_config.get('time_step', 0.01),
            pitch_floor=self.pitch_config.get('pitch_floor', 75),
            pitch_ceiling=self.pitch_config.get('pitch_ceiling', 600)
        )

        # 获取基频值
        pitch_values = pitch.selected_array['frequency']

        # 移除无声段（0 Hz）
        pitch_values = pitch_values[pitch_values > 0]

        if len(pitch_values) == 0:
            # 如果没有有效的基频点，返回全零数组
            return np.zeros(num_points)

        # 插值到固定数量的点
        if len(pitch_values) < num_points:
            # 如果点数不足，进行插值
            x_old = np.linspace(0, 1, len(pitch_values))
            x_new = np.linspace(0, 1, num_points)
            pitch_interpolated = np.interp(x_new, x_old, pitch_values)
        elif len(pitch_values) > num_points:
            # 如果点数过多，进行下采样
            indices = np.linspace(0, len(pitch_values) - 1, num_points, dtype=int)
            pitch_interpolated = pitch_values[indices]
        else:
            pitch_interpolated = pitch_values

        return pitch_interpolated

    def compute_pitch_statistics(
        self,
        pitch_values: np.ndarray
    ) -> Dict[str, float]:
        """计算基频统计特征

        Args:
            pitch_values: 基频序列

        Returns:
            统计特征字典
        """
        # 移除无声段
        valid_pitch = pitch_values[pitch_values > 0]

        if len(valid_pitch) == 0:
            return {
                'f0_mean': 0,
                'f0_std': 0,
                'f0_min': 0,
                'f0_max': 0,
                'f0_range': 0,
                'f0_median': 0,
                'f0_skew': 0,
                'f0_kurtosis': 0
            }

        from scipy import stats

        return {
            'f0_mean': float(np.mean(valid_pitch)),
            'f0_std': float(np.std(valid_pitch)),
            'f0_min': float(np.min(valid_pitch)),
            'f0_max': float(np.max(valid_pitch)),
            'f0_range': float(np.max(valid_pitch) - np.min(valid_pitch)),
            'f0_median': float(np.median(valid_pitch)),
            'f0_skew': float(stats.skew(valid_pitch)),
            'f0_kurtosis': float(stats.kurtosis(valid_pitch))
        }

    def normalize_pitch(
        self,
        pitch_values: np.ndarray,
        method: str = 'zscore'
    ) -> np.ndarray:
        """归一化基频序列

        Args:
            pitch_values: 基频序列
            method: 归一化方法 ('zscore', 'minmax')

        Returns:
            归一化后的基频序列
        """
        # 移除无声段
        valid_mask = pitch_values > 0
        valid_pitch = pitch_values[valid_mask]

        if len(valid_pitch) == 0:
            return pitch_values

        if method == 'zscore':
            mean = np.mean(valid_pitch)
            std = np.std(valid_pitch)
            if std > 0:
                pitch_values[valid_mask] = (valid_pitch - mean) / std
        elif method == 'minmax':
            min_val = np.min(valid_pitch)
            max_val = np.max(valid_pitch)
            if max_val > min_val:
                pitch_values[valid_mask] = (valid_pitch - min_val) / (max_val - min_val)

        return pitch_values

    def compute_delta_features(
        self,
        features: np.ndarray,
        order: int = 1
    ) -> np.ndarray:
        """计算差分特征

        Args:
            features: 特征序列 (time_steps, feature_dim)
            order: 差分阶数

        Returns:
            差分特征
        """
        return librosa.feature.delta(features.T, order=order).T
