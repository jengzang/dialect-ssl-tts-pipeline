"""特征提取模块

提供基于 Praat 的语音特征提取功能。
"""

import parselmouth
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import re
import logging
from tqdm import tqdm

from .base import BaseFeatureExtractor


class PraatFeatureExtractor(BaseFeatureExtractor):
    """基于 Praat 的特征提取器

    提取语音特征，包括：
    - 共振峰 (F1, F2, F3)
    - 基频统计值 (mean, std, min, max)
    - 时长
    """

    def __init__(self, config: Dict):
        """初始化特征提取器

        Args:
            config: 配置字典
        """
        super().__init__(config)
        self.logger = logging.getLogger(__name__)

        # 获取 Praat 参数
        self.pitch_config = config.get('feature_extraction', {}).get('pitch', {})
        self.formant_config = config.get('feature_extraction', {}).get('formants', {})

    def extract(
        self,
        audio_path: str,
        textgrid_path: str,
        **kwargs
    ) -> List[Dict[str, any]]:
        """从音频和 TextGrid 提取特征

        Args:
            audio_path: 音频文件路径
            textgrid_path: TextGrid 文件路径
            **kwargs: 其他参数

        Returns:
            特征字典列表
        """
        # 加载音频和 TextGrid
        sound = parselmouth.Sound(audio_path)
        textgrid = parselmouth.praat.call("Read from file", textgrid_path)

        # 提取基频和共振峰
        pitch = sound.to_pitch(
            time_step=self.pitch_config.get('time_step', 0.01),
            pitch_floor=self.pitch_config.get('pitch_floor', 75),
            pitch_ceiling=self.pitch_config.get('pitch_ceiling', 600)
        )

        formants = sound.to_formant_burg(
            max_formant=self.formant_config.get('max_formant', 5500)
        )

        # 获取音素层的区间数量（假设音素层在第 2 层）
        num_intervals = parselmouth.praat.call(textgrid, "Get number of intervals", 2)

        results = []
        file_name = Path(audio_path).name

        # 遍历每个音素
        for interval in range(1, num_intervals + 1):
            # 获取音素标签
            phoneme = parselmouth.praat.call(
                textgrid, "Get label of interval", 2, interval
            ).strip()

            # 移除声调标记（1-6）
            phoneme = re.sub(r'[1-6]$', '', phoneme)

            # 跳过空标签
            if not phoneme:
                continue

            # 获取时间边界
            start_time = parselmouth.praat.call(
                textgrid, "Get start point", 2, interval
            )
            end_time = parselmouth.praat.call(
                textgrid, "Get end point", 2, interval
            )
            duration = end_time - start_time

            # 获取对应的词
            interval_word = parselmouth.praat.call(
                textgrid, "Get interval at time", 1, start_time
            )
            word = parselmouth.praat.call(
                textgrid, "Get label of interval", 1, interval_word
            ).strip()

            # 提取基频统计值
            try:
                min_pitch = parselmouth.praat.call(
                    pitch, "Get minimum", start_time, end_time, "Hertz", "Parabolic"
                )
                max_pitch = parselmouth.praat.call(
                    pitch, "Get maximum", start_time, end_time, "Hertz", "Parabolic"
                )
                mean_pitch = parselmouth.praat.call(
                    pitch, "Get mean", start_time, end_time, "Hertz"
                )
                std_pitch = parselmouth.praat.call(
                    pitch, "Get standard deviation", start_time, end_time, "Hertz"
                )
            except Exception as e:
                self.logger.warning(f"提取基频失败: {file_name}, {phoneme}, {e}")
                min_pitch = max_pitch = mean_pitch = std_pitch = 0

            # 提取共振峰
            try:
                f1 = parselmouth.praat.call(
                    formants, "Get mean", 1, start_time, end_time, "hertz"
                )
                f2 = parselmouth.praat.call(
                    formants, "Get mean", 2, start_time, end_time, "hertz"
                )
                f3 = parselmouth.praat.call(
                    formants, "Get mean", 3, start_time, end_time, "hertz"
                )
            except Exception as e:
                self.logger.warning(f"提取共振峰失败: {file_name}, {phoneme}, {e}")
                f1 = f2 = f3 = 0

            # 构建特征字典
            results.append({
                "file_name": file_name,
                "word": word,
                "phoneme": phoneme,
                "duration": duration,
                "f1": f1,
                "f2": f2,
                "f3": f3,
                "mean_f0": mean_pitch,
                "std_f0": std_pitch,
                "min_f0": min_pitch,
                "max_f0": max_pitch
            })

        return results

    def extract_batch(
        self,
        audio_dir: str,
        textgrid_dir: str,
        output_file: str,
        target_phonemes: Optional[List[str]] = None,
        audio_ext: str = ".mp3"
    ):
        """批量提取特征

        Args:
            audio_dir: 音频目录
            textgrid_dir: TextGrid 目录
            output_file: 输出 CSV 文件路径
            target_phonemes: 目标音素列表（如果为 None，则提取所有音素）
            audio_ext: 音频文件扩展名
        """
        audio_path = Path(audio_dir)
        textgrid_path = Path(textgrid_dir)
        all_results = []

        # 获取所有音频文件
        audio_files = list(audio_path.glob(f"*{audio_ext}"))

        self.logger.info(f"找到 {len(audio_files)} 个音频文件")

        # 遍历音频文件
        for audio_file in tqdm(audio_files, desc="提取特征"):
            # 构建对应的 TextGrid 文件路径
            tg_file = textgrid_path / audio_file.name.replace(audio_ext, ".TextGrid")

            if not tg_file.exists():
                self.logger.warning(f"TextGrid 文件不存在: {tg_file}")
                continue

            try:
                # 提取特征
                results = self.extract(str(audio_file), str(tg_file))
                all_results.extend(results)
            except Exception as e:
                self.logger.error(f"处理文件失败: {audio_file}, {e}")

        # 转换为 DataFrame
        df = pd.DataFrame(all_results)

        # 过滤目标音素
        if target_phonemes is not None:
            df = df[df['phoneme'].isin(target_phonemes)]
            self.logger.info(f"过滤后保留 {len(df)} 条记录")

        # 保存到 CSV
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)

        self.logger.info(f"特征已保存到: {output_path}")
        self.logger.info(f"总共提取 {len(df)} 条记录")

        return df
