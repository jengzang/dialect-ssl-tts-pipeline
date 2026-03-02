"""
方言翻译模型（基于 LoRA 微调）

使用 LoRA (Low-Rank Adaptation) 技术微调大语言模型，
实现方言到普通话的翻译任务。

支持的模型：
- ChatGLM
- Qwen
- LLaMA
"""

import torch
import torch.nn as nn
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig
)
from peft import (
    LoraConfig,
    get_peft_model,
    TaskType,
    PeftModel
)
import logging
from typing import Dict, Any, Optional, List
from pathlib import Path

logger = logging.getLogger(__name__)


class DialectTranslator:
    """方言翻译模型（LoRA 微调）"""

    def __init__(self, config: Dict[str, Any]):
        """
        初始化翻译模型

        Args:
            config: 配置字典
        """
        self.config = config
        self.model_name = config['model_name']
        self.device = config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        self.max_length = config.get('max_length', 512)

        # LoRA 配置
        self.lora_config = LoraConfig(
            r=config.get('lora_r', 8),
            lora_alpha=config.get('lora_alpha', 32),
            target_modules=config.get('target_modules', ['q_proj', 'v_proj']),
            lora_dropout=config.get('lora_dropout', 0.1),
            bias='none',
            task_type=TaskType.CAUSAL_LM
        )

        self.model = None
        self.tokenizer = None

    def load_base_model(self, quantization: bool = False):
        """
        加载基础模型

        Args:
            quantization: 是否使用量化（4-bit）
        """
        logger.info(f"Loading base model: {self.model_name}")

        # 量化配置（可选）
        if quantization:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type='nf4',
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True
            )
        else:
            bnb_config = None

        # 加载模型
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                quantization_config=bnb_config,
                device_map='auto' if quantization else None,
                trust_remote_code=True,
                torch_dtype=torch.float16 if quantization else torch.float32
            )

            if not quantization:
                self.model = self.model.to(self.device)

            logger.info("Base model loaded successfully")

        except Exception as e:
            logger.error(f"Failed to load base model: {e}")
            raise

        # 加载 tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True
        )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        logger.info("Tokenizer loaded successfully")

    def apply_lora(self):
        """应用 LoRA 适配器"""
        logger.info("Applying LoRA adapter...")

        self.model = get_peft_model(self.model, self.lora_config)
        self.model.print_trainable_parameters()

        logger.info("LoRA adapter applied")

    def translate(
        self,
        dialect_text: str,
        temperature: float = 0.7,
        top_p: float = 0.9,
        max_new_tokens: int = 256
    ) -> str:
        """
        翻译方言文本

        Args:
            dialect_text: 方言文本
            temperature: 采样温度
            top_p: nucleus sampling 参数
            max_new_tokens: 最大生成长度

        Returns:
            普通话翻译
        """
        if self.model is None or self.tokenizer is None:
            raise ValueError("Model not loaded. Call load_base_model() first.")

        # 构建 prompt
        prompt = self._build_translation_prompt(dialect_text)

        # Tokenize
        inputs = self.tokenizer(
            prompt,
            return_tensors='pt',
            max_length=self.max_length,
            truncation=True
        ).to(self.device)

        # 生成
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )

        # 解码
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # 提取翻译结果
        translation = self._extract_translation(generated_text, prompt)

        return translation

    def _build_translation_prompt(self, dialect_text: str) -> str:
        """构建翻译 prompt"""
        prompt_template = self.config.get(
            'prompt_template',
            "请将以下方言翻译成普通话：\n方言：{dialect}\n普通话："
        )

        return prompt_template.format(dialect=dialect_text)

    def _extract_translation(self, generated_text: str, prompt: str) -> str:
        """从生成文本中提取翻译结果"""
        # 移除 prompt 部分
        if generated_text.startswith(prompt):
            translation = generated_text[len(prompt):].strip()
        else:
            translation = generated_text.strip()

        # 移除可能的结束标记
        for end_marker in ['\n\n', '<|endoftext|>', '</s>']:
            if end_marker in translation:
                translation = translation.split(end_marker)[0].strip()

        return translation

    def save_model(self, output_dir: str):
        """
        保存 LoRA 模型

        Args:
            output_dir: 输出目录
        """
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True, parents=True)

        # 保存 LoRA 权重
        self.model.save_pretrained(output_path)
        self.tokenizer.save_pretrained(output_path)

        logger.info(f"Model saved to: {output_path}")

    def load_lora_model(self, lora_model_path: str):
        """
        加载 LoRA 微调后的模型

        Args:
            lora_model_path: LoRA 模型路径
        """
        logger.info(f"Loading LoRA model from: {lora_model_path}")

        # 先加载基础模型
        if self.model is None:
            self.load_base_model()

        # 加载 LoRA 权重
        self.model = PeftModel.from_pretrained(
            self.model,
            lora_model_path
        )

        logger.info("LoRA model loaded successfully")

    def batch_translate(
        self,
        dialect_texts: List[str],
        batch_size: int = 4
    ) -> List[str]:
        """
        批量翻译

        Args:
            dialect_texts: 方言文本列表
            batch_size: 批次大小

        Returns:
            翻译结果列表
        """
        translations = []

        for i in range(0, len(dialect_texts), batch_size):
            batch = dialect_texts[i:i + batch_size]
            batch_translations = [self.translate(text) for text in batch]
            translations.extend(batch_translations)

            logger.info(f"Translated {i + len(batch)}/{len(dialect_texts)}")

        return translations
