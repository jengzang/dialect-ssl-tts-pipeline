"""
指令遵循模型

基于 LoRA 微调的指令遵循模型，支持多种方言任务。

特性：
- 指令格式化
- 少样本提示
- 思维链推理
"""

import logging
from typing import Dict, Any, List, Optional
import torch
import torch.nn as nn
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel
)
from peft import (
    get_peft_model,
    LoraConfig,
    TaskType,
    PeftModel
)

logger = logging.getLogger(__name__)


class InstructionTunedModel(nn.Module):
    """指令遵循模型"""

    def __init__(
        self,
        model_name: str,
        lora_r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.1,
        device: str = "cuda"
    ):
        """
        初始化指令遵循模型

        Args:
            model_name: 预训练模型名称
            lora_r: LoRA rank
            lora_alpha: LoRA alpha
            lora_dropout: LoRA dropout
            device: 设备
        """
        super().__init__()

        self.model_name = model_name
        self.device = device

        logger.info(f"Loading base model: {model_name}")

        # 加载基础模型
        try:
            self.base_model = AutoModelForCausalLM.from_pretrained(
                model_name,
                trust_remote_code=True,
                torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                use_safetensors=True
            )
        except Exception as e:
            logger.warning(f"Failed to load with safetensors: {e}")
            logger.info("Trying to load from PR branch with safetensors...")
            self.base_model = AutoModelForCausalLM.from_pretrained(
                model_name,
                revision="refs/pr/8",
                trust_remote_code=True,
                torch_dtype=torch.float16 if device == "cuda" else torch.float32
            )

        # 配置 LoRA
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=["c_attn", "c_proj"],  # GPT-2 模块名
            bias="none"
        )

        # 应用 LoRA
        self.model = get_peft_model(self.base_model, lora_config)
        self.model = self.model.to(device)

        logger.info(f"Model initialized with {self.count_parameters()} parameters")
        logger.info(f"Trainable parameters: {self.count_trainable_parameters()}")

    def count_parameters(self) -> int:
        """计算总参数数"""
        return sum(p.numel() for p in self.parameters())

    def count_trainable_parameters(self) -> int:
        """计算可训练参数数"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        前向传播

        Args:
            input_ids: 输入 token IDs
            attention_mask: 注意力掩码
            labels: 标签

        Returns:
            包含损失和输出的字典
        """
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )

        return {
            'loss': outputs.loss,
            'logits': outputs.logits
        }

    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        max_length: int = 256,
        num_beams: int = 4,
        temperature: float = 0.7,
        top_p: float = 0.9,
        **kwargs
    ) -> torch.Tensor:
        """
        生成文本

        Args:
            input_ids: 输入 token IDs
            attention_mask: 注意力掩码
            max_length: 最大生成长度
            num_beams: beam search 数量
            temperature: 温度参数
            top_p: nucleus sampling 参数
            **kwargs: 其他生成参数

        Returns:
            生成的 token IDs
        """
        return self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=max_length,
            num_beams=num_beams,
            temperature=temperature,
            top_p=top_p,
            early_stopping=True,
            **kwargs
        )

    def generate_with_few_shot(
        self,
        tokenizer,
        instruction: str,
        examples: List[Dict[str, str]],
        input_text: str,
        max_length: int = 256
    ) -> str:
        """
        使用少样本提示生成

        Args:
            tokenizer: Tokenizer
            instruction: 指令
            examples: 少样本示例列表
            input_text: 输入文本
            max_length: 最大长度

        Returns:
            生成的文本
        """
        # 构建少样本提示
        prompt = f"{instruction}\n\n"

        for i, example in enumerate(examples, 1):
            prompt += f"示例 {i}:\n"
            prompt += f"输入：{example['input']}\n"
            prompt += f"输出：{example['output']}\n\n"

        prompt += f"现在请处理：\n输入：{input_text}\n输出："

        # Tokenize
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            max_length=max_length,
            truncation=True
        ).to(self.device)

        # 生成
        with torch.no_grad():
            outputs = self.generate(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                max_length=max_length
            )

        # 解码
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # 提取输出部分
        if "输出：" in generated_text:
            generated_text = generated_text.split("输出：")[-1].strip()

        return generated_text

    def generate_with_chain_of_thought(
        self,
        tokenizer,
        instruction: str,
        input_text: str,
        max_length: int = 512
    ) -> str:
        """
        使用思维链推理生成

        Args:
            tokenizer: Tokenizer
            instruction: 指令
            input_text: 输入文本
            max_length: 最大长度

        Returns:
            生成的文本
        """
        # 构建思维链提示
        prompt = f"{instruction}\n\n"
        prompt += "请一步步思考并给出答案。\n\n"
        prompt += f"输入：{input_text}\n\n"
        prompt += "思考过程：\n"

        # Tokenize
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            max_length=max_length,
            truncation=True
        ).to(self.device)

        # 生成
        with torch.no_grad():
            outputs = self.generate(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                max_length=max_length,
                temperature=0.8  # 稍高的温度以鼓励推理
            )

        # 解码
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

        return generated_text

    def save_pretrained(self, save_directory: str):
        """保存模型"""
        logger.info(f"Saving model to {save_directory}")
        self.model.save_pretrained(save_directory)

    @classmethod
    def from_pretrained(
        cls,
        model_path: str,
        device: str = "cuda"
    ):
        """加载预训练模型"""
        logger.info(f"Loading model from {model_path}")

        # 创建新实例
        model = cls.__new__(cls)
        model.device = device

        # 加载 PEFT 模型
        try:
            base_model = AutoModelForCausalLM.from_pretrained(
                model_path,
                trust_remote_code=True,
                torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                use_safetensors=True
            )
        except Exception:
            base_model = AutoModelForCausalLM.from_pretrained(
                model_path,
                revision="refs/pr/8",
                trust_remote_code=True,
                torch_dtype=torch.float16 if device == "cuda" else torch.float32
            )

        model.model = PeftModel.from_pretrained(base_model, model_path)
        model.model = model.model.to(device)

        logger.info("Model loaded successfully")
        return model


def create_instruction_tuned_model(
    model_name: str = "uer/gpt2-chinese-cluecorpussmall",
    lora_r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.1,
    device: str = "cuda"
) -> InstructionTunedModel:
    """
    创建指令遵循模型的便捷函数

    Args:
        model_name: 预训练模型名称
        lora_r: LoRA rank
        lora_alpha: LoRA alpha
        lora_dropout: LoRA dropout
        device: 设备

    Returns:
        指令遵循模型
    """
    return InstructionTunedModel(
        model_name=model_name,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        device=device
    )
