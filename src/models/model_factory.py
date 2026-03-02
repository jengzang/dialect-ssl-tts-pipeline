"""
模型工厂

统一的模型加载和创建接口，支持多种模型架构。

支持的模型：
- GPT-2 Chinese (不同大小)
- 自定义配置的 GPT-2
- 其他 HuggingFace 模型
"""

import logging
from typing import Dict, Any, Optional, List
import torch
import torch.nn as nn
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoConfig,
    GPT2Config
)
from peft import (
    get_peft_model,
    LoraConfig,
    TaskType
)

logger = logging.getLogger(__name__)


# 预定义模型配置
MODEL_CONFIGS = {
    "gpt2-chinese-small": {
        "model_name": "uer/gpt2-chinese-cluecorpussmall",
        "params": "103M",
        "hidden_size": 768,
        "num_layers": 12,
        "num_heads": 12
    },
    "gpt2-chinese-base": {
        "model_name": "uer/gpt2-chinese-poem",
        "params": "117M",
        "hidden_size": 768,
        "num_layers": 12,
        "num_heads": 12
    },
    "gpt2-custom-tiny": {
        "model_name": None,  # 自定义配置
        "params": "25M",
        "hidden_size": 384,
        "num_layers": 6,
        "num_heads": 6
    },
    "gpt2-custom-small": {
        "model_name": None,
        "params": "50M",
        "hidden_size": 512,
        "num_layers": 8,
        "num_heads": 8
    }
}


class ModelFactory:
    """模型工厂"""

    def __init__(self, device: str = "cuda"):
        """
        初始化模型工厂

        Args:
            device: 设备
        """
        self.device = device
        logger.info(f"ModelFactory initialized with device: {device}")

    def create_model(
        self,
        model_key: str,
        use_lora: bool = True,
        lora_r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.1,
        vocab_size: int = 21128
    ) -> nn.Module:
        """
        创建模型

        Args:
            model_key: 模型键（在 MODEL_CONFIGS 中）
            use_lora: 是否使用 LoRA
            lora_r: LoRA rank
            lora_alpha: LoRA alpha
            lora_dropout: LoRA dropout
            vocab_size: 词表大小

        Returns:
            模型
        """
        if model_key not in MODEL_CONFIGS:
            raise ValueError(f"Unknown model key: {model_key}")

        config = MODEL_CONFIGS[model_key]
        logger.info(f"Creating model: {model_key} ({config['params']} parameters)")

        # 加载或创建模型
        if config["model_name"]:
            # 从预训练模型加载
            model = self._load_pretrained_model(config["model_name"])
        else:
            # 创建自定义配置的模型
            model = self._create_custom_model(
                hidden_size=config["hidden_size"],
                num_layers=config["num_layers"],
                num_heads=config["num_heads"],
                vocab_size=vocab_size
            )

        # 应用 LoRA
        if use_lora:
            model = self._apply_lora(
                model,
                lora_r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout
            )

        # 移动到设备
        model = model.to(self.device)

        # 统计参数
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        logger.info(f"Model created:")
        logger.info(f"  Total parameters: {total_params:,}")
        logger.info(f"  Trainable parameters: {trainable_params:,}")
        logger.info(f"  Trainable ratio: {trainable_params/total_params*100:.2f}%")

        return model

    def _load_pretrained_model(self, model_name: str) -> nn.Module:
        """加载预训练模型"""
        try:
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                trust_remote_code=True,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                use_safetensors=True
            )
        except Exception as e:
            logger.warning(f"Failed to load with safetensors: {e}")
            logger.info("Trying to load from PR branch...")
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                revision="refs/pr/8",
                trust_remote_code=True,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
            )

        return model

    def _create_custom_model(
        self,
        hidden_size: int,
        num_layers: int,
        num_heads: int,
        vocab_size: int
    ) -> nn.Module:
        """创建自定义配置的模型"""
        config = GPT2Config(
            vocab_size=vocab_size,
            n_positions=512,
            n_embd=hidden_size,
            n_layer=num_layers,
            n_head=num_heads,
            n_inner=hidden_size * 4,
            activation_function="gelu_new",
            resid_pdrop=0.1,
            embd_pdrop=0.1,
            attn_pdrop=0.1,
            layer_norm_epsilon=1e-5,
            initializer_range=0.02
        )

        model = AutoModelForCausalLM.from_config(config)
        logger.info(f"Created custom GPT-2 model with {num_layers} layers, {hidden_size} hidden size")

        return model

    def _apply_lora(
        self,
        model: nn.Module,
        lora_r: int,
        lora_alpha: int,
        lora_dropout: float
    ) -> nn.Module:
        """应用 LoRA"""
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=["c_attn", "c_proj"],
            bias="none"
        )

        model = get_peft_model(model, lora_config)
        logger.info(f"Applied LoRA with r={lora_r}, alpha={lora_alpha}")

        return model

    def load_tokenizer(self, model_key: str) -> AutoTokenizer:
        """
        加载 tokenizer

        Args:
            model_key: 模型键

        Returns:
            Tokenizer
        """
        config = MODEL_CONFIGS[model_key]

        if config["model_name"]:
            tokenizer = AutoTokenizer.from_pretrained(
                config["model_name"],
                trust_remote_code=True
            )
        else:
            # 使用默认的 GPT-2 Chinese tokenizer
            tokenizer = AutoTokenizer.from_pretrained(
                "uer/gpt2-chinese-cluecorpussmall",
                trust_remote_code=True
            )

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        return tokenizer

    def get_model_info(self, model_key: str) -> Dict[str, Any]:
        """
        获取模型信息

        Args:
            model_key: 模型键

        Returns:
            模型信息
        """
        if model_key not in MODEL_CONFIGS:
            raise ValueError(f"Unknown model key: {model_key}")

        return MODEL_CONFIGS[model_key].copy()

    def list_available_models(self) -> List[str]:
        """列出可用的模型"""
        return list(MODEL_CONFIGS.keys())


def create_model_comparison_suite(
    model_keys: List[str],
    device: str = "cuda",
    use_lora: bool = True,
    lora_r: int = 8
) -> Dict[str, nn.Module]:
    """
    创建模型比较套件

    Args:
        model_keys: 模型键列表
        device: 设备
        use_lora: 是否使用 LoRA
        lora_r: LoRA rank

    Returns:
        模型字典
    """
    factory = ModelFactory(device=device)
    models = {}

    for key in model_keys:
        logger.info(f"\nCreating model: {key}")
        model = factory.create_model(
            model_key=key,
            use_lora=use_lora,
            lora_r=lora_r
        )
        models[key] = model

    logger.info(f"\nCreated {len(models)} models for comparison")
    return models


def get_model_size_mb(model: nn.Module) -> float:
    """
    获取模型大小（MB）

    Args:
        model: 模型

    Returns:
        模型大小（MB）
    """
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()

    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    size_mb = (param_size + buffer_size) / 1024 / 1024
    return size_mb
