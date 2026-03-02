"""
多任务方言模型

结合翻译和分类任务的多任务架构。

架构：
- 共享编码器（带 LoRA）
- 任务特定头：
  1. 翻译头（语言模型头）
  2. 分类头（分类器）
"""

import logging
from typing import Dict, Any, Optional, Tuple
import torch
import torch.nn as nn
from transformers import (
    AutoModelForCausalLM,
    AutoConfig,
    PreTrainedModel
)
from peft import (
    get_peft_model,
    LoraConfig,
    TaskType,
    PeftModel
)

logger = logging.getLogger(__name__)


class MultitaskDialectModel(nn.Module):
    """多任务方言模型"""

    def __init__(
        self,
        model_name: str,
        num_classes: int = 4,
        lora_r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.1,
        use_task_specific_lora: bool = True,
        device: str = "cuda"
    ):
        """
        初始化多任务模型

        Args:
            model_name: 预训练模型名称
            num_classes: 分类任务的类别数
            lora_r: LoRA rank
            lora_alpha: LoRA alpha
            lora_dropout: LoRA dropout
            use_task_specific_lora: 是否使用任务特定的 LoRA
            device: 设备
        """
        super().__init__()

        self.model_name = model_name
        self.num_classes = num_classes
        self.use_task_specific_lora = use_task_specific_lora
        self.device = device

        logger.info(f"Loading base model: {model_name}")

        # 加载基础模型
        try:
            # 尝试使用 safetensors 格式
            self.base_model = AutoModelForCausalLM.from_pretrained(
                model_name,
                trust_remote_code=True,
                torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                use_safetensors=True
            )
        except Exception as e:
            logger.warning(f"Failed to load with safetensors: {e}")
            logger.info("Trying to load from PR branch with safetensors...")
            # 尝试从 PR 分支加载（有 safetensors）
            self.base_model = AutoModelForCausalLM.from_pretrained(
                model_name,
                revision="refs/pr/8",
                trust_remote_code=True,
                torch_dtype=torch.float16 if device == "cuda" else torch.float32
            )

        # 获取隐藏层大小
        self.hidden_size = self.base_model.config.hidden_size

        # 配置 LoRA
        if use_task_specific_lora:
            logger.info("Using task-specific LoRA adapters")
            # 为翻译任务配置 LoRA
            self.translation_lora_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                target_modules=["c_attn", "c_proj"],  # GPT-2 模块名
                bias="none"
            )

            # 为分类任务配置 LoRA
            self.classification_lora_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                target_modules=["c_attn", "c_proj"],  # GPT-2 模块名
                bias="none"
            )

            # 应用 LoRA（先应用翻译任务的）
            self.model = get_peft_model(self.base_model, self.translation_lora_config)
        else:
            logger.info("Using shared LoRA adapter")
            # 共享 LoRA
            lora_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                target_modules=["c_attn", "c_proj"],  # GPT-2 模块名
                bias="none"
            )
            self.model = get_peft_model(self.base_model, lora_config)

        # 确保模型在正确的设备上
        self.model = self.model.to(device)

        # 分类头
        self.classification_head = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.hidden_size // 2, num_classes)
        )

        # 移动到设备并设置正确的 dtype
        if device == "cuda":
            self.classification_head = self.classification_head.to(device).half()
        else:
            self.classification_head = self.classification_head.to(device)

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
        task_id: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        classification_label: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        前向传播

        Args:
            input_ids: 输入 token IDs
            attention_mask: 注意力掩码
            task_id: 任务 ID（0=翻译，1=分类）
            labels: 翻译任务的标签
            classification_label: 分类任务的标签

        Returns:
            包含损失和输出的字典
        """
        # 检查任务类型
        task_ids = task_id.cpu().numpy() if task_id.dim() > 0 else [task_id.item()]
        is_translation = all(tid == 0 for tid in task_ids)
        is_classification = all(tid == 1 for tid in task_ids)

        if is_translation:
            # 翻译任务：使用语言模型头
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )

            return {
                'loss': outputs.loss,
                'logits': outputs.logits,
                'task_type': 'translation'
            }

        elif is_classification:
            # 分类任务：使用分类头
            # 获取最后一层隐藏状态
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True
            )

            # 使用最后一个 token 的隐藏状态
            last_hidden_state = outputs.hidden_states[-1]
            # 获取序列最后一个有效 token 的表示
            sequence_lengths = (attention_mask.sum(dim=1) - 1).long()
            batch_size = last_hidden_state.shape[0]
            batch_indices = torch.arange(batch_size, device=last_hidden_state.device, dtype=torch.long)
            pooled_output = last_hidden_state[batch_indices, sequence_lengths]

            # 分类
            classification_logits = self.classification_head(pooled_output)

            # 计算损失
            loss = None
            if classification_label is not None:
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(classification_logits, classification_label)

            return {
                'loss': loss,
                'logits': classification_logits,
                'task_type': 'classification'
            }

        else:
            # 混合批次（不推荐，但支持）
            raise ValueError(
                "Mixed task batches are not supported. "
                "Please ensure all samples in a batch have the same task_id."
            )

    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        max_length: int = 128,
        **kwargs
    ) -> torch.Tensor:
        """
        生成文本（用于翻译任务）

        Args:
            input_ids: 输入 token IDs
            attention_mask: 注意力掩码
            max_length: 最大生成长度
            **kwargs: 其他生成参数

        Returns:
            生成的 token IDs
        """
        return self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=max_length,
            **kwargs
        )

    def save_pretrained(self, save_directory: str):
        """保存模型"""
        logger.info(f"Saving model to {save_directory}")
        self.model.save_pretrained(save_directory)

        # 保存分类头
        classification_head_path = f"{save_directory}/classification_head.pt"
        torch.save(self.classification_head.state_dict(), classification_head_path)
        logger.info(f"Saved classification head to {classification_head_path}")

    @classmethod
    def from_pretrained(
        cls,
        model_path: str,
        num_classes: int = 4,
        device: str = "cuda"
    ):
        """加载预训练模型"""
        logger.info(f"Loading model from {model_path}")

        # 加载 LoRA 模型
        model = cls.__new__(cls)
        model.device = device
        model.num_classes = num_classes

        # 加载 PEFT 模型
        try:
            base_model = AutoModelForCausalLM.from_pretrained(
                model_path,
                trust_remote_code=True,
                torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                use_safetensors=True
            )
        except Exception:
            # 尝试从 PR 分支加载
            base_model = AutoModelForCausalLM.from_pretrained(
                model_path,
                revision="refs/pr/8",
                trust_remote_code=True,
                torch_dtype=torch.float16 if device == "cuda" else torch.float32
            )

        model.model = PeftModel.from_pretrained(base_model, model_path)

        model.hidden_size = model.model.config.hidden_size

        # 加载分类头
        model.classification_head = nn.Sequential(
            nn.Linear(model.hidden_size, model.hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(model.hidden_size // 2, num_classes)
        )

        classification_head_path = f"{model_path}/classification_head.pt"
        model.classification_head.load_state_dict(
            torch.load(classification_head_path, map_location=device)
        )

        model.to(device)

        logger.info("Model loaded successfully")
        return model


def create_multitask_model(
    model_name: str = "uer/gpt2-chinese-cluecorpussmall",
    num_classes: int = 4,
    lora_r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.1,
    use_task_specific_lora: bool = False,
    device: str = "cuda"
) -> MultitaskDialectModel:
    """
    创建多任务模型的便捷函数

    Args:
        model_name: 预训练模型名称
        num_classes: 分类任务的类别数
        lora_r: LoRA rank
        lora_alpha: LoRA alpha
        lora_dropout: LoRA dropout
        use_task_specific_lora: 是否使用任务特定的 LoRA
        device: 设备

    Returns:
        多任务模型
    """
    return MultitaskDialectModel(
        model_name=model_name,
        num_classes=num_classes,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        use_task_specific_lora=use_task_specific_lora,
        device=device
    )

