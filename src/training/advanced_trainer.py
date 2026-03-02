#!/usr/bin/env python3
"""
高级微调训练器

支持多种参数高效微调方法：
- LoRA (Low-Rank Adaptation)
- Prefix Tuning
- Adapter Layers
- Full Fine-tuning
"""

import logging
from pathlib import Path
from typing import Optional, Dict, Any, List
import time
import json

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    get_linear_schedule_with_warmup
)
from peft import LoraConfig, get_peft_model, TaskType

logger = logging.getLogger(__name__)


class AdvancedFinetuner:
    """
    高级微调训练器

    支持多种微调方法的统一接口
    """

    def __init__(
        self,
        model_name: str,
        method: str = "lora",
        method_config: Optional[Dict[str, Any]] = None,
        device: str = "cuda",
        mixed_precision: bool = True
    ):
        """
        Args:
            model_name: 预训练模型名称
            method: 微调方法 (lora, prefix, adapter, full)
            method_config: 方法特定配置
            device: 设备
            mixed_precision: 是否使用混合精度训练
        """
        self.model_name = model_name
        self.method = method
        self.method_config = method_config or {}
        self.device = device
        self.mixed_precision = mixed_precision

        logger.info(f"Initializing AdvancedFinetuner:")
        logger.info(f"  Model: {model_name}")
        logger.info(f"  Method: {method}")
        logger.info(f"  Device: {device}")
        logger.info(f"  Mixed precision: {mixed_precision}")

        # 加载模型和 tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True
        )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # 根据方法创建模型
        self.model = self._create_model()

        # 混合精度训练
        self.scaler = torch.cuda.amp.GradScaler() if mixed_precision and device == "cuda" else None

        # 训练统计
        self.training_stats = {
            "method": method,
            "model_name": model_name,
            "total_params": self._count_parameters(),
            "trainable_params": self._count_trainable_parameters(),
            "training_time": 0,
            "peak_memory_mb": 0,
            "epochs": 0,
            "final_loss": 0
        }

        logger.info(f"Model statistics:")
        logger.info(f"  Total parameters: {self.training_stats['total_params']:,}")
        logger.info(f"  Trainable parameters: {self.training_stats['trainable_params']:,}")
        logger.info(f"  Trainable ratio: {self.training_stats['trainable_params'] / self.training_stats['total_params'] * 100:.2f}%")

    def _create_model(self) -> nn.Module:
        """根据方法创建模型"""
        if self.method == "lora":
            return self._create_lora_model()
        elif self.method == "prefix":
            return self._create_prefix_model()
        elif self.method == "adapter":
            return self._create_adapter_model()
        elif self.method == "full":
            return self._create_full_model()
        else:
            raise ValueError(f"Unknown method: {self.method}")

    def _create_lora_model(self) -> nn.Module:
        """创建 LoRA 模型"""
        logger.info("Creating LoRA model...")

        # 加载基础模型
        model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
        )

        # LoRA 配置
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=self.method_config.get("r", 8),
            lora_alpha=self.method_config.get("alpha", 16),
            lora_dropout=self.method_config.get("dropout", 0.1),
            target_modules=self.method_config.get("target_modules", ["c_attn", "c_proj"]),
            bias="none"
        )

        # 应用 LoRA
        model = get_peft_model(model, lora_config)
        model = model.to(self.device)

        logger.info(f"LoRA applied: r={lora_config.r}, alpha={lora_config.lora_alpha}")

        return model

    def _create_prefix_model(self) -> nn.Module:
        """创建 Prefix Tuning 模型"""
        logger.info("Creating Prefix Tuning model...")

        from src.models.prefix_tuning_model import create_prefix_tuning_model

        model = create_prefix_tuning_model(
            model_name=self.model_name,
            prefix_length=self.method_config.get("prefix_length", 10),
            prefix_hidden_size=self.method_config.get("prefix_hidden_size", 512),
            device=self.device
        )

        return model

    def _create_adapter_model(self) -> nn.Module:
        """创建 Adapter 模型"""
        logger.info("Creating Adapter model...")

        from src.models.adapter_model import create_adapter_model

        model = create_adapter_model(
            model_name=self.model_name,
            adapter_size=self.method_config.get("adapter_size", 64),
            adapter_activation=self.method_config.get("adapter_activation", "gelu"),
            device=self.device
        )

        return model

    def _create_full_model(self) -> nn.Module:
        """创建全量微调模型"""
        logger.info("Creating full fine-tuning model...")

        model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
        )

        model = model.to(self.device)

        return model

    def _count_parameters(self) -> int:
        """统计总参数数"""
        return sum(p.numel() for p in self.model.parameters())

    def _count_trainable_parameters(self) -> int:
        """统计可训练参数数"""
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)

    def train(
        self,
        train_dataloader: DataLoader,
        val_dataloader: Optional[DataLoader] = None,
        epochs: int = 3,
        learning_rate: float = 5e-5,
        warmup_steps: int = 100,
        gradient_accumulation_steps: int = 1,
        max_grad_norm: float = 1.0,
        output_dir: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        训练模型

        Args:
            train_dataloader: 训练数据加载器
            val_dataloader: 验证数据加载器
            epochs: 训练轮数
            learning_rate: 学习率
            warmup_steps: 预热步数
            gradient_accumulation_steps: 梯度累积步数
            max_grad_norm: 最大梯度范数
            output_dir: 输出目录

        Returns:
            训练统计信息
        """
        logger.info("Starting training...")
        logger.info(f"  Epochs: {epochs}")
        logger.info(f"  Learning rate: {learning_rate}")
        logger.info(f"  Warmup steps: {warmup_steps}")

        # 优化器
        optimizer = torch.optim.AdamW(
            [p for p in self.model.parameters() if p.requires_grad],
            lr=learning_rate
        )

        # 学习率调度器
        total_steps = len(train_dataloader) * epochs // gradient_accumulation_steps
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )

        # 训练循环
        self.model.train()
        start_time = time.time()
        global_step = 0
        total_loss = 0

        for epoch in range(epochs):
            epoch_loss = 0
            epoch_steps = 0

            for step, batch in enumerate(train_dataloader):
                # 移动到设备
                batch = {k: v.to(self.device) for k, v in batch.items()}

                # 前向传播
                if self.mixed_precision and self.scaler:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(**batch)
                        loss = outputs.loss / gradient_accumulation_steps
                    self.scaler.scale(loss).backward()
                else:
                    outputs = self.model(**batch)
                    loss = outputs.loss / gradient_accumulation_steps
                    loss.backward()

                epoch_loss += loss.item() * gradient_accumulation_steps
                epoch_steps += 1

                # 梯度累积
                if (step + 1) % gradient_accumulation_steps == 0:
                    if self.mixed_precision and self.scaler:
                        self.scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)
                        self.scaler.step(optimizer)
                        self.scaler.update()
                    else:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)
                        optimizer.step()

                    scheduler.step()
                    optimizer.zero_grad()
                    global_step += 1

                # 记录内存使用
                if self.device == "cuda":
                    memory_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
                    self.training_stats["peak_memory_mb"] = max(
                        self.training_stats["peak_memory_mb"],
                        memory_mb
                    )

            avg_epoch_loss = epoch_loss / epoch_steps
            logger.info(f"Epoch {epoch + 1}/{epochs} - Loss: {avg_epoch_loss:.4f}")

            # 验证
            if val_dataloader:
                val_loss = self.evaluate(val_dataloader)
                logger.info(f"Validation Loss: {val_loss:.4f}")

        # 训练结束
        training_time = time.time() - start_time
        self.training_stats["training_time"] = training_time
        self.training_stats["epochs"] = epochs
        self.training_stats["final_loss"] = avg_epoch_loss

        logger.info(f"Training completed in {training_time:.2f} seconds")

        # 保存模型
        if output_dir:
            self.save_model(output_dir)

        return self.training_stats

    def evaluate(self, dataloader: DataLoader) -> float:
        """评估模型"""
        self.model.eval()
        total_loss = 0
        total_steps = 0

        with torch.no_grad():
            for batch in dataloader:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                outputs = self.model(**batch)
                total_loss += outputs.loss.item()
                total_steps += 1

        self.model.train()
        return total_loss / total_steps

    def save_model(self, output_dir: str):
        """保存模型"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # 保存模型
        if self.method == "lora":
            self.model.save_pretrained(str(output_path))
        else:
            torch.save(self.model.state_dict(), output_path / "model.pt")

        # 保存 tokenizer
        self.tokenizer.save_pretrained(str(output_path))

        # 保存统计信息
        with open(output_path / "training_stats.json", "w", encoding="utf-8") as f:
            json.dump(self.training_stats, f, indent=2, ensure_ascii=False)

        logger.info(f"Model saved to {output_dir}")

    def get_stats(self) -> Dict[str, Any]:
        """获取训练统计信息"""
        return self.training_stats.copy()


if __name__ == "__main__":
    # 测试代码
    logging.basicConfig(level=logging.INFO)

    # 创建训练器
    trainer = AdvancedFinetuner(
        model_name="gpt2",
        method="lora",
        method_config={"r": 8, "alpha": 16},
        device="cpu"
    )

    print(f"Trainable params: {trainer.training_stats['trainable_params']:,}")
    print(f"Total params: {trainer.training_stats['total_params']:,}")
