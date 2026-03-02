"""
方言翻译模型训练器（LoRA 微调）

使用 LoRA 技术微调大语言模型，实现方言到普通话的翻译。
"""

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import get_linear_schedule_with_warmup
import logging
from typing import Dict, Any, List, Tuple, Optional
from pathlib import Path
from tqdm import tqdm
import json

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    logging.warning("WandB not available. Install: pip install wandb")

from src.models.dialect_translator import DialectTranslator

logger = logging.getLogger(__name__)


class DialectTranslationDataset(Dataset):
    """方言翻译数据集"""

    def __init__(
        self,
        data_path: str,
        tokenizer,
        max_length: int = 512
    ):
        """
        初始化数据集

        Args:
            data_path: 数据文件路径（JSON 格式）
            tokenizer: Tokenizer
            max_length: 最大序列长度
        """
        self.tokenizer = tokenizer
        self.max_length = max_length

        # 加载数据
        with open(data_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)

        logger.info(f"Loaded {len(self.data)} examples from {data_path}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        dialect_text = item['dialect']
        mandarin_text = item['mandarin']

        # 构建输入和标签
        input_text = f"请将以下方言翻译成普通话：\n方言：{dialect_text}\n普通话："
        target_text = f"{mandarin_text}"

        # Tokenize
        full_text = input_text + target_text

        encoding = self.tokenizer(
            full_text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        input_ids = encoding['input_ids'].squeeze()
        attention_mask = encoding['attention_mask'].squeeze()

        # 创建标签（只计算目标部分的损失）
        input_encoding = self.tokenizer(
            input_text,
            max_length=self.max_length,
            truncation=True
        )
        input_length = len(input_encoding['input_ids'])

        labels = input_ids.clone()
        labels[:input_length] = -100  # 忽略输入部分的损失

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }


class DialectTranslationTrainer:
    """方言翻译训练器"""

    def __init__(self, config: Dict[str, Any]):
        """
        初始化训练器

        Args:
            config: 配置字典
        """
        self.config = config
        self.device = config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')

        # 初始化模型
        self.translator = DialectTranslator(config)

        # 训练参数
        self.epochs = config.get('epochs', 3)
        self.batch_size = config.get('batch_size', 4)
        self.learning_rate = config.get('learning_rate', 2e-4)
        self.warmup_steps = config.get('warmup_steps', 100)
        self.gradient_accumulation_steps = config.get('gradient_accumulation_steps', 4)
        self.max_grad_norm = config.get('max_grad_norm', 1.0)

        # WandB 配置
        self.use_wandb = config.get('use_wandb', False)
        self.wandb_project = config.get('wandb_project', 'dialect-translation')
        self.wandb_run_name = config.get('wandb_run_name', None)

        # 输出目录
        self.output_dir = Path(config.get('output_dir', 'checkpoints/dialect_translator'))
        self.output_dir.mkdir(exist_ok=True, parents=True)

        self.optimizer = None
        self.scheduler = None

        # 初始化 WandB
        if self.use_wandb and WANDB_AVAILABLE:
            wandb.init(
                project=self.wandb_project,
                name=self.wandb_run_name,
                config=config
            )
            logger.info("WandB initialized")
        elif self.use_wandb and not WANDB_AVAILABLE:
            logger.warning("WandB requested but not available")

    def prepare_model(self, quantization: bool = False):
        """准备模型"""
        logger.info("Preparing model...")

        # 加载基础模型
        self.translator.load_base_model(quantization=quantization)

        # 应用 LoRA
        self.translator.apply_lora()

        logger.info("Model prepared")

    def prepare_data(
        self,
        train_data_path: str,
        val_data_path: Optional[str] = None
    ) -> Tuple[DataLoader, Optional[DataLoader]]:
        """
        准备数据

        Args:
            train_data_path: 训练数据路径
            val_data_path: 验证数据路径（可选）

        Returns:
            训练和验证 DataLoader
        """
        logger.info("Preparing data...")

        # 训练集
        train_dataset = DialectTranslationDataset(
            train_data_path,
            self.translator.tokenizer,
            self.translator.max_length
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0  # Windows 兼容性
        )

        # 验证集（可选）
        val_loader = None
        if val_data_path:
            val_dataset = DialectTranslationDataset(
                val_data_path,
                self.translator.tokenizer,
                self.translator.max_length
            )

            val_loader = DataLoader(
                val_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=0
            )

        logger.info(f"Train batches: {len(train_loader)}")
        if val_loader:
            logger.info(f"Val batches: {len(val_loader)}")

        return train_loader, val_loader

    def train(
        self,
        train_data_path: str,
        val_data_path: Optional[str] = None
    ):
        """
        训练模型

        Args:
            train_data_path: 训练数据路径
            val_data_path: 验证数据路径（可选）
        """
        logger.info("Starting training...")

        # 准备数据
        train_loader, val_loader = self.prepare_data(train_data_path, val_data_path)

        # 准备优化器
        self.optimizer = torch.optim.AdamW(
            self.translator.model.parameters(),
            lr=self.learning_rate
        )

        # 准备学习率调度器
        total_steps = len(train_loader) * self.epochs // self.gradient_accumulation_steps
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=self.warmup_steps,
            num_training_steps=total_steps
        )

        # 训练循环
        best_val_loss = float('inf')
        global_step = 0

        for epoch in range(self.epochs):
            logger.info(f"Epoch {epoch + 1}/{self.epochs}")

            # 训练
            train_loss = self._train_epoch(train_loader, global_step)
            logger.info(f"Train loss: {train_loss:.4f}")

            # 记录到 WandB
            if self.use_wandb and WANDB_AVAILABLE:
                wandb.log({
                    'epoch': epoch + 1,
                    'train_loss': train_loss
                })

            # 验证
            if val_loader:
                val_loss = self._validate(val_loader)
                logger.info(f"Val loss: {val_loss:.4f}")

                # 记录到 WandB
                if self.use_wandb and WANDB_AVAILABLE:
                    wandb.log({
                        'epoch': epoch + 1,
                        'val_loss': val_loss
                    })

                # 保存最佳模型
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    self._save_checkpoint(epoch, 'best')
                    logger.info("Saved best model")

            # 保存检查点
            self._save_checkpoint(epoch, f'epoch_{epoch + 1}')

        logger.info("Training completed")

        # 结束 WandB
        if self.use_wandb and WANDB_AVAILABLE:
            wandb.finish()

    def _train_epoch(self, train_loader: DataLoader, global_step: int) -> float:
        """训练一个 epoch"""
        self.translator.model.train()
        total_loss = 0
        self.optimizer.zero_grad()

        progress_bar = tqdm(train_loader, desc="Training")

        for step, batch in enumerate(progress_bar):
            # 移动到设备
            batch = {k: v.to(self.device) for k, v in batch.items()}

            # 前向传播
            outputs = self.translator.model(**batch)
            loss = outputs.loss / self.gradient_accumulation_steps

            # 反向传播
            loss.backward()

            # 梯度累积
            if (step + 1) % self.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(
                    self.translator.model.parameters(),
                    self.max_grad_norm
                )
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()
                global_step += 1

            total_loss += loss.item() * self.gradient_accumulation_steps

            # 更新进度条
            progress_bar.set_postfix({'loss': loss.item()})

            # 记录到 WandB（每 10 步）
            if self.use_wandb and WANDB_AVAILABLE and step % 10 == 0:
                wandb.log({
                    'train_step_loss': loss.item() * self.gradient_accumulation_steps,
                    'learning_rate': self.scheduler.get_last_lr()[0],
                    'global_step': global_step
                })

        return total_loss / len(train_loader)

    def _validate(self, val_loader: DataLoader) -> float:
        """验证"""
        self.translator.model.eval()
        total_loss = 0

        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validating"):
                batch = {k: v.to(self.device) for k, v in batch.items()}
                outputs = self.translator.model(**batch)
                total_loss += outputs.loss.item()

        return total_loss / len(val_loader)

    def _save_checkpoint(self, epoch: int, name: str):
        """保存检查点"""
        checkpoint_dir = self.output_dir / name
        self.translator.save_model(str(checkpoint_dir))

        # 保存训练状态
        state = {
            'epoch': epoch,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict()
        }

        torch.save(state, checkpoint_dir / 'training_state.pt')

        logger.info(f"Checkpoint saved: {checkpoint_dir}")
