"""
指令数据集构建器

将方言任务转换为指令格式，支持多种指令模板。

支持的任务类型：
1. 翻译（方言 → 普通话）
2. 反向翻译（普通话 → 方言）
3. 方言识别
4. 方言解释
5. 文化背景
"""

import logging
from typing import Dict, Any, List, Optional
import json
import random
from pathlib import Path

logger = logging.getLogger(__name__)


# 指令模板
INSTRUCTION_TEMPLATES = {
    "alpaca": {
        "translation": "### 指令:\n{instruction}\n\n### 输入:\n{input}\n\n### 回答:\n{output}",
        "classification": "### 指令:\n{instruction}\n\n### 输入:\n{input}\n\n### 回答:\n{output}",
        "explanation": "### 指令:\n{instruction}\n\n### 输入:\n{input}\n\n### 回答:\n{output}"
    },
    "vicuna": {
        "translation": "USER: {instruction}\n{input}\nASSISTANT: {output}",
        "classification": "USER: {instruction}\n{input}\nASSISTANT: {output}",
        "explanation": "USER: {instruction}\n{input}\nASSISTANT: {output}"
    },
    "simple": {
        "translation": "{instruction}\n输入：{input}\n输出：{output}",
        "classification": "{instruction}\n输入：{input}\n输出：{output}",
        "explanation": "{instruction}\n输入：{input}\n输出：{output}"
    }
}


# 指令变体（用于数据增强）
TRANSLATION_INSTRUCTIONS = [
    "请将以下方言翻译成普通话",
    "把下面的方言转换为标准普通话",
    "翻译这段方言为普通话",
    "将这句方言话翻译成普通话",
    "请用普通话表达以下方言内容"
]

REVERSE_TRANSLATION_INSTRUCTIONS = [
    "请将以下普通话翻译成方言",
    "把下面的普通话转换为方言",
    "用方言表达以下普通话内容",
    "将这句普通话翻译成方言",
    "请用方言说出以下内容"
]

CLASSIFICATION_INSTRUCTIONS = [
    "请识别以下文本的方言类型",
    "判断这段话是什么方言",
    "识别这句话的方言种类",
    "这是哪种方言？",
    "请分析这段文本属于哪种方言"
]

EXPLANATION_INSTRUCTIONS = [
    "请解释以下方言词汇的含义",
    "这个方言词是什么意思？",
    "解释一下这个方言表达",
    "这句方言话的意思是什么？",
    "请说明这个方言词汇的含义"
]


class InstructionBuilder:
    """指令数据集构建器"""

    def __init__(
        self,
        template_style: str = "alpaca",
        add_system_prompt: bool = True,
        shuffle_instructions: bool = True
    ):
        """
        初始化指令构建器

        Args:
            template_style: 模板风格（alpaca, vicuna, simple）
            add_system_prompt: 是否添加系统提示
            shuffle_instructions: 是否随机选择指令变体
        """
        self.template_style = template_style
        self.add_system_prompt = add_system_prompt
        self.shuffle_instructions = shuffle_instructions

        if template_style not in INSTRUCTION_TEMPLATES:
            raise ValueError(f"Unknown template style: {template_style}")

        self.templates = INSTRUCTION_TEMPLATES[template_style]

        logger.info(f"InstructionBuilder initialized with style: {template_style}")

    def build_translation_instruction(
        self,
        dialect_text: str,
        mandarin_text: str,
        reverse: bool = False
    ) -> Dict[str, str]:
        """
        构建翻译指令

        Args:
            dialect_text: 方言文本
            mandarin_text: 普通话文本
            reverse: 是否反向翻译（普通话 → 方言）

        Returns:
            指令字典
        """
        if reverse:
            instruction = random.choice(REVERSE_TRANSLATION_INSTRUCTIONS) if self.shuffle_instructions else REVERSE_TRANSLATION_INSTRUCTIONS[0]
            input_text = mandarin_text
            output_text = dialect_text
        else:
            instruction = random.choice(TRANSLATION_INSTRUCTIONS) if self.shuffle_instructions else TRANSLATION_INSTRUCTIONS[0]
            input_text = dialect_text
            output_text = mandarin_text

        return {
            "instruction": instruction,
            "input": input_text,
            "output": output_text,
            "task_type": "translation"
        }

    def build_classification_instruction(
        self,
        text: str,
        dialect_label: str
    ) -> Dict[str, str]:
        """
        构建分类指令

        Args:
            text: 文本
            dialect_label: 方言标签

        Returns:
            指令字典
        """
        instruction = random.choice(CLASSIFICATION_INSTRUCTIONS) if self.shuffle_instructions else CLASSIFICATION_INSTRUCTIONS[0]

        return {
            "instruction": instruction,
            "input": text,
            "output": dialect_label,
            "task_type": "classification"
        }

    def build_explanation_instruction(
        self,
        dialect_word: str,
        explanation: str
    ) -> Dict[str, str]:
        """
        构建解释指令

        Args:
            dialect_word: 方言词汇
            explanation: 解释

        Returns:
            指令字典
        """
        instruction = random.choice(EXPLANATION_INSTRUCTIONS) if self.shuffle_instructions else EXPLANATION_INSTRUCTIONS[0]

        return {
            "instruction": instruction,
            "input": dialect_word,
            "output": explanation,
            "task_type": "explanation"
        }

    def format_instruction(self, instruction_dict: Dict[str, str]) -> str:
        """
        格式化指令

        Args:
            instruction_dict: 指令字典

        Returns:
            格式化后的指令文本
        """
        task_type = instruction_dict.get("task_type", "translation")
        template = self.templates.get(task_type, self.templates["translation"])

        formatted = template.format(
            instruction=instruction_dict["instruction"],
            input=instruction_dict["input"],
            output=instruction_dict["output"]
        )

        if self.add_system_prompt:
            system_prompt = "你是一个专业的方言助手，能够帮助用户进行方言翻译、识别和解释。\n\n"
            formatted = system_prompt + formatted

        return formatted

    def build_from_translation_data(
        self,
        translation_data: List[Dict[str, str]],
        include_reverse: bool = True
    ) -> List[Dict[str, Any]]:
        """
        从翻译数据构建指令数据集

        Args:
            translation_data: 翻译数据列表
            include_reverse: 是否包含反向翻译

        Returns:
            指令数据集
        """
        instructions = []

        for item in translation_data:
            dialect = item.get("dialect", "")
            mandarin = item.get("mandarin", "")

            if not dialect or not mandarin:
                continue

            # 正向翻译
            inst = self.build_translation_instruction(dialect, mandarin, reverse=False)
            instructions.append(inst)

            # 反向翻译
            if include_reverse:
                inst_reverse = self.build_translation_instruction(dialect, mandarin, reverse=True)
                instructions.append(inst_reverse)

        logger.info(f"Built {len(instructions)} translation instructions")
        return instructions

    def build_from_classification_data(
        self,
        classification_data: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        从分类数据构建指令数据集

        Args:
            classification_data: 分类数据列表

        Returns:
            指令数据集
        """
        instructions = []

        label_map = {
            0: "上海话",
            1: "粤语",
            2: "普通话",
            3: "其他方言",
            "shanghai": "上海话",
            "cantonese": "粤语",
            "mandarin": "普通话",
            "other": "其他方言"
        }

        for item in classification_data:
            text = item.get("text", item.get("dialect", ""))
            label = item.get("label", item.get("accent", 0))

            if not text:
                continue

            # 转换标签
            dialect_label = label_map.get(label, "未知方言")

            inst = self.build_classification_instruction(text, dialect_label)
            instructions.append(inst)

        logger.info(f"Built {len(instructions)} classification instructions")
        return instructions

    def save_instructions(
        self,
        instructions: List[Dict[str, Any]],
        output_path: str,
        format_text: bool = True
    ):
        """
        保存指令数据集

        Args:
            instructions: 指令列表
            output_path: 输出路径
            format_text: 是否格式化为文本
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if format_text:
            # 格式化为文本
            formatted_instructions = []
            for inst in instructions:
                formatted = self.format_instruction(inst)
                formatted_instructions.append({
                    **inst,
                    "formatted_text": formatted
                })
            instructions = formatted_instructions

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(instructions, f, ensure_ascii=False, indent=2)

        logger.info(f"Saved {len(instructions)} instructions to {output_path}")


def create_instruction_dataset(
    translation_data_path: Optional[str] = None,
    classification_data_path: Optional[str] = None,
    output_path: str = "data/instruction_dataset.json",
    template_style: str = "alpaca",
    include_reverse_translation: bool = True
) -> List[Dict[str, Any]]:
    """
    创建指令数据集的便捷函数

    Args:
        translation_data_path: 翻译数据路径
        classification_data_path: 分类数据路径
        output_path: 输出路径
        template_style: 模板风格
        include_reverse_translation: 是否包含反向翻译

    Returns:
        指令数据集
    """
    builder = InstructionBuilder(template_style=template_style)

    all_instructions = []

    # 加载翻译数据
    if translation_data_path:
        with open(translation_data_path, 'r', encoding='utf-8') as f:
            translation_data = json.load(f)
        translation_instructions = builder.build_from_translation_data(
            translation_data,
            include_reverse=include_reverse_translation
        )
        all_instructions.extend(translation_instructions)

    # 加载分类数据
    if classification_data_path:
        with open(classification_data_path, 'r', encoding='utf-8') as f:
            classification_data = json.load(f)
        classification_instructions = builder.build_from_classification_data(
            classification_data
        )
        all_instructions.extend(classification_instructions)

    # 打乱顺序
    random.shuffle(all_instructions)

    # 保存
    builder.save_instructions(all_instructions, output_path)

    logger.info(f"Created instruction dataset with {len(all_instructions)} samples")
    return all_instructions
