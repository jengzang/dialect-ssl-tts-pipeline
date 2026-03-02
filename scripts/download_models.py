#!/usr/bin/env python3
"""
模型下载脚本

提前下载项目所需的所有预训练模型到本地缓存
"""

import os
import sys
from pathlib import Path

# 设置环境变量，使用国内镜像（可选）
# os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

print("="*80)
print("模型下载脚本")
print("="*80)
print("\n这个脚本会下载项目所需的所有预训练模型")
print("下载位置：~/.cache/huggingface/hub/")
print("\n注意：")
print("- 总下载大小约 1-2 GB")
print("- 需要稳定的网络连接")
print("- 可能需要 10-30 分钟")
print("\n" + "="*80 + "\n")

# 项目中使用的模型列表
MODELS = {
    "小型模型（必需）": [
        {
            "name": "gpt2",
            "size": "~500 MB",
            "description": "GPT-2 英文基础模型",
            "usage": "测试脚本、示例代码"
        },
        {
            "name": "uer/gpt2-chinese-cluecorpussmall",
            "size": "~400 MB",
            "description": "GPT-2 中文小模型",
            "usage": "多任务学习、指令微调、模型比较"
        },
    ],
    "中型模型（可选）": [
        {
            "name": "uer/gpt2-chinese-poem",
            "size": "~450 MB",
            "description": "GPT-2 中文诗歌模型",
            "usage": "模型比较"
        },
        {
            "name": "facebook/wav2vec2-base",
            "size": "~360 MB",
            "description": "Wav2Vec 2.0 基础模型",
            "usage": "语音识别、口音分类"
        },
    ],
    "大型模型（按需）": [
        {
            "name": "Qwen/Qwen-7B-Chat",
            "size": "~14 GB",
            "description": "Qwen 7B 对话模型",
            "usage": "方言翻译（需要大显存）"
        },
    ]
}


def download_model(model_name, model_type="auto"):
    """
    下载单个模型

    Args:
        model_name: 模型名称
        model_type: 模型类型 (auto/tokenizer/model)
    """
    from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM

    print(f"\n{'='*60}")
    print(f"下载: {model_name}")
    print(f"{'='*60}")

    try:
        # 下载 tokenizer
        if model_type in ["auto", "tokenizer"]:
            print("  [1/2] 下载 tokenizer...")
            tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                trust_remote_code=True
            )
            print("  [OK] Tokenizer 下载完成")

        # 下载模型
        if model_type in ["auto", "model"]:
            print("  [2/2] 下载模型...")

            # 根据模型类型选择合适的类
            if "wav2vec" in model_name.lower():
                from transformers import Wav2Vec2Model
                model = Wav2Vec2Model.from_pretrained(
                    model_name,
                    trust_remote_code=True
                )
            else:
                model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    trust_remote_code=True
                )
            print("  [OK] 模型下载完成")

        print(f"\n[OK] {model_name} 下载成功！")
        return True

    except Exception as e:
        print(f"\n[FAIL] {model_name} 下载失败: {e}")
        return False


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description="下载项目所需的预训练模型")
    parser.add_argument(
        "--category",
        type=str,
        choices=["small", "medium", "large", "all"],
        default="small",
        help="下载类别：small(小型), medium(中型), large(大型), all(全部)"
    )
    parser.add_argument(
        "--model",
        type=str,
        help="下载指定模型（模型名称）"
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="列出所有模型"
    )
    parser.add_argument(
        "--yes", "-y",
        action="store_true",
        help="自动确认，跳过交互式提示"
    )

    args = parser.parse_args()

    # 列出模型
    if args.list:
        print("\n项目所需模型列表：\n")
        for category, models in MODELS.items():
            print(f"\n{category}:")
            for model in models:
                print(f"  - {model['name']}")
                print(f"    大小: {model['size']}")
                print(f"    说明: {model['description']}")
                print(f"    用途: {model['usage']}")
        print("\n")
        return

    # 下载指定模型
    if args.model:
        print(f"\n下载指定模型: {args.model}\n")
        success = download_model(args.model)
        sys.exit(0 if success else 1)

    # 根据类别下载
    categories_to_download = []

    if args.category == "small":
        categories_to_download = ["小型模型（必需）"]
    elif args.category == "medium":
        categories_to_download = ["小型模型（必需）", "中型模型（可选）"]
    elif args.category == "large":
        categories_to_download = ["小型模型（必需）", "中型模型（可选）", "大型模型（按需）"]
    elif args.category == "all":
        categories_to_download = list(MODELS.keys())

    # 收集要下载的模型
    models_to_download = []
    for category in categories_to_download:
        if category in MODELS:
            models_to_download.extend(MODELS[category])

    print(f"\n准备下载 {len(models_to_download)} 个模型\n")

    # 确认
    if not args.yes:
        response = input("是否继续？(y/n): ")
        if response.lower() != 'y':
            print("取消下载")
            return
    else:
        print("自动确认模式，开始下载...\n")

    # 下载模型
    results = {}
    for i, model_info in enumerate(models_to_download, 1):
        model_name = model_info["name"]
        print(f"\n[{i}/{len(models_to_download)}] 处理: {model_name}")
        print(f"大小: {model_info['size']}")

        success = download_model(model_name)
        results[model_name] = success

    # 总结
    print("\n" + "="*80)
    print("下载总结")
    print("="*80)

    success_count = sum(results.values())
    total_count = len(results)

    for model_name, success in results.items():
        status = "[OK]" if success else "[FAIL]"
        print(f"  {status} {model_name}")

    print(f"\n总计: {success_count}/{total_count} 成功")

    if success_count == total_count:
        print("\n[OK] 所有模型下载完成！")
        print("\n模型存储位置:")
        cache_dir = Path.home() / ".cache" / "huggingface" / "hub"
        print(f"  {cache_dir}")
    else:
        print(f"\n[FAIL] {total_count - success_count} 个模型下载失败")
        print("\n可能的原因:")
        print("  1. 网络连接不稳定")
        print("  2. 磁盘空间不足")
        print("  3. 模型名称错误")
        print("\n建议:")
        print("  - 检查网络连接")
        print("  - 使用国内镜像: export HF_ENDPOINT=https://hf-mirror.com")
        print("  - 单独下载失败的模型: python scripts/download_models.py --model <模型名>")


if __name__ == "__main__":
    main()
