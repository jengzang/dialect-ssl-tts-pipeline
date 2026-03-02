"""
测试 Project 1 的核心功能

验证：
1. MT 评估指标计算
2. 数据增强
"""

import sys
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))


def test_mt_metrics():
    """测试 MT 评估指标"""
    print("=== Testing MT Metrics ===")

    try:
        from src.evaluation.mt_metrics import MTMetrics

        # 创建测试数据
        predictions = [
            "你好，今天天气很好",
            "他们去洗衣服了",
            "我喜欢吃粢饭糕"
        ]

        references = [
            "你好，今天天气很好",
            "他们去洗衣服了",
            "我喜欢吃糍饭糕"  # 略有不同
        ]

        # 计算指标
        metrics = MTMetrics()
        results = metrics.compute_all(predictions, references)

        print(f"BLEU-4: {results.get('bleu', 0):.2f}")
        print(f"ROUGE-L F1: {results.get('rougeL_fmeasure', 0):.2f}")
        print(f"ChrF: {results.get('chrf', 0):.2f}")

        print("[OK] MT Metrics test passed\n")
        return True

    except Exception as e:
        print(f"[ERROR] MT Metrics test failed: {e}\n")
        import traceback
        traceback.print_exc()
        return False


def test_data_augmentation():
    """测试数据增强"""
    print("=== Testing Data Augmentation ===")

    try:
        from src.data_pipeline.dialect_augmentation import DialectDataAugmenter

        # 创建测试数据
        test_data = [
            {"dialect": "侬好", "mandarin": "你好"},
            {"dialect": "伊拉去汏衣裳哉", "mandarin": "他们去洗衣服了"},
            {"dialect": "吾欢喜吃粢饭糕", "mandarin": "我喜欢吃粢饭糕"}
        ]

        # 测试增强
        augmenter = DialectDataAugmenter(seed=42)
        augmented = augmenter.augment_dataset(test_data, target_size=10)

        print(f"Original size: {len(test_data)}")
        print(f"Augmented size: {len(augmented)}")

        # 测试划分
        splits = augmenter.split_dataset(augmented, 0.7, 0.15, 0.15)
        print(f"Train: {len(splits['train'])}")
        print(f"Val: {len(splits['val'])}")
        print(f"Test: {len(splits['test'])}")

        print("[OK] Data Augmentation test passed\n")
        return True

    except Exception as e:
        print(f"[ERROR] Data Augmentation test failed: {e}\n")
        import traceback
        traceback.print_exc()
        return False


def main():
    """运行所有测试"""
    print("=" * 50)
    print("Project 1: Enhanced Evaluation System - Tests")
    print("=" * 50)
    print()

    results = []

    # 测试 MT 指标
    results.append(("MT Metrics", test_mt_metrics()))

    # 测试数据增强
    results.append(("Data Augmentation", test_data_augmentation()))

    # 总结
    print("=" * 50)
    print("Test Summary")
    print("=" * 50)

    for name, passed in results:
        status = "[PASSED]" if passed else "[FAILED]"
        print(f"{name}: {status}")

    all_passed = all(passed for _, passed in results)

    if all_passed:
        print("\n[OK] All tests passed!")
        sys.exit(0)
    else:
        print("\n[ERROR] Some tests failed!")
        sys.exit(1)


if __name__ == '__main__':
    main()
