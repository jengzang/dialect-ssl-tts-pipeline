#!/usr/bin/env python3
"""
统一测试运行器

运行所有项目的测试，并生成报告
"""

import subprocess
import sys
import time
from pathlib import Path
from typing import Tuple, Dict

# 配置
TEST_FILES = [
    "scripts/test_project_01.py",
    "scripts/test_project_03.py",
    "scripts/test_project_04.py",
    "scripts/test_project_05.py",
    "scripts/test_project_06.py",
    "scripts/test_project_07.py",
]


def run_test(test_file: str, timeout: int = 120) -> Tuple[bool, str, str, float]:
    """
    运行单个测试

    Args:
        test_file: 测试文件路径
        timeout: 超时时间（秒）

    Returns:
        (成功, 标准输出, 标准错误, 运行时间)
    """
    start_time = time.time()

    try:
        result = subprocess.run(
            [sys.executable, test_file],
            capture_output=True,
            text=True,
            timeout=timeout
        )

        elapsed_time = time.time() - start_time
        success = result.returncode == 0

        return success, result.stdout, result.stderr, elapsed_time

    except subprocess.TimeoutExpired:
        elapsed_time = time.time() - start_time
        return False, "", f"Test timeout after {timeout}s", elapsed_time

    except Exception as e:
        elapsed_time = time.time() - start_time
        return False, "", str(e), elapsed_time


def print_test_output(test_name: str, success: bool, stdout: str, stderr: str, elapsed_time: float):
    """打印测试输出"""
    status = "[PASSED]" if success else "[FAILED]"
    print(f"\n{test_name}: {status} ({elapsed_time:.2f}s)")

    if not success:
        # 打印错误信息（截取前500字符）
        if stderr:
            print(f"\nError output (first 500 chars):")
            print("-" * 60)
            print(stderr[:500])
            if len(stderr) > 500:
                print(f"\n... ({len(stderr) - 500} more characters)")
            print("-" * 60)


def generate_report(results: Dict[str, Tuple[bool, float]]):
    """生成测试报告"""
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)

    total = len(results)
    passed = sum(1 for success, _ in results.values() if success)
    failed = total - passed

    print(f"\nResults:")
    for test_file, (success, elapsed_time) in results.items():
        test_name = Path(test_file).stem
        status = "[PASSED]" if success else "[FAILED]"
        print(f"  {test_name:<25} {status:>12}  ({elapsed_time:.2f}s)")

    print(f"\nTotal: {passed}/{total} passed, {failed}/{total} failed")

    if passed == total:
        print("\nAll tests passed!")
        return 0
    else:
        print(f"\n{failed} test(s) failed")
        return 1


def main():
    """主函数"""
    print("="*80)
    print("LLM Training Path - Unified Test Runner")
    print("="*80)
    print(f"\nRunning {len(TEST_FILES)} test suites...\n")

    results = {}

    for i, test_file in enumerate(TEST_FILES, 1):
        test_name = Path(test_file).stem
        print(f"[{i}/{len(TEST_FILES)}] Running {test_name}...", end=" ", flush=True)

        success, stdout, stderr, elapsed_time = run_test(test_file)

        results[test_file] = (success, elapsed_time)

        # 简短输出
        if success:
            print(f"[OK] ({elapsed_time:.2f}s)")
        else:
            print(f"[FAIL] ({elapsed_time:.2f}s)")

            # 打印详细错误（可选）
            if "--verbose" in sys.argv:
                print_test_output(test_name, success, stdout, stderr, elapsed_time)

    # 生成报告
    exit_code = generate_report(results)

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
