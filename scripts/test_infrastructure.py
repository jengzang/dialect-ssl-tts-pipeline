"""基础设施验证脚本

测试配置加载、日志和设备管理功能。
"""

import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.utils import get_config_loader, setup_logger, get_device


def test_config_loader():
    """测试配置加载器"""
    print("=" * 50)
    print("测试配置加载器")
    print("=" * 50)

    config_loader = get_config_loader()
    config = config_loader.load()

    print(f"[OK] 配置文件加载成功")
    print(f"  - 路径配置: {config.get('paths', {})}")
    print(f"  - SVM 配置: {config.get('svm', {})}")

    # 测试嵌套键访问
    data_dir = config_loader.get('paths.data_dir')
    print(f"  - 数据目录: {data_dir}")

    # 测试课程配置
    svm_config = config_loader.get_lesson_config('svm')
    print(f"  - SVM 课程配置: {svm_config}")

    print()


def test_logger():
    """测试日志系统"""
    print("=" * 50)
    print("测试日志系统")
    print("=" * 50)

    config_loader = get_config_loader()
    logging_config = config_loader.config.get('logging', {})
    logging_config['log_dir'] = config_loader.get('paths.log_dir', 'logs')

    logger = setup_logger('test_logger', logging_config)

    logger.debug("这是一条 DEBUG 消息")
    logger.info("这是一条 INFO 消息")
    logger.warning("这是一条 WARNING 消息")
    logger.error("这是一条 ERROR 消息")

    print(f"[OK] 日志系统工作正常")
    print(f"  - 日志文件保存在: {logging_config['log_dir']}")
    print()


def test_device_manager():
    """测试设备管理器"""
    print("=" * 50)
    print("测试设备管理器")
    print("=" * 50)

    try:
        import torch
        device = get_device('auto')
        print(f"[OK] 设备管理器工作正常")
        print(f"  - 当前设备: {device}")
        print(f"  - CUDA 可用: {torch.cuda.is_available()}")

        if torch.cuda.is_available():
            print(f"  - GPU 名称: {torch.cuda.get_device_name(0)}")
    except ImportError:
        print("[WARN] PyTorch 未安装，跳过设备管理器测试")

    print()


def main():
    """主函数"""
    print("\n" + "=" * 50)
    print("基础设施验证")
    print("=" * 50 + "\n")

    try:
        test_config_loader()
        test_logger()
        test_device_manager()

        print("=" * 50)
        print("[OK] 所有测试通过！")
        print("=" * 50)

    except Exception as e:
        print(f"\n[ERROR] 测试失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
