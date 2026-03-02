"""
训练进度监控脚本
"""

import time
import sys
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

print("=" * 60)
print("Training Progress Monitor")
print("=" * 60)
print("\n正在监控训练进度...")
print("按 Ctrl+C 退出监控\n")

# 检查训练任务状态
import subprocess

try:
    while True:
        # 检查 GPU 使用情况
        try:
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=utilization.gpu,memory.used,memory.total', '--format=csv,noheader,nounits'],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                gpu_info = result.stdout.strip()
                print(f"\r[GPU] {gpu_info}  ", end='', flush=True)
        except:
            pass

        time.sleep(2)

except KeyboardInterrupt:
    print("\n\n监控已停止")
    print("\n训练仍在后台运行")
    print("使用以下命令查看完整日志:")
    print("  ls -lh checkpoints/dialect_translation_v2/")
    print("  ls -lh logs/")
