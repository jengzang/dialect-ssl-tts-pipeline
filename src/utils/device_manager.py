"""设备管理模块

提供 CPU/GPU 设备的自动检测和管理功能。
"""

from typing import Optional, Union, Any
import logging

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None


class DeviceManager:
    """设备管理器

    负责检测和管理计算设备（CPU/GPU）。
    """

    def __init__(self, device: Optional[str] = None):
        """初始化设备管理器

        Args:
            device: 设备类型，可选值：'auto', 'cpu', 'cuda', 'cuda:0' 等
                   如果为 None 或 'auto'，则自动检测
        """
        if not TORCH_AVAILABLE:
            raise ImportError(
                "PyTorch 未安装。请先安装 PyTorch: pip install torch"
            )

        self.logger = logging.getLogger(__name__)
        self._device = self._initialize_device(device)

    def _initialize_device(self, device: Optional[str]) -> Any:
        """初始化设备

        Args:
            device: 设备类型

        Returns:
            torch.device 实例
        """
        if device is None or device == 'auto':
            # 自动检测
            if torch.cuda.is_available():
                device_name = 'cuda'
                self.logger.info(f"检测到 GPU: {torch.cuda.get_device_name(0)}")
            else:
                device_name = 'cpu'
                self.logger.info("未检测到 GPU，使用 CPU")
        else:
            device_name = device

        device_obj = torch.device(device_name)

        # 验证设备是否可用
        if device_obj.type == 'cuda' and not torch.cuda.is_available():
            self.logger.warning("指定了 CUDA 设备但不可用，回退到 CPU")
            device_obj = torch.device('cpu')

        self.logger.info(f"使用设备: {device_obj}")
        return device_obj

    @property
    def device(self) -> Any:
        """获取当前设备

        Returns:
            torch.device 实例
        """
        return self._device

    def to_device(self, tensor: Any) -> Any:
        """将张量移动到当前设备

        Args:
            tensor: 输入张量

        Returns:
            移动后的张量
        """
        return tensor.to(self._device)

    def is_cuda(self) -> bool:
        """检查是否使用 CUDA

        Returns:
            是否使用 CUDA
        """
        return self._device.type == 'cuda'

    def get_device_info(self) -> dict:
        """获取设备信息

        Returns:
            设备信息字典
        """
        info = {
            'device': str(self._device),
            'type': self._device.type,
            'cuda_available': torch.cuda.is_available()
        }

        if torch.cuda.is_available():
            info.update({
                'cuda_device_count': torch.cuda.device_count(),
                'cuda_device_name': torch.cuda.get_device_name(0),
                'cuda_memory_allocated': torch.cuda.memory_allocated(0),
                'cuda_memory_reserved': torch.cuda.memory_reserved(0)
            })

        return info

    def empty_cache(self):
        """清空 CUDA 缓存"""
        if self.is_cuda():
            torch.cuda.empty_cache()
            self.logger.info("已清空 CUDA 缓存")


def get_device(device: Optional[str] = None):
    """获取设备（便捷函数）

    Args:
        device: 设备类型

    Returns:
        torch.device 实例，如果 PyTorch 未安装则返回 None
    """
    if not TORCH_AVAILABLE:
        return None

    manager = DeviceManager(device)
    return manager.device
