from enum import Enum


class DeviceType(str, Enum):
    """Device type for model inference"""

    CPU = "cpu"
    CUDA = "cuda"
    MPS = "mps"

    def __str__(self):
        return self.value
