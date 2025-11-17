from enum import Enum


class FileType(str, Enum):
    """File type for uploads"""

    IMAGE = "image"
    VIDEO = "video"

    def __str__(self):
        return self.value
