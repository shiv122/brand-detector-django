from enum import Enum


class ProcessingStatus(str, Enum):
    """Status of video/image processing session"""

    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"

    def __str__(self):
        return self.value
