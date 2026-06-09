from enum import Enum


class ProcessingStatus(str, Enum):
    """Status of video/image processing session.

    Lifecycle for a video:
        QUEUED -> PROCESSING -> COMPLETED
                            \\-> FAILED        (unrecoverable error, logged)
                            \\-> INTERRUPTED   (worker died / stalled — resumable)

    INTERRUPTED is the key addition: a job that stops without finishing (OOM,
    redeploy, a dead heartbeat) is no longer silently left as PROCESSING or
    falsely marked COMPLETED — it is INTERRUPTED, which the reaper re-enqueues
    from `processed_frames` and the UI offers a "Resume" for.
    """

    PENDING = "pending"
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    INTERRUPTED = "interrupted"

    def __str__(self):
        return self.value


# Work is still in flight (or waiting to start) — counts as "active".
ACTIVE_STATUSES = (
    ProcessingStatus.PENDING.value,
    ProcessingStatus.QUEUED.value,
    ProcessingStatus.PROCESSING.value,
)

# Stopped before finishing but safe to resume from processed_frames.
RESUMABLE_STATUSES = (
    ProcessingStatus.INTERRUPTED.value,
    ProcessingStatus.FAILED.value,
)
