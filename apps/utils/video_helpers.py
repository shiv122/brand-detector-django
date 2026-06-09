"""
Video helper functions - Laravel-style.

Frame extraction goes through ffmpeg (NOT OpenCV's frame-by-frame
``VideoCapture.read()``) for two reasons that were causing silent early
stops in production:

1. Robustness. ``cv2.VideoCapture.read()`` returns ``(False, None)`` on BOTH a
   true end-of-file AND a transient decode error (a damaged packet, a VFR
   discontinuity, a truncated download). The old loop treated any False as EOF
   and "completed" the video early. ffmpeg, with ``-fflags +discardcorrupt
   -err_detect ignore_err``, skips bad packets and keeps going, and its exit
   code tells us cleanly whether the stream really ended or aborted.

2. Correct sampling. ``-vf fps=N`` samples by wall-clock time, so "1 fps" means
   one frame per second regardless of the container's (often missing/garbage)
   fps metadata. The old ``video_fps // target_fps`` skip math silently
   collapsed to "process every frame" whenever OpenCV misread the fps.
"""

import json
import subprocess
from typing import Tuple, Optional

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Legacy OpenCV helpers — still used by the single-image / info paths.
# ---------------------------------------------------------------------------
def get_video_info(video_path: str) -> Tuple[int, int, int, int]:
    """Get video information (fps, total frames, width, height) via OpenCV."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("Could not open video file")

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    cap.release()

    return fps, total_frames, width, height


def calculate_skip_frames(video_fps: int, target_fps: int) -> int:
    """Calculate how many frames to skip to achieve target FPS."""
    return max(1, video_fps // target_fps)


# ---------------------------------------------------------------------------
# ffmpeg-based extraction (the path the background detection worker uses).
# ---------------------------------------------------------------------------
class VideoProbeError(Exception):
    """ffprobe could not read the stream (unreadable / not a video)."""


class FrameReadError(Exception):
    """ffmpeg aborted mid-stream — the run did NOT reach a clean EOF.

    Raising this (instead of silently stopping) is what lets the worker mark
    the session INTERRUPTED and resume, rather than falsely COMPLETED.
    """


def probe_video(video_path: str) -> dict:
    """Return {width, height, duration, fps, nb_frames} via ffprobe.

    Every field is best-effort: a missing/garbage value comes back as 0 rather
    than raising, EXCEPT a totally unreadable stream which raises
    VideoProbeError. `fps` is the source's average rate (informational only —
    sampling is done by time, not by this number).
    """
    cmd = [
        "ffprobe",
        "-v", "error",
        "-select_streams", "v:0",
        "-show_entries",
        "stream=width,height,avg_frame_rate,nb_frames,duration:format=duration",
        "-of", "json",
        video_path,
    ]
    try:
        out = subprocess.run(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=120
        )
    except (FileNotFoundError, subprocess.SubprocessError) as exc:
        raise VideoProbeError(f"ffprobe failed: {exc}") from exc

    if out.returncode != 0:
        raise VideoProbeError(
            f"ffprobe exited {out.returncode}: {out.stderr.decode('utf-8', 'replace')[:300]}"
        )

    try:
        data = json.loads(out.stdout or b"{}")
    except json.JSONDecodeError as exc:
        raise VideoProbeError(f"ffprobe returned non-JSON: {exc}") from exc

    streams = data.get("streams") or []
    if not streams:
        raise VideoProbeError("no video stream found")
    stream = streams[0]

    def _fraction(value: str) -> float:
        # avg_frame_rate is "30000/1001" etc.
        try:
            if not value or value == "0/0":
                return 0.0
            if "/" in value:
                num, den = value.split("/", 1)
                den_f = float(den)
                return float(num) / den_f if den_f else 0.0
            return float(value)
        except (ValueError, ZeroDivisionError):
            return 0.0

    def _to_float(value) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return 0.0

    duration = _to_float(stream.get("duration")) or _to_float(
        (data.get("format") or {}).get("duration")
    )

    return {
        "width": int(_to_float(stream.get("width"))),
        "height": int(_to_float(stream.get("height"))),
        "duration": duration,
        "fps": _fraction(stream.get("avg_frame_rate", "")),
        "nb_frames": int(_to_float(stream.get("nb_frames"))),
    }


class FfmpegFrameReader:
    """Iterate (global_index, timestamp_seconds, bgr_frame) sampled at target fps.

    Pipes raw ``bgr24`` frames from ffmpeg's stdout — one ndarray per yield, no
    accumulation, so memory stays flat no matter how long the video is.

    Resume: ``start_frame`` is in target-fps units (i.e. the number of frames
    already processed). We input-seek to ``start_frame / target_fps`` seconds
    so a resumed run decodes only the remaining tail, not the whole file again.

    After iteration ends, inspect ``clean_eof``:
      - True  -> ffmpeg reached the real end of the stream (exit 0).
      - False -> ffmpeg aborted (non-zero exit / killed); the caller should
                 treat this as INTERRUPTED and resume, NOT as completed.
    ``stderr_tail`` holds the last ffmpeg diagnostics for logging.
    """

    def __init__(
        self,
        video_path: str,
        target_fps: float,
        width: int,
        height: int,
        start_frame: int = 0,
    ):
        if width <= 0 or height <= 0:
            raise FrameReadError(
                f"unusable frame size {width}x{height}; cannot decode raw frames"
            )
        self.video_path = video_path
        self.target_fps = float(target_fps) if target_fps else 1.0
        self.width = int(width)
        self.height = int(height)
        self.start_frame = max(0, int(start_frame))
        self.clean_eof = False
        self.stderr_tail = ""
        self._proc: Optional[subprocess.Popen] = None

    def _build_cmd(self):
        start_seconds = self.start_frame / self.target_fps
        cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error"]
        # Tolerate damaged packets instead of bailing out (the whole point).
        cmd += ["-fflags", "+discardcorrupt", "-err_detect", "ignore_err"]
        # Fast input-side seek for resume (keyframe-accurate; the fps filter
        # re-times from the seek point so global indexing stays consistent).
        if start_seconds > 0:
            cmd += ["-ss", f"{start_seconds:.3f}"]
        cmd += [
            "-i", self.video_path,
            "-vf", f"fps={self.target_fps}",
            "-pix_fmt", "bgr24",
            "-f", "rawvideo",
            "-",
        ]
        return cmd

    def __iter__(self):
        frame_bytes = self.width * self.height * 3
        try:
            self._proc = subprocess.Popen(
                self._build_cmd(),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=frame_bytes,
            )
        except FileNotFoundError as exc:
            raise FrameReadError(
                "ffmpeg not found — install ffmpeg to process videos"
            ) from exc

        index = self.start_frame
        try:
            stdout = self._proc.stdout
            while True:
                buf = _read_exact(stdout, frame_bytes)
                if buf is None:
                    break  # stream ended (clean or not — decided below)
                frame = np.frombuffer(buf, dtype=np.uint8).reshape(
                    (self.height, self.width, 3)
                )
                timestamp = index / self.target_fps
                yield index, timestamp, frame
                index += 1
        finally:
            self._finalize()

    def _finalize(self):
        proc = self._proc
        if proc is None:
            return
        try:
            if proc.stdout:
                proc.stdout.close()
            stderr = b""
            try:
                # Drain stderr and let ffmpeg exit.
                stderr = proc.stderr.read() if proc.stderr else b""
            except Exception:  # noqa: BLE001
                pass
            proc.wait(timeout=30)
            self.stderr_tail = stderr.decode("utf-8", "replace")[-1000:]
            self.clean_eof = proc.returncode == 0
        except subprocess.TimeoutExpired:
            proc.kill()
            self.clean_eof = False
        except Exception:  # noqa: BLE001
            self.clean_eof = False

    def close(self):
        """Kill ffmpeg early (e.g. caller hit an error). Safe to call twice."""
        proc = self._proc
        if proc and proc.poll() is None:
            proc.kill()
            try:
                proc.wait(timeout=10)
            except Exception:  # noqa: BLE001
                pass


def _read_exact(stream, n: int) -> Optional[bytes]:
    """Read exactly n bytes from a pipe, or None at EOF.

    A pipe read can return a short chunk; rawvideo frames must be whole, so we
    loop until we have a full frame. A trailing partial read (truncated final
    frame) is discarded as EOF.
    """
    chunks = []
    remaining = n
    while remaining > 0:
        chunk = stream.read(remaining)
        if not chunk:
            return None if remaining == n else None
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)
