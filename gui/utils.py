import json
import platform
import subprocess
import time
import cv2
import numpy as np
from typing import Literal, Optional, cast
import numpy.typing as npt


def _macos_camera_names() -> list[str]:
    """Camera names as macOS reports them, in AVFoundation device order."""
    try:
        output = subprocess.run(
            ["system_profiler", "SPCameraDataType", "-json"],
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        ).stdout
        cameras = json.loads(output).get("SPCameraDataType", [])
        return [cam.get("_name", f"Camera {i}") for i, cam in enumerate(cameras)]
    except Exception:
        return []


def list_available_cameras(max_index: int = 5) -> list[tuple[int, str]]:
    """Return (device_index, label) for each available camera.

    On macOS the device list comes from ``system_profiler`` so cameras get real
    names (e.g. "FaceTime HD Camera" vs an iPhone Continuity Camera) and we avoid
    probing out-of-range indices. Elsewhere we fall back to probing indices.
    """
    if platform.system() == "Darwin":
        names = _macos_camera_names()
        if names:
            return list(enumerate(names))

    cameras: list[tuple[int, str]] = []
    for i in range(max_index):
        cap = _open_capture(i)
        if cap.isOpened():
            cameras.append((i, f"Camera {i}"))
        cap.release()
    return cameras


def _open_capture(index: int) -> cv2.VideoCapture:
    """Open a capture device, preferring the AVFoundation backend on macOS.

    Letting OpenCV pick its default backend on macOS can select the wrong
    device (e.g. an iPhone Continuity Camera instead of the built-in one).
    Requesting AVFoundation explicitly makes the ``index`` map to the same
    device order the OS uses.
    """
    if platform.system() == "Darwin":
        cap = cv2.VideoCapture(index, cv2.CAP_AVFOUNDATION)
        if cap.isOpened():
            return cap
        cap.release()
    return cv2.VideoCapture(index)


def capture_frame(
    index: int,
    warmup_frames: int = 40,
    warmup_delay: float = 0.05,
) -> tuple[Optional[npt.NDArray[np.uint8]], Optional[Literal["unopened", "black"]]]:
    """Capture a single non-black frame from the camera at ``index``.

    Cameras return black frames for a moment after opening while the sensor
    warms up, so we poll for up to ``warmup_frames`` reads (with a short delay
    between each) and only accept a frame that actually contains signal. This
    avoids the "green light flicks on then a black image appears" behaviour.

    Returns ``(frame, None)`` on success, or ``(None, reason)`` where ``reason``
    is ``"unopened"`` (device wouldn't open) or ``"black"`` (opened but never
    produced a real frame).
    """
    cap = _open_capture(index)
    try:
        if not cap.isOpened():
            return None, "unopened"
        for _ in range(warmup_frames):
            ret, frame = cap.read()
            if (
                ret
                and frame is not None
                and cv2.countNonZero(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
            ):
                return cast(npt.NDArray[np.uint8], frame), None
            time.sleep(warmup_delay)
        return None, "black"
    finally:
        cap.release()


def apply_hsv_filter(image: npt.NDArray[np.uint8],
                     lower_hsv: npt.NDArray[np.uint8],
                     upper_hsv: npt.NDArray[np.uint8]) -> npt.NDArray[np.uint8]:

    hsv = cast(npt.NDArray[np.uint8], cv2.cvtColor(image, cv2.COLOR_BGR2HSV))
    mask = cast(npt.NDArray[np.uint8], cv2.inRange(hsv, lower_hsv, upper_hsv))
    result = cast(npt.NDArray[np.uint8],
                  cv2.bitwise_and(image, image, mask=mask))
    return result
