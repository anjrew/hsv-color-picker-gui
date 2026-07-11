import json
import platform
import subprocess
import cv2
import numpy as np
from typing import cast
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
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            cameras.append((i, f"Camera {i}"))
        cap.release()
    return cameras


def apply_hsv_filter(image: npt.NDArray[np.uint8],
                     lower_hsv: npt.NDArray[np.uint8],
                     upper_hsv: npt.NDArray[np.uint8]) -> npt.NDArray[np.uint8]:

    hsv = cast(npt.NDArray[np.uint8], cv2.cvtColor(image, cv2.COLOR_BGR2HSV))
    mask = cast(npt.NDArray[np.uint8], cv2.inRange(hsv, lower_hsv, upper_hsv))
    result = cast(npt.NDArray[np.uint8],
                  cv2.bitwise_and(image, image, mask=mask))
    return result
