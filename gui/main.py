import io
import json
from typing import Any, List, Literal, Optional, cast

import cv2
import numpy as np
import numpy.typing as npt
import streamlit as st
from PIL import Image
from utils import apply_hsv_filter, capture_frame, list_available_cameras


@st.cache_data(show_spinner=False)
def get_available_cameras() -> List[tuple[int, str]]:
    """Cached camera enumeration so it doesn't re-run on every Streamlit rerun."""
    return list_available_cameras()


def main() -> None:
    st.set_page_config(layout="wide")
    st.title("HSV Color Filter App")

    # Top section for input source and image selection
    source: Optional[Literal["Upload Images", "Camera Feed"]] = st.radio(
        "Select input source:", ("Upload Images", "Camera Feed")
    )
    image: Optional[npt.NDArray[np.uint8]] = None
    uploaded_files: List[Any] = []
    current_image_index: int = 0

    if source == "Upload Images":
        uploaded_files = cast(
            List[Any],
            st.file_uploader(
                "Choose images...",
                type=["jpg", "jpeg", "png"],
                accept_multiple_files=True,
            ),
        )

        if not uploaded_files:
            st.warning("Please upload one or more images.")
            return

        st.success(f"Uploaded {len(uploaded_files)} images.")

        # Image selection controls
        (col1,) = st.columns([2])
        with col1:
            current_image_index = int(
                st.number_input(
                    "Select image index",
                    min_value=0,
                    max_value=len(uploaded_files) - 1,
                    value=0,
                    step=1,
                )
            )

        file = uploaded_files[current_image_index]
        # Use getvalue() rather than read(): Streamlit hands back the same
        # UploadedFile on every rerun, so read() would consume the buffer once
        # and return empty bytes on the next slider change.
        pil_image = Image.open(io.BytesIO(file.getvalue()))
        image = np.array(pil_image.convert("RGB"), dtype=np.uint8)
        image = cast(npt.NDArray[np.uint8], cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

    else:  # source == "Camera Feed"
        if st.button("Refresh camera list"):
            get_available_cameras.clear()
        available_cameras: List[tuple[int, str]] = get_available_cameras()
        if not available_cameras:
            st.error("No cameras detected.")
            return
        selected_camera: tuple[int, str] = st.selectbox(
            "Select camera",
            options=available_cameras,
            format_func=lambda camera: camera[1],
        )
        camera_index: int = selected_camera[0]

        # macOS sometimes maps a camera name to the wrong device index, so the
        # built-in camera's light flicks on and an iPhone Continuity Camera
        # opens instead. This lets you try indices until the right one opens.
        if st.checkbox("Wrong camera opens? Set the index manually"):
            camera_index = int(
                st.number_input(
                    "Camera index",
                    min_value=0,
                    max_value=10,
                    value=int(camera_index),
                    step=1,
                    key="camera_index_override",
                    help="Try 0, 1, 2… until the built-in camera's light stays "
                    "on and you get a live image.",
                )
            )

        frame, error = capture_frame(camera_index)
        if error == "unopened":
            st.error(
                f"Unable to access camera index {camera_index}. Try a different "
                "index, or 'Refresh camera list' if you plugged/unplugged a "
                "device."
            )
            return
        if error == "black":
            st.error(
                "The camera opened but only returned black frames — it may "
                "still be waking up. Try again, or pick a different camera "
                "index above."
            )
            return
        image = cast(npt.NDArray[np.uint8], frame)

    # Main area for image display and HSV controls
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Original Image")
        st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), use_container_width=True)

    with col2:
        st.subheader("Filtered Image")

        # HSV sliders
        st.sidebar.subheader("HSV Color Filter")
        h_min: int = st.sidebar.slider("Hue Min", 0, 179, 0)
        h_max: int = st.sidebar.slider("Hue Max", 0, 179, 179)
        s_min: int = st.sidebar.slider("Saturation Min", 0, 255, 0)
        s_max: int = st.sidebar.slider("Saturation Max", 0, 255, 255)
        v_min: int = st.sidebar.slider("Value Min", 0, 255, 0)
        v_max: int = st.sidebar.slider("Value Max", 0, 255, 255)

        lower_hsv: npt.NDArray[np.uint8] = np.array(
            [h_min, s_min, v_min], dtype=np.uint8
        )
        upper_hsv: npt.NDArray[np.uint8] = np.array(
            [h_max, s_max, v_max], dtype=np.uint8
        )
        filtered_image: npt.NDArray[np.uint8] = apply_hsv_filter(
            image, lower_hsv, upper_hsv
        )
        st.image(
            cv2.cvtColor(filtered_image, cv2.COLOR_BGR2RGB), use_container_width=True
        )

    # JSON Export and Display
    hsv_values = {
        "h_min": h_min,
        "h_max": h_max,
        "s_min": s_min,
        "s_max": s_max,
        "v_min": v_min,
        "v_max": v_max,
    }
    json_string = json.dumps(hsv_values, indent=2)
    st.sidebar.subheader("HSV Values (JSON)")
    st.sidebar.code(json_string, language="json")

    if st.sidebar.button("Export HSV Values"):
        st.sidebar.download_button(
            label="Download HSV Values as JSON",
            data=json_string,
            file_name="hsv_values.json",
            mime="application/json",
        )


if __name__ == "__main__":
    main()
