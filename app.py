import os
import tempfile
from pathlib import Path

import cv2
import pandas as pd
import plotly.express as px
import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode

from src.traffic_processor import (
    TrafficProcessor,
    ExternalCameraProcessor,
    get_shared_state,
    reset_shared_state,
    process_uploaded_video,
)

st.set_page_config(
    page_title="5G Smart Campus Traffic Analytics",
    page_icon=":material/traffic:",
    layout="wide",
    initial_sidebar_state="expanded",
)

LIVE_LABELS = [
    ("Cars", "car"),
    ("Motorcycles", "motorcycle"),
    ("Buses", "bus"),
    ("Trucks", "truck"),
    ("Bicycles", "bicycle"),
    ("Pedestrians", "person"),
]

state = get_shared_state()

# Session-state holder for an OpenCV USB/IP camera. It is separate from WebRTC.
if "external_camera" not in st.session_state:
    st.session_state.external_camera = None
if "external_camera_source_key" not in st.session_state:
    st.session_state.external_camera_source_key = None

with st.sidebar:
    st.title("Campus traffic", icon=":material/traffic:")
    st.caption("5G mobility analytics")
    page = st.radio("Navigation", ["Traffic Monitoring", "Project Overview"])
    st.divider()
    st.markdown("### Current implementation")
    st.success("Phase 1 · Camera / Video Input")
    st.success("Phase 2 · YOLOv8 Detection")
    st.success("Phase 3 · ByteTrack + Counting")
    st.info("Congestion prediction and 5G/MEC integration are planned for later.")

st.title("5G smart campus traffic", icon=":material/monitoring:")
st.caption("Live detection, object tracking, and line-crossing analytics")

if page == "Project Overview":
    st.markdown("### Project objective")
    st.write(
        "The system detects people and vehicles, assigns tracking IDs, and counts each tracked "
        "object when it crosses a virtual line. Test it with the laptop webcam, an external USB/IP "
        "CCTV stream, or an uploaded traffic video."
    )
    st.markdown("### Processing architecture")
    st.code(
        "Laptop Webcam (WebRTC) / USB CCTV / IP CCTV (RTSP) / Uploaded Video\n"
        "                         ↓\n"
        "                       OpenCV\n"
        "                         ↓\n"
        "                       YOLOv8\n"
        "                         ↓\n"
        "                      ByteTrack\n"
        "                         ↓\n"
        "            Virtual Line Crossing + Counting\n"
        "                         ↓\n"
        "                 Streamlit Dashboard",
        language="text",
    )
    st.markdown("### Supported object classes")
    st.write("Cars, motorcycles, buses, trucks, bicycles and pedestrians.")
    st.warning(
        "This is currently a local prototype. Real 5G transmission, MEC deployment, "
        "congestion estimation, prediction and alerts are not implemented."
    )
else:
    mode = st.radio(
        "Choose input source",
        ["📹 Live Camera", "📁 Upload Video"],
        horizontal=True,
        key="input_mode",
    )
    st.divider()

    if mode == "📹 Live Camera":
        camera_source = st.radio(
            "Select camera device",
            ["Laptop Camera (Built-in)", "External USB Camera", "Sparsh IP CCTV (RTSP)"],
            horizontal=True,
            key="camera_device_type",
        )

        if camera_source == "Laptop Camera (Built-in)":
            # If an OpenCV camera was active, stop it before switching to WebRTC.
            if st.session_state.external_camera is not None:
                st.session_state.external_camera.stop()
                st.session_state.external_camera = None
                st.session_state.external_camera_source_key = None

            left, right = st.columns([1.7, 1])
            with left:
                st.markdown("### 💻 Laptop webcam monitoring")
                st.caption("Press START and allow camera access in your browser.")
                ctx = webrtc_streamer(
                    key="campus-traffic-webcam-dual",
                    mode=WebRtcMode.SENDRECV,
                    video_processor_factory=lambda: TrafficProcessor(state),
                    media_stream_constraints={"video": True, "audio": False},
                    async_processing=True,
                )
                if ctx.state.playing:
                    st.success("Laptop camera is running. Objects crossing the red line are counted.")
                else:
                    st.info("Camera is stopped. Press START above the video to begin.")

            with right:
                st.markdown("### 📊 Live count summary")

                @st.fragment(run_every="1s")
                def show_laptop_metrics():
                    snapshot = state.snapshot()
                    total = sum(snapshot["counts"].values())
                    a, b = st.columns(2)
                    a.metric("Total crossings", total)
                    b.metric("Currently tracked", snapshot["current_objects"])
                    st.markdown("#### Category-wise crossing count")
                    metric_cols = st.columns(2)
                    for i, (label, key) in enumerate(LIVE_LABELS):
                        metric_cols[i % 2].metric(label, snapshot["counts"].get(key, 0))
                    st.caption(f"Model status: {snapshot['model_status']}")
                    if snapshot["last_error"]:
                        st.error(snapshot["last_error"])
                    chart_df = pd.DataFrame([
                        {"Category": label, "Count": snapshot["counts"].get(key, 0)}
                        for label, key in LIVE_LABELS
                    ])
                    fig = px.bar(
                        chart_df, x="Category", y="Count", text="Count",
                        title="Live line-crossing count", color="Category"
                    )
                    fig.update_layout(
                        showlegend=False, height=300,
                        margin=dict(l=10, r=10, t=45, b=10),
                        template="plotly_white",
                        paper_bgcolor="#FFFFFF",
                        plot_bgcolor="#FFFFFF",
                        font=dict(family="Arial, sans-serif", color="#29445A"),
                        xaxis=dict(showgrid=False, title=None),
                        yaxis=dict(gridcolor="#E8EEF2", zeroline=False, title=None),
                    )
                    st.plotly_chart(fig, width="stretch")

                show_laptop_metrics()

        else:
            st.markdown("### 🎥 External camera monitoring")
            if camera_source == "External USB Camera":
                usb_index = st.selectbox(
                    "External USB camera device index",
                    options=[1, 2, 3, 4, 5],
                    index=0,
                    help="Windows mein built-in webcam usually 0 hota hai; external USB camera often 1 hota hai. Indices vary kar sakte hain."
                )
                source_value = int(usb_index)
                source_key = f"usb:{usb_index}"
                camera_name = f"External USB Camera (index {usb_index})"
            else:
                rtsp_url = st.text_input(
                    "Sparsh CCTV RTSP URL",
                    type="password",
                    placeholder="rtsp://username:password@camera-ip:554/stream1",
                    help="Actual RTSP path Sparsh camera/DVR/NVR model aur settings par depend karta hai."
                )
                source_value = rtsp_url.strip()
                source_key = f"rtsp:{source_value}"
                camera_name = "Sparsh IP CCTV"

            c1, c2 = st.columns([1, 1])
            with c1:
                start_clicked = st.button("Start external camera", type="primary", width="stretch")
            with c2:
                stop_clicked = st.button("Stop external camera", width="stretch")

            if stop_clicked:
                if st.session_state.external_camera is not None:
                    st.session_state.external_camera.stop()
                st.session_state.external_camera = None
                st.session_state.external_camera_source_key = None
                st.success("External camera stopped.")

            if start_clicked:
                if camera_source == "Sparsh IP CCTV" and not source_value:
                    st.warning("Please enter the CCTV RTSP URL first.")
                else:
                    if st.session_state.external_camera is not None:
                        st.session_state.external_camera.stop()
                    reset_shared_state()
                    camera = ExternalCameraProcessor(
                        shared_state=state,
                        camera_source=source_value,
                        camera_name=camera_name,
                    )
                    try:
                        camera.start()
                        st.session_state.external_camera = camera
                        st.session_state.external_camera_source_key = source_key
                        st.success(f"Connecting to {camera_name}...")
                    except Exception as exc:
                        st.session_state.external_camera = None
                        st.error(f"Camera connection failed: {type(exc).__name__}: {exc}")

            @st.fragment(run_every="0.3s")
            def show_external_camera():
                camera = st.session_state.get("external_camera")
                if camera is None:
                    st.info("Select a camera and press Start External Camera.")
                    return
                frame = camera.get_frame()
                if frame is not None:
                    st.image(
                        cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
                        channels="RGB",
                        width="stretch",
                    )
                else:
                    st.info("Camera connected. Waiting for video frames...")

            left, right = st.columns([1.7, 1])
            with left:
                st.markdown("#### Live camera feed")
                show_external_camera()
            with right:
                st.markdown("#### 📊 Live count summary")

                @st.fragment(run_every="1s")
                def show_external_metrics():
                    snapshot = state.snapshot()
                    total = sum(snapshot["counts"].values())
                    a, b = st.columns(2)
                    a.metric("Total crossings", total)
                    b.metric("Currently tracked", snapshot["current_objects"])
                    st.markdown("##### Category-wise crossing count")
                    metric_cols = st.columns(2)
                    for i, (label, key) in enumerate(LIVE_LABELS):
                        metric_cols[i % 2].metric(label, snapshot["counts"].get(key, 0))
                    st.caption(f"Model status: {snapshot['model_status']}")
                    if snapshot["last_error"]:
                        st.error(snapshot["last_error"])

                show_external_metrics()

            if st.button("↻ Reset external camera counts"):
                reset_shared_state()
                st.rerun()

            st.caption(
                "USB camera ke liye device index select karein. IP CCTV ke liye valid RTSP URL "
                "aur same/reachable network required hai. Ek time par ek hi camera source run karein."
            )

    else:
        # Stop an external camera before switching to uploaded-video mode.
        if st.session_state.external_camera is not None:
            st.session_state.external_camera.stop()
            st.session_state.external_camera = None
            st.session_state.external_camera_source_key = None

        st.markdown("### 📁 Upload a traffic video")
        st.write("Choose a video from your laptop. The app will process its frames and create an annotated output video.")
        uploaded_file = st.file_uploader(
            "Choose video",
            type=["mp4", "avi", "mov", "mkv"],
            accept_multiple_files=False,
            help="Recommended: a short traffic video (10–60 seconds) for the first test.",
        )

        if uploaded_file is not None:
            st.video(uploaded_file)
            st.caption(f"Selected file: {uploaded_file.name} · {uploaded_file.size / (1024 * 1024):.2f} MB")

            if st.button("Process uploaded video", type="primary", width="content"):
                suffix = Path(uploaded_file.name).suffix.lower() or ".mp4"
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as input_tmp:
                    input_tmp.write(uploaded_file.getbuffer())
                    input_path = input_tmp.name

                output_path = str(
                    Path(tempfile.gettempdir()) /
                    f"campus_traffic_annotated_{os.getpid()}.mp4"
                )
                progress = st.progress(0, text="Preparing video...")
                status = st.empty()

                def update_progress(value, message):
                    progress.progress(min(max(int(value), 0), 100), text=message)
                    status.write(message)

                try:
                    result = process_uploaded_video(
                        input_path=input_path,
                        output_path=output_path,
                        progress_callback=update_progress,
                    )
                    st.session_state["uploaded_video_result"] = {
                        "output_path": output_path,
                        "counts": result["counts"],
                        "unique_objects": result["unique_objects"],
                        "frames": result["frames"],
                        "fps": result["fps"],
                        "duration": result["duration"],
                        "filename": uploaded_file.name,
                    }
                    progress.progress(100, text="Video processing completed.")
                    status.success("Video analysis completed successfully.")
                except Exception as exc:
                    st.error(f"Video processing failed: {type(exc).__name__}: {exc}")
                finally:
                    try:
                        os.remove(input_path)
                    except OSError:
                        pass

        result = st.session_state.get("uploaded_video_result")
        if result:
            st.divider()
            st.markdown("## 📊 Video analysis result")
            st.caption(
                f"Processed: {result['filename']} · "
                f"{result['frames']} frames · "
                f"{result['fps']:.1f} FPS · "
                f"{result['duration']:.1f} seconds"
            )

            total_crossings = sum(result["counts"].values())
            m1, m2, m3 = st.columns(3)
            m1.metric("Total line crossings", total_crossings)
            m2.metric("Unique tracked objects", result["unique_objects"])
            m3.metric("Frames processed", result["frames"])

            count_df = pd.DataFrame([
                {"Category": label, "Count": result["counts"].get(key, 0)}
                for label, key in LIVE_LABELS
            ])
            chart_col, table_col = st.columns([1.5, 1])
            with chart_col:
                fig = px.bar(
                    count_df, x="Category", y="Count", text="Count",
                    title="Category-wise line-crossing count", color="Category"
                )
                fig.update_layout(
                    showlegend=False, height=360,
                    margin=dict(l=10, r=10, t=45, b=10),
                    template="plotly_white",
                    paper_bgcolor="#FFFFFF",
                    plot_bgcolor="#FFFFFF",
                    font=dict(family="Arial, sans-serif", color="#29445A"),
                    xaxis=dict(showgrid=False, title=None),
                    yaxis=dict(gridcolor="#E8EEF2", zeroline=False, title=None),
                )
                st.plotly_chart(fig, width="stretch")
            with table_col:
                st.markdown("#### Count summary")
                st.dataframe(count_df, hide_index=True, width="stretch")

            output_path = result["output_path"]
            if os.path.exists(output_path):
                st.markdown("### 🎬 Processed video")
                st.video(output_path)
                with open(output_path, "rb") as video_file:
                    st.download_button(
                        "⬇ Download annotated video",
                        data=video_file.read(),
                        file_name="campus_traffic_annotated.mp4",
                        mime="video/mp4",
                    )

            st.caption(
                "Uploaded-video counts are based on objects crossing the virtual line. "
                "Unique tracked objects are shown separately; a vehicle that never crosses "
                "the line will not add to the crossing count."
            )

st.markdown("---")
st.caption(
    "Prototype scope: Phase 1–3 only | Laptop webcam + External USB/IP CCTV + uploaded video | "
    "5G/MEC integration planned for a later phase"
)
