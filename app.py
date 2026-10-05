"""
app.py — RoadWatch AI Streamlit Web Application

A portfolio-quality web application for automated detection of potholes and road cracks
in uploaded images and videos using Ultralytics YOLO and OpenCV.

Run locally with:
    streamlit run app.py
"""

import os
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import PIL.Image
import streamlit as st
import torch
from ultralytics import YOLO

# ==============================================================================
# Page Configuration & Custom CSS Styling
# ==============================================================================
st.set_page_config(
    page_title="RoadWatch AI — Damage Detection",
    page_icon="🛣️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
        /* Modern Glassmorphic Container Cards */
        .metric-card {
            background: rgba(255, 255, 255, 0.05);
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 12px;
            padding: 18px;
            text-align: center;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05);
        }
        .metric-value {
            font-size: 2rem;
            font-weight: 700;
            color: #00d2ff;
        }
        .metric-label {
            font-size: 0.9rem;
            color: #a0aec0;
            margin-top: 4px;
        }
        /* Custom Warning & Info Boxes */
        .disclaimer-box {
            background-color: rgba(255, 193, 7, 0.1);
            border-left: 4px solid #ffc107;
            padding: 12px 16px;
            border-radius: 4px;
            margin-bottom: 20px;
            font-size: 0.9rem;
        }
        .success-box {
            background-color: rgba(40, 167, 69, 0.1);
            border-left: 4px solid #28a745;
            padding: 12px 16px;
            border-radius: 4px;
            margin-bottom: 20px;
        }
    </style>
""",
    unsafe_allow_html=True,
)

# Standard class mapping for Road Damage Datasets (RDD2022 & Roboflow)
CLASS_NAME_MAP = {
    "D00": "Longitudinal Crack (D00)",
    "D10": "Transverse Crack (D10)",
    "D20": "Alligator Crack (D20)",
    "D40": "Pothole (D40)",
    "pothole": "Pothole",
    "0": "Longitudinal Crack (D00)",
    "1": "Transverse Crack (D10)",
    "2": "Alligator Crack (D20)",
    "3": "Pothole (D40)",
}


def format_class_name(raw_name: str) -> str:
    """Format raw model class names into human-readable labels."""
    return CLASS_NAME_MAP.get(str(raw_name), str(raw_name).capitalize())


@st.cache_resource
def load_yolo_model(model_path: str):
    """
    Load YOLO model checkpoint from file.
    Cached by Streamlit to prevent reloading model on every UI interaction.
    """
    return YOLO(model_path)


# ==============================================================================
# Main Application Header & Information
# ==============================================================================
def render_header():
    st.title("🛣️ RoadWatch AI: Road Damage Detection System")
    st.caption("AI-powered analysis of road surface defects (Potholes & Cracks) for infrastructure monitoring")

    with st.expander("ℹ️ About & Operating Instructions", expanded=False):
        st.markdown(
            """
            ### Overview
            **RoadWatch AI** is a computer vision prototype designed to assist civil engineers, road maintenance teams, 
            and municipal surveyors in identifying and categorizing road surface hazards.

            ### Supported Damage Categories (RDD2022 standard)
            - **`D00` — Longitudinal Crack**: Cracks running parallel to the direction of traffic.
            - **`D10` — Transverse Crack**: Cracks running perpendicular to the direction of traffic.
            - **`D20` — Alligator Crack**: Interconnected cracks forming a pattern resembling alligator skin.
            - **`D40` — Pothole**: Bowl-shaped depressions formed by water infiltration and traffic stress.

            ### How to Use
            1. Train a model using `python train.py --data path/to/data.yaml` or upload custom `.pt` weights in the sidebar.
            2. Choose **Image Mode** or **Video Mode** from the sidebar.
            3. Upload road imagery or dashcam footage.
            4. Adjust detection parameters (Confidence Threshold & IoU).
            5. Review detection bounding boxes, score breakdowns, and summary tables.
            """
        )

    st.markdown(
        """
        <div class="disclaimer-box">
            ⚠️ <b>Research & Reporting Prototype Disclaimer:</b> This application is built as an educational tool and prototype. 
            It is not certified for automated municipal road inspection, legal assessments, or structural safety determinations. 
            Results depend strictly on training data quality and environmental factors.
        </div>
        """,
        unsafe_allow_html=True,
    )


# ==============================================================================
# Sidebar Configuration & Model Status
# ==============================================================================
def render_sidebar():
    st.sidebar.header("⚙️ Model Configuration")

    default_model_path = Path("models/best.pt")
    model_loaded = False
    model = None
    selected_model_path = None

    if default_model_path.exists():
        st.sidebar.success(f"✅ Loaded trained model: `models/best.pt`")
        selected_model_path = str(default_model_path)
    else:
        st.sidebar.warning("⚠️ Default model (`models/best.pt`) not found.")

    # Option to upload custom weights or specify path
    with st.sidebar.expander("🔧 Model File Settings", expanded=not default_model_path.exists()):
        custom_file = st.file_uploader(
            "Upload YOLO weights (.pt)", type=["pt"], help="Upload a custom trained YOLO PyTorch model file."
        )
        if custom_file:
            temp_weights_dir = Path("temp_models")
            temp_weights_dir.mkdir(exist_ok=True)
            custom_path = temp_weights_dir / custom_file.name
            with open(custom_path, "wb") as f:
                f.write(custom_file.getbuffer())
            selected_model_path = str(custom_path)
            st.sidebar.success(f"✅ Using uploaded weights: `{custom_file.name}`")

    if selected_model_path:
        try:
            model = load_yolo_model(selected_model_path)
            model_loaded = True
        except Exception as e:
            st.sidebar.error(f"❌ Failed to load model: {e}")

    st.sidebar.divider()
    st.sidebar.header("🎛️ Detection Thresholds")
    conf_thresh = st.sidebar.slider(
        "Confidence Threshold",
        min_value=0.05,
        max_value=1.00,
        value=0.25,
        step=0.05,
        help="Minimum confidence score required to display a detection box.",
    )
    iou_thresh = st.sidebar.slider(
        "IoU Threshold (NMS)",
        min_value=0.10,
        max_value=0.95,
        value=0.45,
        step=0.05,
        help="Intersection over Union threshold for Non-Maximum Suppression.",
    )

    st.sidebar.divider()
    st.sidebar.header("📡 Processing Mode")
    app_mode = st.sidebar.radio("Select Input Type", ["📸 Image Detection", "🎥 Video Detection"])

    # Compute Hardware info
    device_info = "CUDA (GPU)" if torch.cuda.is_available() else "CPU"
    st.sidebar.caption(f"💻 Computing Hardware: **{device_info}**")

    return model, model_loaded, conf_thresh, iou_thresh, app_mode


# ==============================================================================
# Image Processing Module
# ==============================================================================
def process_image_mode(model, conf_thresh: float, iou_thresh: float):
    st.subheader("📸 Image Detection Mode")

    uploaded_file = st.file_uploader(
        "Upload a road image (JPG, JPEG, PNG)",
        type=["jpg", "jpeg", "png"],
        help="Select a clear photograph of a road surface.",
    )

    if not uploaded_file:
        st.info("👆 Upload an image above to begin detection.")
        return

    # Load image safely
    try:
        image = PIL.Image.open(uploaded_file).convert("RGB")
    except Exception as e:
        st.error(f"❌ Error loading image file: {e}")
        return

    # Perform inference
    with st.spinner("Processing image with YOLO model..."):
        start_time = time.time()
        results = model.predict(source=image, conf=conf_thresh, iou=iou_thresh, verbose=False)
        inference_time = (time.time() - start_time) * 1000  # ms

    res = results[0]
    annotated_bgr = res.plot()  # BGR array returned by Ultralytics
    annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)

    # Extract detection details
    boxes = res.boxes
    detections = []

    if boxes is not None and len(boxes) > 0:
        for box in boxes:
            cls_id = int(box.cls[0].item())
            raw_class_name = model.names[cls_id] if hasattr(model, "names") else str(cls_id)
            human_class_name = format_class_name(raw_class_name)
            confidence = float(box.conf[0].item())
            xyxy = box.xyxy[0].tolist()

            detections.append(
                {
                    "Class Code": raw_class_name,
                    "Damage Type": human_class_name,
                    "Confidence": round(confidence, 4),
                    "Confidence (%)": f"{confidence * 100:.1f}%",
                    "BBox (xmin, ymin, xmax, ymax)": [round(val, 1) for val in xyxy],
                }
            )

    # Display Images side by side
    col1, col2 = st.columns(2)
    with col1:
        st.image(image, caption="Original Input Image", use_container_width=True)
    with col2:
        st.image(annotated_rgb, caption="Annotated Detection Result", use_container_width=True)

    # Summary Metrics
    st.divider()
    st.subheader("📊 Detection Summary & Analytics")

    df_detections = pd.DataFrame(detections)

    m1, m2, m3 = st.columns(3)
    m1.metric("Total Defects Detected", len(detections))
    m2.metric(
        "Max Confidence Score",
        f"{df_detections['Confidence'].max() * 100:.1f}%" if not df_detections.empty else "N/A",
    )
    m3.metric("Inference Speed", f"{inference_time:.1f} ms")

    if not df_detections.empty:
        st.markdown("#### Detected Damage Breakdown")
        col_chart, col_table = st.columns([1, 2])

        with col_chart:
            counts = df_detections["Damage Type"].value_counts()
            st.bar_chart(counts)

        with col_table:
            st.dataframe(
                df_detections[["Damage Type", "Confidence (%)", "BBox (xmin, ymin, xmax, ymax)"]],
                use_container_width=True,
            )

        # Download option for CSV
        csv_data = df_detections.to_csv(index=False).encode("utf-8")
        st.download_button(
            "📥 Download Detections CSV",
            data=csv_data,
            file_name="roadwatch_image_detections.csv",
            mime="text/csv",
        )
    else:
        st.success("✅ No road damage detected in this image at the selected confidence threshold.")


# ==============================================================================
# Video Processing Module
# ==============================================================================
def process_video_mode(model, conf_thresh: float, iou_thresh: float):
    st.subheader("🎥 Video Detection Mode")

    uploaded_video = st.file_uploader(
        "Upload a road video (MP4, MOV, AVI)",
        type=["mp4", "mov", "avi"],
        help="Select dashcam footage or drone video of a road.",
    )

    if not uploaded_video:
        st.info("👆 Upload a video file above to begin processing.")
        return

    # User controls for video processing limits
    c1, c2 = st.columns(2)
    with c1:
        max_frames = st.slider(
            "Max Frames to Process",
            min_value=30,
            max_value=1500,
            value=300,
            step=30,
            help="Limit the number of processed frames to prevent browser timeout on large videos.",
        )
    with c2:
        frame_stride = st.slider(
            "Frame Step Stride",
            min_value=1,
            max_value=5,
            value=1,
            step=1,
            help="Process every N-th frame (higher stride speeds up processing).",
        )

    if not st.button("🚀 Start Video Analysis"):
        st.info("Click 'Start Video Analysis' to begin frame-by-frame detection.")
        return

    # Use temporary files for input and output video processing
    with tempfile.TemporaryDirectory() as temp_dir:
        input_video_path = os.path.join(temp_dir, "temp_input_video.mp4")
        output_video_path = os.path.join(temp_dir, "temp_output_video.mp4")

        # Save uploaded bytes to temp file
        with open(input_video_path, "wb") as f:
            f.write(uploaded_video.read())

        cap = cv2.VideoCapture(input_video_path)
        if not cap.isOpened():
            st.error("❌ Error: Could not open the uploaded video file.")
            return

        fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_stream_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        frames_to_process = min(total_stream_frames, max_frames) if total_stream_frames > 0 else max_frames

        # Define VideoWriter (mp4v codec)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

        progress_bar = st.progress(0, text="Initializing video inference...")
        status_text = st.empty()

        processed_frame_count = 0
        total_detections_count = 0
        damage_class_counts = {}
        start_processing_time = time.time()

        current_frame_idx = 0

        while cap.isOpened() and processed_frame_count < frames_to_process:
            ret, frame = cap.read()
            if not ret:
                break

            current_frame_idx += 1
            if current_frame_idx % frame_stride != 0:
                continue

            # Run detection on frame
            results = model.predict(source=frame, conf=conf_thresh, iou=iou_thresh, verbose=False)
            res = results[0]
            annotated_frame = res.plot()

            # Record stats
            boxes = res.boxes
            if boxes is not None and len(boxes) > 0:
                total_detections_count += len(boxes)
                for box in boxes:
                    cls_id = int(box.cls[0].item())
                    raw_name = model.names[cls_id] if hasattr(model, "names") else str(cls_id)
                    h_name = format_class_name(raw_name)
                    damage_class_counts[h_name] = damage_class_counts.get(h_name, 0) + 1

            # Write frame to output video
            out.write(annotated_frame)
            processed_frame_count += 1

            # Update UI progress
            elapsed = time.time() - start_processing_time
            current_fps = round(processed_frame_count / elapsed, 1) if elapsed > 0 else 0
            progress_pct = min(1.0, processed_frame_count / frames_to_process)

            progress_bar.progress(
                progress_pct,
                text=f"Processing frame {processed_frame_count}/{frames_to_process} ({int(progress_pct * 100)}%)",
            )
            status_text.markdown(
                f"⚡ **Speed**: {current_fps} FPS | 🔍 **Detections so far**: {total_detections_count}"
            )

        cap.release()
        out.release()
        total_time = time.time() - start_processing_time
        avg_fps = round(processed_frame_count / total_time, 1) if total_time > 0 else 0

        progress_bar.progress(1.0, text="Processing completed!")
        status_text.success(f"✅ Video processing complete in {total_time:.1f}s (Average {avg_fps} FPS)")

        # Display Processed Video
        st.divider()
        st.subheader("📺 Processed Video Result")

        if os.path.exists(output_video_path) and os.path.getsize(output_video_path) > 0:
            with open(output_video_path, "rb") as video_bytes:
                st.video(video_bytes.read())
        else:
            st.warning("⚠️ Unable to render output video stream.")

        # Summary Metrics
        st.subheader("📊 Video Analysis Summary")
        v1, v2, v3, v4 = st.columns(4)
        v1.metric("Processed Frames", processed_frame_count)
        v2.metric("Total Defects Detected", total_detections_count)
        v3.metric("Processing Speed", f"{avg_fps} FPS")
        v4.metric(
            "Defects / Frame",
            f"{round(total_detections_count / processed_frame_count, 2)}" if processed_frame_count > 0 else "0",
        )

        if damage_class_counts:
            st.markdown("#### Video Defect Distribution")
            df_video_stats = pd.DataFrame(
                list(damage_class_counts.items()), columns=["Damage Category", "Total Count"]
            )
            st.bar_chart(df_video_stats.set_index("Damage Category"))
            st.dataframe(df_video_stats, use_container_width=True)


# ==============================================================================
# Main Entry Point
# ==============================================================================
def main():
    render_header()
    model, model_loaded, conf_thresh, iou_thresh, app_mode = render_sidebar()

    if not model_loaded:
        st.warning(
            """
            ### ⚠️ Trained Model Missing (`models/best.pt`)
            To use **RoadWatch AI**, you must train a road damage model or upload trained weights.

            #### 🛠️ How to Train a Model:
            1. Download a dataset such as **RDD2022** or **Roboflow Potholes** (see `README.md`).
            2. Organize the dataset into YOLO format with a `data.yaml` configuration.
            3. Run the training script in your terminal:
               ```bash
               python train.py --data path/to/data.yaml --epochs 25
               ```
            4. The script automatically saves the trained model to `models/best.pt`.
            5. Refresh this app or upload your custom `.pt` file in the sidebar!
            """
        )
        return

    if app_mode == "📸 Image Detection":
        process_image_mode(model, conf_thresh, iou_thresh)
    elif app_mode == "🎥 Video Detection":
        process_video_mode(model, conf_thresh, iou_thresh)


if __name__ == "__main__":
    main()
