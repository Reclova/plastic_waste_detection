import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from pathlib import Path
import tempfile

# Page Configuration
st.set_page_config(
    page_title="Plastic Waste Detection",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .metric-card {
        background-color: #f0f2f6;
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .header-title {
        color: #1f77b4;
        font-size: 2.5em;
        font-weight: bold;
        text-align: center;
        margin-bottom: 10px;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="header-title">♻️ Plastic Waste Detection</div>', unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: gray;'>YOLO-based Object Detection | GPU Acceleration</p>", unsafe_allow_html=True)

# Load Model
@st.cache_resource
def load_model():
    return YOLO("weights/best.pt")

try:
    model = load_model()
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# Sidebar Settings
st.sidebar.header("⚙️ Settings")

conf_threshold = st.sidebar.slider(
    "Confidence Threshold",
    min_value=0.3,
    max_value=0.95,
    value=0.65,
    step=0.05,
    help="Semakin tinggi nilai, semakin ketat dalam mendeteksi objek"
)

iou_threshold = st.sidebar.slider(
    "IOU Threshold",
    min_value=0.3,
    max_value=0.9,
    value=0.5,
    step=0.05,
    help="Non-Maximum Suppression threshold"
)

device = st.sidebar.selectbox(
    "Select Device",
    ["0 (GPU)", "cpu"],
    help="Pilih device untuk inference"
)

device_value = 0 if device == "0 (GPU)" else "cpu"

# Mode Selection
st.sidebar.header("📋 Mode Deteksi")
mode = st.sidebar.radio(
    "Pilih Mode",
    ("🖼️ Upload Gambar", "🎥 Upload Video", "📷 Real-time Webcam")
)

if mode == "🖼️ Upload Gambar":
    st.header("Deteksi Gambar")
    
    uploaded_image = st.file_uploader(
        "Upload gambar (JPG, PNG, JPEG)",
        type=["jpg", "png", "jpeg"]
    )

    if uploaded_image is not None:
        # Read Image
        image_bytes = uploaded_image.read()
        image = cv2.imdecode(
            np.frombuffer(image_bytes, np.uint8),
            cv2.IMREAD_COLOR
        )

        # Run Detection
        results = model(image, conf=conf_threshold, iou=iou_threshold, device=device_value)
        annotated = results[0].plot()

        # Display Results
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("📸 Gambar Original")
            st.image(
                cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
                use_container_width=True
            )

        with col2:
            st.subheader("🎯 Hasil Deteksi")
            st.image(
                cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
                use_container_width=True
            )

        # Display Statistics
        st.divider()
        detections = results[0].boxes
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Deteksi", len(detections))
        
        with col2:
            if len(detections) > 0:
                avg_conf = np.mean([float(conf) for conf in detections.conf])
                st.metric("Avg Confidence", f"{avg_conf:.2%}")
            else:
                st.metric("Avg Confidence", "N/A")
        
        with col3:
            st.metric("Image Size", f"{image.shape[1]}x{image.shape[0]}")
        
        with col4:
            st.metric("Conf Threshold", f"{conf_threshold:.2f}")

        # Detailed Detection Info
        if len(detections) > 0:
            st.subheader("📊 Detail Deteksi")
            detection_data = []
            
            for i, (box, conf, cls) in enumerate(zip(detections.xyxy, detections.conf, detections.cls)):
                detection_data.append({
                    "ID": i + 1,
                    "Class": int(cls.item()),
                    "Confidence": f"{float(conf):.2%}",
                    "X1": int(box[0]),
                    "Y1": int(box[1]),
                    "X2": int(box[2]),
                    "Y2": int(box[3])
                })
            
            st.dataframe(detection_data, use_container_width=True, hide_index=True)


elif mode == "🎥 Upload Video":
    st.header("Deteksi Video")
    
    uploaded_video = st.file_uploader(
        "Upload video (MP4, AVI, MOV, MKV)",
        type=["mp4", "avi", "mov", "mkv"]
    )

    if uploaded_video is not None:
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("📹 Video Original")
            st.video(uploaded_video)

        with col2:
            st.subheader("⚙️ Processing")
            progress_placeholder = st.empty()
            status_placeholder = st.empty()
            
            # Save uploaded video
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_file:
                tmp_file.write(uploaded_video.getbuffer())
                video_path = tmp_file.name

            try:
                # Open Video
                cap = cv2.VideoCapture(video_path)
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                fps = cap.get(cv2.CAP_PROP_FPS)
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

                # Create Output Video dengan codec H.264
                output_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
                fourcc = cv2.VideoWriter_fourcc(*'H264')
                out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

                frame_count = 0
                
                with status_placeholder.container():
                    st.info(f"Total Frames: {total_frames} | FPS: {fps:.2f}")

                # Process Frames
                frames_data = []
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break

                    # Run Detection
                    results = model(frame, conf=conf_threshold, iou=iou_threshold, device=device_value)
                    annotated = results[0].plot()
                    frames_data.append(annotated)
                    out.write(annotated)

                    frame_count += 1
                    progress = frame_count / total_frames
                    progress_placeholder.progress(progress, f"Processing: {frame_count}/{total_frames} frames")

                cap.release()
                out.release()

                # Ensure file is properly closed
                import time
                time.sleep(1)

                # Verify file exists and has content
                if Path(output_path).exists() and Path(output_path).stat().st_size > 0:
                    status_placeholder.success(f"✅ Video processed successfully!")

                    # Display Output Video
                    st.subheader("🎯 Hasil Deteksi Video")
                    with open(output_path, 'rb') as f:
                        video_bytes = f.read()
                        st.video(video_bytes)

                    # Download Button
                    st.download_button(
                        label="⬇️ Download Hasil Video",
                        data=video_bytes,
                        file_name="detected_video.mp4",
                        mime="video/mp4"
                    )
                else:
                    st.error("❌ Gagal membuat video output. File kosong atau tidak valid.")
                    
            except Exception as e:
                status_placeholder.error(f"❌ Error processing video: {str(e)}")
                st.error(f"Detail error: {str(e)}")
            finally:
                if cap.isOpened():
                    cap.release()
                if out.isOpened():
                    out.release()


elif mode == "📷 Real-time Webcam":
    st.header("Real-time Detection")
    
    col1, col2 = st.columns([3, 1])

    with col1:
        FRAME_WINDOW = st.image([])
    
    with col2:
        st.subheader("📊 Stats")
        fps_placeholder = st.empty()
        detection_placeholder = st.empty()

    start_button = st.button(
        "▶️ Start Webcam",
        key="start_webcam",
        use_container_width=True
    )

    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        st.error("❌ Webcam tidak dapat diakses. Pastikan device terhubung.")
    else:
        if start_button:
            frame_count = 0
            import time
            
            stop_button = st.button(
                "⏹️ Stop Webcam",
                key="stop_webcam",
                use_container_width=True
            )

            placeholder_stop = st.empty()
            
            while True:
                ret, frame = cap.read()
                
                if not ret:
                    st.warning("⚠️ Gagal membaca frame dari webcam")
                    break

                # Run Detection
                start_time = time.time()
                results = model(frame, conf=conf_threshold, iou=iou_threshold, device=device_value)
                annotated = results[0].plot()
                inference_time = time.time() - start_time
                fps = 1 / inference_time if inference_time > 0 else 0

                # Update Display
                FRAME_WINDOW.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB))
                
                fps_placeholder.metric("FPS", f"{fps:.1f}")
                detection_placeholder.metric("Deteksi", len(results[0].boxes))

                frame_count += 1

                # Check for stop button
                if placeholder_stop.button("⏹️ Stop", key=f"stop_{frame_count}"):
                    break

        cap.release()
        st.info("Webcam ditutup")