import streamlit as st
import numpy as np
from ultralytics import YOLO
from pathlib import Path
import tempfile
from PIL import Image
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
import av

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
    ("🖼️ Upload Gambar", "🎥 Upload Video", "📷 Real-time WebRTC")
)

# ============== IMAGE DETECTION ==============
if mode == "🖼️ Upload Gambar":
    st.header("Deteksi Gambar")
    
    uploaded_image = st.file_uploader(
        "Upload gambar (JPG, PNG, JPEG)",
        type=["jpg", "png", "jpeg"]
    )

    if uploaded_image is not None:
        # Read Image using PIL
        image = Image.open(uploaded_image).convert("RGB")
        image_array = np.array(image)

        # Run Detection
        results = model(image_array, conf=conf_threshold, iou=iou_threshold, device=device_value)
        
        # Get annotated image
        annotated_array = results[0].plot()
        annotated_image = Image.fromarray(annotated_array)

        # Display Results
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("📸 Gambar Original")
            st.image(image, use_container_width=True)

        with col2:
            st.subheader("🎯 Hasil Deteksi")
            st.image(annotated_image, use_container_width=True)

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
            st.metric("Image Size", f"{image_array.shape[1]}x{image_array.shape[0]}")
        
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


# ============== VIDEO DETECTION ==============
elif mode == "🎥 Upload Video":
    st.header("Deteksi Video")
    
    uploaded_video = st.file_uploader(
        "Upload video (MP4, AVI, MOV, MKV)",
        type=["mp4", "avi", "mov", "mkv"]
    )

    if uploaded_video is not None:
        st.warning("⚠️ Video processing pada Streamlit Cloud terbatas. Untuk video besar, gunakan lokal atau gunakan mode gambar.")
        
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("📹 Video Original")
            st.video(uploaded_video)

        with col2:
            st.subheader("⚙️ Processing")
            progress_placeholder = st.empty()
            
            try:
                # Save uploaded video
                with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_file:
                    tmp_file.write(uploaded_video.getbuffer())
                    video_path = tmp_file.name

                # Use ultralytics result.save() untuk hasil video
                results = model.predict(
                    video_path,
                    conf=conf_threshold,
                    iou=iou_threshold,
                    device=device_value,
                    save=True,
                    verbose=False
                )
                
                # Find output video
                output_base = Path("runs/detect")
                if output_base.exists():
                    # Cari folder terbaru
                    folders = sorted(output_base.glob("predict*"), key=lambda x: x.stat().st_mtime, reverse=True)
                    if folders:
                        output_video = folders[0] / Path(video_path).name
                        if output_video.exists():
                            progress_placeholder.success("✅ Video processed successfully!")
                            
                            with open(output_video, 'rb') as f:
                                video_bytes = f.read()
                                st.subheader("🎯 Hasil Deteksi Video")
                                st.video(video_bytes)
                                
                                st.download_button(
                                    label="⬇️ Download Hasil Video",
                                    data=video_bytes,
                                    file_name="detected_video.mp4",
                                    mime="video/mp4"
                                )
                        else:
                            st.error("❌ Gagal menemukan video output")
                    else:
                        st.error("❌ Gagal memproses video")
                else:
                    st.error("❌ Output directory tidak ditemukan")
                    
            except Exception as e:
                st.error(f"❌ Error processing video: {str(e)}")
                st.info("💡 Tip: Untuk video besar, coba gunakan resolusi lebih rendah atau coba mode gambar terlebih dahulu")


# ============== WEBRTC DETECTION ==============
elif mode == "📷 Real-time WebRTC":
    st.header("Real-time Detection (WebRTC)")
    
    st.info(
        "ℹ️ Mode ini menggunakan WebRTC untuk streaming real-time dari webcam Anda.\n"
        "Pastikan browser Anda mengizinkan akses ke webcam."
    )
    
    # RTC Configuration
    rtc_configuration = RTCConfiguration(
        {"iceServers": [{"urls": ["stun:stun1.l.google.com:19302"]}]}
    )
    
    class VideoProcessor:
        def __init__(self):
            self.frame_count = 0
            self.total_detections = 0
        
        def recv(self, frame):
            img = frame.to_ndarray(format="bgr24")
            
            # Run detection
            results = model(
                img,
                conf=conf_threshold,
                iou=iou_threshold,
                device=device_value,
                verbose=False
            )
            
            # Plot results
            annotated = results[0].plot()
            detections = results[0].boxes
            
            self.frame_count += 1
            self.total_detections += len(detections)
            
            return av.VideoFrame.from_ndarray(annotated, format="bgr24")
    
    # WebRTC Streamer
    webrtc_ctx = webrtc_streamer(
        key="plastic-detection-webrtc",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=rtc_configuration,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
        video_processor_factory=VideoProcessor,
    )
    
    # Display stats
    if webrtc_ctx.state.playing:
        st.info("🎥 Kamera aktif - Deteksi sedang berjalan")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Status", "🟢 Aktif")
        with col2:
            st.metric("Confidence", f"{conf_threshold:.2f}")
        with col3:
            st.metric("IOU Threshold", f"{iou_threshold:.2f}")
    else:
        st.warning("⏸️ Tekan tombol 'Start' di atas untuk memulai deteksi")