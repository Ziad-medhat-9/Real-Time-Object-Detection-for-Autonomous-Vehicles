import streamlit as st
import cv2
import tempfile
import os
from ultralytics import YOLO
from PIL import Image
import numpy as np

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Autonomous Vehicle Vision",
    page_icon="🚘",
    layout="wide"
)

# --- 2. MODEL LOADING ---
@st.cache_resource
def load_model(model_path="best.pt"):
    """
    Loads the YOLO model securely. Caches the model to prevent reloading.
    """
    if not os.path.exists(model_path):
        st.error(f"❌ Model file not found! Please upload '{model_path}' to your project folder.")
        return None
    
    try:
        return YOLO(model_path)
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# --- 3. HELPER FUNCTIONS ---
def get_video_writer(output_path, fps, width, height):
    """
    Attempts to initialize a video writer with browser-compatible codecs.
    """
    # Try H.264 (avc1) first - best for browsers (Chrome/Edge)
    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Check if writer opened successfully. If not, fallback to mp4v (Windows local)
    if not out.isOpened():
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    return out

# --- 4. MAIN APP LOGIC ---
def main():
    # Load Model
    model = load_model()
    
    # Sidebar
    st.sidebar.title("⚙️ Detection Settings")
    conf_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.35, 0.05)
    st.sidebar.markdown("---")
    st.sidebar.info("Select **Image** or **Video** mode.")

    # Main Title
    st.title("🚘 Autonomous Vehicle Object Detection")
    st.markdown("Analyze road scenes for cars, pedestrians, signs, and more.")

    # Mode Selector
    mode = st.radio("Select Input Type:", ["🖼️ Image", "🎥 Video"], horizontal=True)

    # --- IMAGE MODE ---
    if mode == "🖼️ Image":
        uploaded_file = st.file_uploader("Upload an Image", type=['jpg', 'jpeg', 'png'])
        
        if uploaded_file and model:
            image = Image.open(uploaded_file)
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Original Input")
                st.image(image, use_column_width=True)
            
            if st.button("🔍 Detect Objects", type="primary"):
                with col2:
                    st.subheader("Detection Result")
                    with st.spinner("Analyzing..."):
                        results = model.predict(image, conf=conf_threshold)
                        
                        # Plot returns BGR numpy array
                        res_plotted = results[0].plot()
                        
                        # Convert BGR to RGB for Streamlit display
                        res_image = cv2.cvtColor(res_plotted, cv2.COLOR_BGR2RGB)
                        
                        st.image(res_image, use_column_width=True)
                        st.success(f"✅ Found {len(results[0].boxes)} objects.")

    # --- VIDEO MODE ---
    elif mode == "🎥 Video":
        uploaded_video = st.file_uploader("Upload a Video", type=['mp4', 'avi', 'mov', 'mkv'])
        
        if uploaded_video and model:
            # Create temp files
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
            tfile.write(uploaded_video.read())
            video_path = tfile.name
            
            output_tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
            output_path = output_tfile.name

            try:
                cap = cv2.VideoCapture(video_path)
                
                if not cap.isOpened():
                    st.error("Error opening video file.")
                else:
                    # Video Metadata
                    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    fps = int(cap.get(cv2.CAP_PROP_FPS))
                    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    
                    st.info(f"Loaded: {width}x{height} @ {fps} FPS | {total_frames} Frames")
                    
                    if st.button("▶️ Start Processing", type="primary"):
                        out = get_video_writer(output_path, fps, width, height)
                        
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        stop_button = st.button("⏹️ Stop Processing")
                        
                        frame_count = 0
                        
                        while cap.isOpened():
                            ret, frame = cap.read()
                            if not ret:
                                break
                            
                            if stop_button:
                                status_text.warning("Processing stopped by user.")
                                break
                            
                            # YOLO Prediction
                            results = model.predict(frame, conf=conf_threshold, verbose=False)
                            res_plotted = results[0].plot()
                            
                            # Write BGR frame to video
                            out.write(res_plotted)
                            
                            # UI Updates
                            frame_count += 1
                            if total_frames > 0:
                                progress_bar.progress(min(frame_count / total_frames, 1.0))
                                status_text.text(f"Processing Frame {frame_count}/{total_frames}")

                        # Cleanup resources
                        cap.release()
                        out.release()
                        
                        # Success Logic
                        status_text.success("✅ Processing Complete!")
                        st.subheader("Processed Video")
                        
                        # Read binary for display/download
                        with open(output_path, 'rb') as v:
                            video_bytes = v.read()
                        
                        st.video(video_bytes)
                        
                        st.download_button(
                            label="⬇️ Download Result",
                            data=video_bytes,
                            file_name="autonomous_result.mp4",
                            mime="video/mp4"
                        )

            except Exception as e:
                st.error(f"An error occurred: {e}")
            finally:
                # Always clean up input temp file
                if os.path.exists(video_path):
                    os.remove(video_path)
                # Note: We keep output_path briefly for download, Streamlit cleans temp eventually

if __name__ == "__main__":
    main()