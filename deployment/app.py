import streamlit as st
import cv2
import tempfile
import os
from ultralytics import YOLO
from PIL import Image
import numpy as np

# --- 1. IMPORT MOVIEPY ---
try:
    from moviepy.editor import VideoFileClip
except ImportError:
    st.error("System Error: MoviePy library not found. Please update requirements.txt.")

# --- 2. PAGE CONFIGURATION ---
st.set_page_config(
    page_title="DEPI Graduation Project",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 3. CUSTOM CSS (DARK PROFESSIONAL THEME) ---
st.markdown("""
    <style>
        /* Import Custom Font */
        @import url('https://fonts.cdnfonts.com/css/stella-aesta');

        /* 1. MAIN BACKGROUND */
        .stApp {
            background-color: #0b0c10; /* Deep Carbon Black */
            color: #c5c6c7; /* Light Grey Text */
        }
        
        /* 2. SIDEBAR (CONTROL PANEL) */
        [data-testid="stSidebar"] {
            background-color: #1f2833; /* Dark Slate */
            border-right: 1px solid #45a29e; /* Thin Teal Border */
        }
        
        /* Sidebar Headers */
        [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {
            color: #66fcf1; /* Neon Teal */
            font-family: 'Helvetica Neue', sans-serif;
            text-transform: uppercase;
            font-size: 0.9rem;
            letter-spacing: 0.1em;
            font-weight: 700;
            margin-bottom: 0px !important;
            padding-bottom: 5px !important;
        }
        
        /* Sidebar Labels */
        [data-testid="stSidebar"] label {
            color: #c5c6c7 !important;
            font-size: 0.75rem;
            text-transform: uppercase;
            font-weight: 600;
        }
        
        /* Sidebar Spacing */
        [data-testid="stSidebar"] div[data-testid="stVerticalBlock"] > div {
            gap: 0.7rem !important; /* Increased gap slightly for better readability */
        }

        /* 3. TYPOGRAPHY */
        /* Main Title with Custom Font */
        h1 {
            font-family: 'Stella Aesta', sans-serif !important;
            color: #66fcf1; 
            font-weight: normal;
            font-size: 3.5rem !important;
            text-shadow: 0px 0px 10px rgba(102, 252, 241, 0.3);
        }
        
        h2, h3, h4 {
            font-family: 'Helvetica Neue', sans-serif;
            color: #ffffff;
            font-weight: 600;
        }
        
        /* 4. PROFESSIONAL BUTTONS (Increased Size) */
        .stButton button {
            background-color: transparent;
            color: #66fcf1;
            border: 1px solid #45a29e;
            border-radius: 0px; 
            padding: 0.6rem 1.2rem; /* INCREASED PADDING HERE */
            font-family: 'Courier New', monospace;
            text-transform: uppercase;
            font-weight: bold;
            letter-spacing: 0.1em;
            transition: all 0.3s ease;
            width: 100%;
            font-size: 0.9rem; /* Slightly larger text */
        }
        
        .stButton button:hover {
            background-color: #45a29e;
            color: #0b0c10;
            box-shadow: 0 0 15px rgba(69, 162, 158, 0.7);
            border-color: #66fcf1;
        }

        /* 5. FILE UPLOADER STYLE */
        [data-testid="stFileUploader"] {
            background-color: #1f2833;
            border: 1px dashed #45a29e;
            border-radius: 4px;
            padding: 20px;
        }
        
        /* 6. METRICS */
        [data-testid="stMetricLabel"] {
            color: #45a29e;
            font-size: 0.8rem;
            text-transform: uppercase;
        }
        [data-testid="stMetricValue"] {
            color: #ffffff;
            font-family: 'Courier New', monospace;
        }

        /* 7. SLIDER COLOR CUSTOMIZATION (Default Red) */
        /* We are letting Streamlit handle the slider colors natively */

    </style>
""", unsafe_allow_html=True)

# --- 4. MODEL LOADING ---
@st.cache_resource
def load_model(model_path="best.pt"):
    if not os.path.exists(model_path):
        st.error(f"Critical Error: Model file '{model_path}' not found.")
        return None
    try:
        return YOLO(model_path)
    except Exception as e:
        st.error(f"Model Load Error: {e}")
        return None

def convert_video_to_h264(input_path, output_path):
    try:
        clip = VideoFileClip(input_path)
        if clip.h > 720:
            clip = clip.resize(height=720)
        clip.write_videofile(output_path, codec="libx264", audio=False, logger=None, preset="ultrafast")
        clip.close()
        return True
    except Exception as e:
        st.error(f"Encoding Error: {e}")
        return False

# --- 5. MAIN APP LOGIC ---
def main():
    model = load_model()
    
    # --- SIDEBAR: CONTROL PANEL ---
    st.sidebar.header("Control Panel")
    st.sidebar.markdown("---")
    
    # Input Selection - USING SELECTBOX AS REQUESTED
    st.sidebar.subheader("System Input")
    input_type = st.sidebar.selectbox(
        "Choose Data Source", 
        ["Image Analysis", "Video Analysis"], 
        index=0
    )
    
    st.sidebar.markdown("---")
    
    # Model Parameters
    st.sidebar.subheader("Parameters")
    conf_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.35, 0.05)
    
    st.sidebar.markdown("---")
    
    # Status Indicator
    st.sidebar.caption("Status: System Online")
    st.sidebar.caption("v1.0 | YOLOv11n")

    # --- MAIN CONTENT AREA ---
    
    # VIEW: DETECTION INTERFACE (Image/Video)
    st.title("Autonomous Vehicle Object Detection")
    st.markdown("---")
    
    
    # --- IMAGE LOGIC ---
    if input_type == "Image Analysis":
        st.subheader("Image Analysis Module")
        uploaded_file = st.file_uploader("Upload Image File", type=['jpg', 'jpeg', 'png'])
        
        if uploaded_file and model:
            image = Image.open(uploaded_file)
            col1, col2 = st.columns(2)
            
            with col1:
                st.caption("INPUT FEED")
                st.image(image, use_column_width=True)
            
            with col2:
                st.caption("ANALYSIS OUTPUT")
                if st.button("Execute Inference", type="primary"):
                    with st.spinner("Processing neural network..."):
                        results = model.predict(image, conf=conf_threshold)
                        res_plotted = results[0].plot()
                        res_image = cv2.cvtColor(res_plotted, cv2.COLOR_BGR2RGB)
                        
                        st.image(res_image, use_column_width=True)
                        
                        count = len(results[0].boxes)
                        st.info(f"Targets Identified: {count}")

    # --- VIDEO LOGIC ---
    elif input_type == "Video Analysis":
        st.subheader("Video Analysis Module")
        uploaded_video = st.file_uploader("Upload Video File", type=['mp4', 'avi', 'mov', 'mkv'])
        
        if uploaded_video and model:
            # Initialize variables to avoid UnboundLocalError
            video_path = None
            raw_path = None
            final_path = None
            cap = None # Initialize cap to None

            try:
                # Create temp files
                tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
                tfile.write(uploaded_video.read())
                video_path = tfile.name
                
                raw_path = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4').name
                final_path = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4').name

                cap = cv2.VideoCapture(video_path)
                
                if not cap.isOpened():
                    st.error("Error opening video stream.")
                else:
                    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    fps = int(cap.get(cv2.CAP_PROP_FPS))
                    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    
                    st.caption(f"Stream Data: {width}x{height} | {fps} FPS | {total_frames} Frames")
                    
                    if st.button("Initiate Sequence", type="primary"):
                        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                        out = cv2.VideoWriter(raw_path, fourcc, fps, (width, height))
                        
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        stop_button = st.button("Abort Sequence")
                        
                        frame_count = 0
                        while cap.isOpened():
                            ret, frame = cap.read()
                            if not ret: break
                            if stop_button:
                                status_text.warning("Sequence Aborted.")
                                break
                            
                            results = model.predict(frame, conf=conf_threshold, verbose=False)
                            res_plotted = results[0].plot()
                            out.write(res_plotted)
                            
                            frame_count += 1
                            if total_frames > 0:
                                progress_bar.progress(min(frame_count / total_frames, 1.0))
                                status_text.text(f"Scanning Frame {frame_count}/{total_frames}")

                        cap.release()
                        out.release()
                        
                        status_text.text("Encoding stream for playback...")
                        if convert_video_to_h264(raw_path, final_path):
                            status_text.success("Sequence Complete.")
                            
                            with open(final_path, 'rb') as v:
                                video_bytes = v.read()
                            
                            st.video(video_bytes)
                            st.download_button("Export Data", video_bytes, "detection_log.mp4", "video/mp4")
                        else:
                            st.error("Encoding protocol failed.")

            except Exception as e:
                st.error(f"Runtime Error: {e}")
            finally:
                # Robust Cleanup
                try: 
                    if cap is not None:
                        cap.release()
                except: pass
                
                for path in [video_path, raw_path, final_path]:
                    # Only delete if path exists and is not None
                    if path and os.path.exists(path) and path != final_path:
                        try: os.remove(path)
                        except: pass

if __name__ == "__main__":
    main()

