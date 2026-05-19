import streamlit as st
import cv2
from PIL import Image
import numpy as np
from ultralytics import YOLO
import tempfile
import os
from datetime import datetime
import platform

class PotholeDetectionApp:
    def __init__(self):
        # Load YOLOv8 model with caching
        @st.cache_resource
        def load_model():
            # Check if model exists, if not, download a sample or use local
            model_path = 'poth/best.pt'
            if not os.path.exists(model_path):
                st.error(f"Model not found at {model_path}. Please ensure best.pt is in the poth/ directory.")
                return None
            return YOLO(model_path)
        
        self.model = load_model()
        
    def setup_page(self):
        st.set_page_config(
            page_title="Pothole Detection System",
            page_icon="🕳️",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        # Custom CSS for better styling
        st.markdown("""
        <style>
        .main-header {
            text-align: center;
            padding: 1rem;
            background: linear-gradient(90deg, #1e3c72, #2a5298);
            border-radius: 10px;
            margin-bottom: 2rem;
            color: white;
        }
        .status-box {
            padding: 1rem;
            border-radius: 5px;
            margin: 1rem 0;
        }
        .detection-info {
            background-color: #f0f2f6;
            padding: 1rem;
            border-radius: 5px;
            margin-top: 1rem;
        }
        .success-text {
            color: #00ff00;
            font-weight: bold;
        }
        .warning-text {
            color: #ff4444;
            font-weight: bold;
        }
        </style>
        """, unsafe_allow_html=True)
        
    def resize_with_aspect_ratio(self, image, max_width=800, max_height=600):
        """Resize image while maintaining aspect ratio"""
        h, w = image.shape[:2]
        
        # Calculate scaling factor
        scale = min(max_width / w, max_height / h)
        
        if scale < 1:
            new_w = int(w * scale)
            new_h = int(h * scale)
            return cv2.resize(image, (new_w, new_h))
        return image
    
    def process_image(self, image_file):
        """Process a single image"""
        try:
            # Read image
            image = Image.open(image_file)
            img_array = np.array(image)
            
            # Convert BGR to RGB if needed
            if len(img_array.shape) == 3 and img_array.shape[2] == 3:
                img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
            
            # Run detection
            results = self.model.predict(img_array, verbose=False)
            
            # Process results
            processed_img = results[0].plot()
            processed_img = cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB)
            
            # Count detections
            num_detections = len(results[0].boxes) if results[0].boxes is not None else 0
            
            return img_array, processed_img, num_detections
            
        except Exception as e:
            st.error(f"Failed to process image: {str(e)}")
            return None, None, 0
    
    def process_video_frame(self, frame):
        """Process a single video frame"""
        try:
            # Run detection
            results = self.model.predict(frame, verbose=False)
            processed_frame = results[0].plot()
            processed_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
            num_detections = len(results[0].boxes) if results[0].boxes is not None else 0
            
            return processed_frame, num_detections
        except Exception as e:
            return frame, 0
    
    def run(self):
        self.setup_page()
        
        # Header
        st.markdown('<div class="main-header">', unsafe_allow_html=True)
        st.title("🕳️ Pothole Detection System")
        st.markdown("### Real-time Pothole Detection using YOLOv8")
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Sidebar for controls
        with st.sidebar:
            st.header("🎮 Controls")
            
            # Input selection
            input_type = st.radio(
                "Select Input Type",
                ["📷 Image", "🎥 Video", "📹 Live Camera"],
                index=0
            )
            
            st.markdown("---")
            
            # Model info
            st.header("ℹ️ Model Information")
            st.info("Using YOLOv8 model trained for pothole detection")
            
            # Detection settings
            st.header("⚙️ Detection Settings")
            confidence_threshold = st.slider(
                "Confidence Threshold",
                min_value=0.0,
                max_value=1.0,
                value=0.25,
                step=0.05,
                help="Lower values detect more potholes but may increase false positives"
            )
            
            iou_threshold = st.slider(
                "IOU Threshold",
                min_value=0.0,
                max_value=1.0,
                value=0.45,
                step=0.05,
                help="Intersection over Union threshold for non-maximum suppression"
            )
            
            # Update model thresholds
            if self.model:
                self.model.conf = confidence_threshold
                self.model.iou = iou_threshold
            
            st.markdown("---")
            
            # Usage tips
            with st.expander("💡 Usage Tips"):
                st.markdown("""
                - **Image**: Upload single images for detection
                - **Video**: Upload video files for batch processing
                - **Live Camera**: Real-time detection using your webcam
                - Adjust confidence threshold for better results
                - Higher confidence = fewer but more accurate detections
                """)
            
            st.markdown("---")
            st.caption("Developed with ❤️ using Streamlit & YOLOv8")
        
        # Check if model loaded successfully
        if self.model is None:
            st.error("❌ Model not loaded. Please check if 'poth/best.pt' exists.")
            return
        
        # Main content area
        if "📷 Image" in input_type:
            self.image_detection_tab()
        
        elif "🎥 Video" in input_type:
            self.video_detection_tab()
        
        else:  # Live Camera
            self.live_camera_tab()
    
    def image_detection_tab(self):
        st.header("📷 Image Detection")
        st.markdown("Upload an image to detect potholes")
        
        # File uploader
        uploaded_file = st.file_uploader(
            "Choose an image...",
            type=['jpg', 'jpeg', 'png', 'bmp'],
            help="Supported formats: JPG, JPEG, PNG, BMP"
        )
        
        if uploaded_file is not None:
            # Create columns for side-by-side display
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Original Image")
                original_image = Image.open(uploaded_file)
                st.image(original_image, use_container_width=True)
            
            # Process button
            if st.button("🔍 Detect Potholes", type="primary", use_container_width=True):
                with st.spinner("Processing image..."):
                    # Reset file pointer
                    uploaded_file.seek(0)
                    
                    # Process image
                    original, processed, num_detections = self.process_image(uploaded_file)
                    
                    if processed is not None:
                        with col2:
                            st.subheader("Detection Result")
                            st.image(processed, use_container_width=True)
                        
                        # Display detection info
                        st.markdown('<div class="detection-info">', unsafe_allow_html=True)
                        col_info1, col_info2, col_info3 = st.columns(3)
                        with col_info1:
                            st.metric("Potholes Detected", num_detections)
                        with col_info2:
                            st.metric("Image Size", f"{original.shape[1]}x{original.shape[0]}")
                        with col_info3:
                            st.metric("Confidence", f"{self.model.conf:.2f}")
                        
                        if num_detections > 0:
                            st.success(f"✅ Detected {num_detections} pothole(s) in the image!")
                        else:
                            st.warning("⚠️ No potholes detected. Try lowering the confidence threshold.")
                        st.markdown('</div>', unsafe_allow_html=True)
    
    def video_detection_tab(self):
        st.header("🎥 Video Detection")
        st.markdown("Upload a video file for batch pothole detection")
        
        # File uploader
        uploaded_file = st.file_uploader(
            "Choose a video...",
            type=['mp4', 'avi', 'mov', 'mkv'],
            help="Supported formats: MP4, AVI, MOV, MKV"
        )
        
        if uploaded_file is not None:
            # Save uploaded video to temporary file
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
            tfile.write(uploaded_file.read())
            video_path = tfile.name
            
            # Display video info
            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
            
            st.info(f"📹 Video Info: {total_frames} frames, {fps:.2f} FPS")
            
            # Process video
            if st.button("▶️ Start Video Detection", type="primary", use_container_width=True):
                # Open video
                cap = cv2.VideoCapture(video_path)
                
                # Create placeholders for display
                frame_placeholder = st.empty()
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # Process frames
                frame_count = 0
                detection_counts = []
                
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    # Process frame
                    processed_frame, num_detections = self.process_video_frame(frame)
                    detection_counts.append(num_detections)
                    
                    # Resize for display
                    processed_frame = self.resize_with_aspect_ratio(processed_frame)
                    
                    # Display frame
                    frame_placeholder.image(processed_frame, use_container_width=True)
                    
                    # Update progress
                    frame_count += 1
                    progress = frame_count / total_frames
                    progress_bar.progress(progress)
                    status_text.text(f"Processing frame {frame_count}/{total_frames} - Detections: {num_detections}")
                    
                    # Optional: Add a small delay for better visualization
                    # time.sleep(0.033)  # ~30 FPS
                
                cap.release()
                
                # Display statistics
                st.markdown('<div class="detection-info">', unsafe_allow_html=True)
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Potholes Detected", sum(detection_counts))
                with col2:
                    avg_detections = np.mean(detection_counts) if detection_counts else 0
                    st.metric("Average per Frame", f"{avg_detections:.2f}")
                with col3:
                    st.metric("Total Frames Processed", frame_count)
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Clean up
                os.unlink(video_path)
    
    def live_camera_tab(self):
        st.header("📹 Live Camera Detection")
        st.warning("⚠️ Note: Live camera requires camera access permissions in your browser")
        
        # Camera selection
        camera_source = st.selectbox(
            "Select Camera Source",
            options=[0, 1],
            format_func=lambda x: f"Camera {x}" if x == 0 else f"Camera {x} (if available)"
        )
        
        start_button = st.button("🎥 Start Live Detection", type="primary", use_container_width=True)
        
        if start_button:
            # Initialize camera
            cap = cv2.VideoCapture(camera_source)
            
            if not cap.isOpened():
                st.error("❌ Could not open camera. Please check:")
                st.markdown("""
                - Camera permissions in your browser
                - Camera is not being used by another application
                - Camera drivers are properly installed
                """)
            else:
                st.success("✅ Camera opened successfully! Detection running...")
                
                # Create placeholder for video feed
                frame_placeholder = st.empty()
                stop_button = st.button("⏹️ Stop Detection", type="secondary", use_container_width=True)
                
                # Statistics
                detection_history = []
                stats_placeholder = st.empty()
                
                while not stop_button and cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        st.error("Failed to grab frame")
                        break
                    
                    # Process frame
                    processed_frame, num_detections = self.process_video_frame(frame)
                    detection_history.append(num_detections)
                    
                    # Keep only last 30 values for rolling stats
                    if len(detection_history) > 30:
                        detection_history.pop(0)
                    
                    # Resize for display
                    processed_frame = self.resize_with_aspect_ratio(processed_frame)
                    
                    # Display frame
                    frame_placeholder.image(processed_frame, use_container_width=True)
                    
                    # Update statistics
                    with stats_placeholder.container():
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Current Detections", num_detections, delta=None)
                        with col2:
                            avg_detections = np.mean(detection_history) if detection_history else 0
                            st.metric("Average (last 30)", f"{avg_detections:.2f}")
                        with col3:
                            status_color = "🟢" if num_detections > 0 else "🟡"
                            st.metric("Status", f"{status_color} Active")
                    
                    # Small delay to prevent overwhelming
                    # time.sleep(0.033)
                
                cap.release()
                st.info("🛑 Live detection stopped")

if __name__ == "__main__":
    app = PotholeDetectionApp()
    app.run()