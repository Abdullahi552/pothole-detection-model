import streamlit as st
import os
import sys

# MUST be the first Streamlit command
st.set_page_config(
    page_title="Pothole Detection System",
    page_icon="🕳️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Now import other libraries
import cv2
from PIL import Image
import numpy as np
import tempfile
from ultralytics import YOLO

# Custom CSS
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

class PotholeDetectionApp:
    def __init__(self):
        # Load YOLOv8 model with caching
        @st.cache_resource
        def load_model():
            model_path = 'poth/best.pt'
            
            if not os.path.exists(model_path):
                st.error(f"""
                ❌ Model not found at `{model_path}`!
                
                Please ensure your model file is in the poth/ directory.
                """)
                return None
            
            try:
                # Load model
                model = YOLO(model_path)
                return model
            except Exception as e:
                st.error(f"Error loading model: {str(e)}")
                return None
        
        self.model = load_model()
    
    def resize_with_aspect_ratio(self, image, max_width=800, max_height=600):
        """Resize image while maintaining aspect ratio"""
        try:
            h, w = image.shape[:2]
            scale = min(max_width / w, max_height / h)
            
            if scale < 1:
                new_w = int(w * scale)
                new_h = int(h * scale)
                return cv2.resize(image, (new_w, new_h))
            return image
        except Exception:
            return image
    
    def process_image(self, image_file):
        """Process a single image"""
        try:
            # Read image
            image = Image.open(image_file)
            img_array = np.array(image)
            
            # Convert RGB to BGR for OpenCV if needed
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
        # Header
        st.markdown('<div class="main-header">', unsafe_allow_html=True)
        st.title("🕳️ Pothole Detection System")
        st.markdown("### Real-time Pothole Detection using YOLOv8")
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Check if model loaded successfully
        if self.model is None:
            st.error("""
            ### ❌ Model Failed to Load
            
            **Please check:**
            1. Make sure `poth/best.pt` exists in your repository
            2. The file is properly committed to GitHub
            3. The file isn't corrupted
            """)
            
            # Show file structure
            st.subheader("📁 Current File Structure:")
            col1, col2 = st.columns(2)
            
            with col1:
                st.code("""
Repository contents:
/
├── pothole.py
├── requirements.txt  
├── packages.txt
└── poth/
    └── best.pt
                """)
            
            with col2:
                if os.path.exists('poth'):
                    st.success("✅ poth/ directory exists")
                    if os.path.exists('poth/best.pt'):
                        file_size = os.path.getsize('poth/best.pt') / (1024*1024)
                        st.success(f"✅ best.pt found ({file_size:.2f} MB)")
                    else:
                        st.error("❌ best.pt not found in poth/ directory")
                else:
                    st.error("❌ poth/ directory not found")
            
            st.stop()
        
        # Sidebar for controls
        with st.sidebar:
            st.header("🎮 Controls")
            
            # Input selection
            input_type = st.radio(
                "Select Input Type",
                ["📷 Image", "🎥 Video"],
                index=0
            )
            
            st.markdown("---")
            
            # Model info
            st.header("ℹ️ Model Information")
            st.success("✅ Model loaded successfully!")
            
            # Detection settings
            st.header("⚙️ Detection Settings")
            confidence_threshold = st.slider(
                "Confidence Threshold",
                min_value=0.0,
                max_value=1.0,
                value=0.25,
                step=0.05,
                help="Lower values detect more objects but may increase false positives"
            )
            
            # Update model confidence threshold
            if self.model:
                self.model.conf = confidence_threshold
            
            st.markdown("---")
            st.caption("Developed with ❤️ using Streamlit & YOLOv8")
        
        # Main content area
        if "📷 Image" in input_type:
            self.image_detection_tab()
        else:
            self.video_detection_tab()
    
    def image_detection_tab(self):
        st.header("📷 Image Detection")
        st.markdown("Upload an image to detect potholes")
        
        uploaded_file = st.file_uploader(
            "Choose an image...",
            type=['jpg', 'jpeg', 'png', 'bmp'],
            help="Supported formats: JPG, JPEG, PNG, BMP"
        )
        
        if uploaded_file is not None:
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Original Image")
                original_image = Image.open(uploaded_file)
                st.image(original_image, use_column_width=True)
            
            if st.button("🔍 Detect Potholes", type="primary", use_container_width=True):
                with st.spinner("Processing image..."):
                    uploaded_file.seek(0)
                    original, processed, num_detections = self.process_image(uploaded_file)
                    
                    if processed is not None:
                        with col2:
                            st.subheader("Detection Result")
                            st.image(processed, use_column_width=True)
                        
                        # Display detection info
                        st.markdown('<div class="detection-info">', unsafe_allow_html=True)
                        if num_detections > 0:
                            st.success(f"✅ Detected {num_detections} pothole(s) in the image!")
                        else:
                            st.warning("⚠️ No potholes detected. Try lowering the confidence threshold.")
                        st.metric("Potholes Detected", num_detections)
                        st.markdown('</div>', unsafe_allow_html=True)
    
    def video_detection_tab(self):
        st.header("🎥 Video Detection")
        st.markdown("Upload a video file for batch pothole detection")
        
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
            
            # Get video info
            cap = cv2.VideoCapture(video_path)
            if cap.isOpened():
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                fps = cap.get(cv2.CAP_PROP_FPS)
                cap.release()
                st.info(f"📹 Video Info: {total_frames} frames, {fps:.2f} FPS")
            else:
                st.error("Failed to read video file")
                return
            
            if st.button("▶️ Start Video Detection", type="primary", use_container_width=True):
                # Create progress indicators
                progress_bar = st.progress(0)
                status_text = st.empty()
                frame_placeholder = st.empty()
                
                cap = cv2.VideoCapture(video_path)
                frame_count = 0
                detection_counts = []
                
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    # Process frame
                    processed_frame, num_detections = self.process_video_frame(frame)
                    
                    # Resize for display
                    processed_frame = self.resize_with_aspect_ratio(processed_frame)
                    
                    # Display frame
                    frame_placeholder.image(processed_frame, use_column_width=True)
                    
                    # Update progress
                    frame_count += 1
                    detection_counts.append(num_detections)
                    progress = frame_count / total_frames
                    progress_bar.progress(progress)
                    status_text.text(f"Processing frame {frame_count}/{total_frames} - Detections: {num_detections}")
                
                cap.release()
                progress_bar.empty()
                
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

if __name__ == "__main__":
    app = PotholeDetectionApp()
    app.run()
