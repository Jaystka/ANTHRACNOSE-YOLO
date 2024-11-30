import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import tempfile
import cv2

# Load model YOLOv8
model_path = "modelYolo.pt"  # Ganti dengan path model And
model = YOLO(model_path)

# Fungsi untuk deteksi pada gambar
def detect_anthracnose(image):
    results = model(image)
    annotated_frame = results[0].plot()  # Visualisasi hasil
    # Ekstraksi prediksi
    detections = []
    for r in results[0].boxes:
        class_id = int(r.cls)
        confidence = float(r.conf)
        detections.append((model.names[class_id], confidence))
    return annotated_frame, detections

# Fungsi untuk deteksi pada video
def detect_video(video_path):
    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    # File sementara untuk menyimpan hasil video
    temp_output = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    out = cv2.VideoWriter(temp_output.name, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        results = model(frame)
        annotated_frame = results[0].plot()
        out.write(annotated_frame)
    
    cap.release()
    out.release()
    return temp_output.name

# Streamlit UI
st.title("Deteksi Penyakit Antraknosa pada Buah Pisang")
st.sidebar.header("Pilihan Deteksi")
mode = st.sidebar.selectbox("Pilih Mode:", ["Gambar", "Video"])

if mode == "Gambar":
    uploaded_image = st.file_uploader("Unggah Gambar", type=["jpg", "jpeg", "png"])
    if uploaded_image:
        image = Image.open(uploaded_image)
        image_np = np.array(image)
        st.image(image, caption="Gambar yang Diupload", use_column_width=True)
        
        st.write("Proses deteksi...")
        annotated_image, detections = detect_anthracnose(image_np)
        st.image(annotated_image, caption="Hasil Deteksi", use_column_width=True)
        
        # Tampilkan hasil prediksi
        st.write("Deteksi:")
        for label, confidence in detections:
            st.write(f"- **{label}** dengan kepercayaan {confidence:.2f}")

elif mode == "Video":
    uploaded_video = st.file_uploader("Unggah Video", type=["mp4", "mov", "avi", "mkv"])
    if uploaded_video:
        temp_input = tempfile.NamedTemporaryFile(delete=False)
        temp_input.write(uploaded_video.read())
        
        st.video(temp_input.name)
        
        st.write("Proses deteksi...")
        output_video_path = detect_video(temp_input.name)
        st.video(output_video_path)

st.write("Aplikasi deteksi penyakit antraknosa pada buah pisang dengan YOLOv8.")