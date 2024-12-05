import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import tempfile
import cv2
import time  # Untuk mengukur waktu
from sklearn.metrics import average_precision_score

# Load model YOLOv8
model_path = "bestModelYolo.pt"  # Ganti dengan path model Anda
model = YOLO(model_path)

# Fungsi untuk deteksi pada gambar
def detect_anthracnose(image):
    start_time = time.time()  # Mulai pengukuran waktu
    results = model(image)
    annotated_frame = results[0].plot()  # Visualisasi hasil
    # Ekstraksi prediksi
    detections = []
    for r in results[0].boxes:
        class_id = int(r.cls)
        confidence = float(r.conf)
        detections.append((model.names[class_id], confidence))
    end_time = time.time()  # Selesai pengukuran waktu
    detection_time = end_time - start_time
    return annotated_frame, detections, detection_time

# Fungsi untuk deteksi pada video
def detect_video(video_path):
    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    # File sementara untuk menyimpan hasil video
    temp_output = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    out = cv2.VideoWriter(temp_output.name, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    
    start_time = time.time()  # Mulai pengukuran waktu
    all_detections = []  # Untuk menyimpan semua deteksi
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        results = model(frame)
        annotated_frame = results[0].plot()
        out.write(annotated_frame)
        
        # Ekstraksi prediksi untuk mAP
        for r in results[0].boxes:
            class_id = int(r.cls)
            confidence = float(r.conf)
            all_detections.append((model.names[class_id], confidence))
    
    end_time = time.time()  # Selesai pengukuran waktu
    cap.release()
    out.release()
    detection_time = end_time - start_time
    return temp_output.name, detection_time, all_detections

# Fungsi untuk menghitung mAP
def calculate_map(true_labels, pred_labels):
    # Menghitung mAP
    y_true = np.zeros((len(true_labels), len(model.names)))
    y_pred = np.zeros((len(pred_labels), len(model.names)))

    for i, label in enumerate(true_labels):
        y_true[i][label] = 1  # Set true label

    for i, (label, confidence) in enumerate(pred_labels):
        y_pred[i][label] = confidence  # Set predicted confidence

    # Menghitung mAP
    mAP = average_precision_score(y_true, y_pred, average='macro')
    return mAP

# Streamlit UI
st.title("Deteksi Penyakit Antraknosa pada Buah Pisang Algoritma Yolo")
st.sidebar.header("Pilihan Deteksi")
mode = st.sidebar.selectbox("Pilih Mode:", ["Gambar", "Video"])

if mode == "Gambar":
    uploaded_image = st.file_uploader("Unggah Gambar", type=["jpg", "jpeg", "png"])
    if uploaded_image:
        image = Image.open(uploaded_image)
        image_np = np.array(image)
        st.image(image, caption="Gambar yang Diupload", use_container_width=True)  # Update parameter
        
        st.write("Proses deteksi...")
        annotated_image, detections, detection_time = detect_anthracnose(image_np)
        st.image(annotated_image, caption="Hasil Deteksi", use_container_width=True)  # Update parameter
        
        # Tampilkan hasil prediksi
        st.write("Deteksi:")
        for label, confidence in detections:
            st.write(f"- **{label}** dengan kepercayaan {confidence:.2f}")
        
        # Tampilkan waktu deteksi
        st.write(f"**Waktu Deteksi:** {detection_time:.2f} detik")

elif mode == "Video":
    uploaded_video = st.file_uploader("Unggah Video", type=["mp4", "mov", "avi", "mkv"])
    if uploaded_video:
        temp_input = tempfile.NamedTemporaryFile(delete=False)
        temp_input.write(uploaded_video.read())
        
        st.video(temp_input.name)
        
        st.write("Proses deteksi...")
        output_video_path, detection_time, all_detections = detect_video(temp_input.name)
        st.video(output_video_path)
        
        # Tampilkan waktu deteksi
        st.write(f"**Waktu Deteksi:** {detection_time:.2f} detik")

        # Contoh label yang benar (harus disesuaikan dengan data Anda)
        true_labels = [0, 1]  # Misalnya, 0 untuk 'antraknosa', 1 untuk 'normal'
        mAP = calculate_map(true_labels, all_detections)
        st.write(f"**mAP:** {mAP:.2f}")

st.write("Aplikasi deteksi penyakit antraknosa pada buah pisang dengan YOLOv8.")
