# app_webrtc.py
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import tensorflow as tf
import numpy as np
import mediapipe as mp
import json
import cv2
from tensorflow.keras.applications.mobilenet_v3 import preprocess_input
from collections import Counter
import time

# ------------------------- Load Label & Model -------------------------
with open("label_map.json", "r") as f:
    label_map = json.load(f)
label_list = [label_map[str(i)] for i in range(len(label_map))]

@st.cache_resource
def load_models():
    model1 = tf.keras.models.load_model("sgd.keras")
    model2 = tf.keras.models.load_model("sgd2.keras")
    model3 = tf.keras.models.load_model("sgd3.keras")
    return model1, model2, model3

model1, model2, model3 = load_models()
IMG_SIZE = model1.input_shape[1]

# ------------------------- MediaPipe Setup -------------------------
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

class HandGestureTransformer(VideoTransformerBase):
    def __init__(self):
        self.hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)
        self.prev_time = time.time()
        self.buffer_preds = []
        self.FRAME_INTERVAL = 3.0  # Deteksi setiap 3 detik
        self.label_final = ""
        self.confidence = 0

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb)
        img_h, img_w, _ = img.shape

        if results.multi_hand_landmarks:
            all_x, all_y = [], []
            for hand_landmarks in results.multi_hand_landmarks:
                all_x.extend([lm.x for lm in hand_landmarks.landmark])
                all_y.extend([lm.y for lm in hand_landmarks.landmark])

            x_min = max(int(min(all_x) * img_w) - 20, 0)
            x_max = min(int(max(all_x) * img_w) + 20, img_w)
            y_min = max(int(min(all_y) * img_h) - 20, 0)
            y_max = min(int(max(all_y) * img_h) + 20, img_h)

            hand_crop = rgb[y_min:y_max, x_min:x_max]
            current_time = time.time()

            if current_time - self.prev_time > self.FRAME_INTERVAL:
                if hand_crop.size > 0:
                    resized = cv2.resize(hand_crop, (IMG_SIZE, IMG_SIZE))
                    input_array = preprocess_input(resized.astype("float32"))
                    input_array = np.expand_dims(input_array, axis=0)

                    preds1 = model1.predict(input_array, verbose=0)
                    preds2 = model2.predict(input_array, verbose=0)
                    preds3 = model3.predict(input_array, verbose=0)

                    label1 = label_list[np.argmax(preds1)]
                    label2 = label_list[np.argmax(preds2)]
                    label3 = label_list[np.argmax(preds3)]

                    self.buffer_preds.extend([label1, label2, label3])

                    vote = Counter(self.buffer_preds).most_common(1)[0]
                    self.label_final = vote[0].upper()
                    count = vote[1]
                    self.confidence = (count / len(self.buffer_preds)) * 100

                    self.buffer_preds.clear()
                    self.prev_time = current_time

            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (255, 204, 0), 2)

        cv2.putText(img, f"Prediksi: {self.label_final} ({self.confidence:.1f}%)", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)

        return img

# ------------------------- Streamlit App -------------------------
st.set_page_config(page_title="TelingaKita - Real Time", layout="centered")
st.title("📹 Deteksi Real-Time BISINDO dengan Kamera")
st.info("Izinkan akses kamera untuk mulai mendeteksi gerakan tangan.")

webrtc_streamer(key="realtime-bisindo", video_transformer_factory=HandGestureTransformer)

st.markdown("""
---
👨‍💻 Aplikasi ini menggunakan *streamlit-webrtc* untuk memungkinkan deteksi gerakan tangan secara real-time melalui browser.
""")
