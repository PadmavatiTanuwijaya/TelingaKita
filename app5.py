# app.py
import streamlit as st
import cv2
import numpy as np
import time
import tensorflow as tf
import mediapipe as mp
import json
from tensorflow.keras.applications.mobilenet_v3 import preprocess_input
from collections import Counter
import streamlit.components.v1 as components
from streamlit_option_menu import option_menu
import base64


st.set_page_config(
    page_title="TelingaKita",
    layout="wide",
    page_icon="TelingaKita.PNG"  # Pastikan file ini ada di direktori yang sama
)

# Fungsi untuk konversi gambar ke base64
def get_base64_image(image_path):
    with open(image_path, "rb") as img_file:
        encoded = base64.b64encode(img_file.read()).decode()
    return f"data:image/PNG;base64,{encoded}"

# ------------------------- Splash Screen -------------------------
if 'splash_shown' not in st.session_state:
    st.session_state.splash_shown = True
    # Konversi gambar
    base64_img = get_base64_image("./TelingaKita.PNG")
    splash_html = f"""
    <style>
    .splash-container {{
        position: fixed;
        top: 0;
        left: 0;
        width: 100vw;
        height: 100vh;
        background-color: #00264d;
        display: flex;
        justify-content: center;
        align-items: center;
        z-index: 9999;
    }}
    </style>
    <div class="splash-container" id="splash">
        <img src="{base64_img}" alt="Logo" style="width: 120px; border-radius: 10px;" />
    </div>
    <script>
    setTimeout(function() {{
        var splash = document.getElementById('splash');
        if (splash) splash.style.display = 'none';
    }}, 1200);
    </script>
    """

    st.markdown(splash_html, unsafe_allow_html=True)
    time.sleep(0.1)
    st.rerun()

# ------------------------- Load Model & Label -------------------------
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
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# ------------------------- Sidebar Menu -------------------------
with st.sidebar:
    st.markdown(
        """
        <style>
        section[data-testid="stSidebar"] {
            background-color: #FCD953;
            border-top-right-radius: 20px;
            border-bottom-right-radius: 20px;
            padding-top: 30px;
        }
        .block-container > div > div:first-child img {
            border-radius: 15px;
            margin-bottom: 10px;
        }
        .css-1d391kg {
            padding: 0 !important;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
    st.image("TelingaKita.PNG", width=180)
    selected = option_menu(
        menu_title=None,
        options=["Deteksi Huruf", "Tentang BISINDO", "Tentang Pengembang"],
        icons=["camera-video", "book", "person-circle"],
        default_index=0,
        orientation="vertical",
        styles={
            "container": {
                "background-color": "#FCD953",
                "padding": "10px",
                "border-radius": "0px"
            },
            "icon": {
                "color": "black",
                "font-size": "22px"
            },
            "nav-link": {
                "color": "black",
                "font-size": "18px",
                "padding": "10px",
                "border-radius": "10px",
                "margin": "5px 0",
                "--hover-color": "#fff3b3"
            },
            "nav-link-selected": {
                "background-color": "#fff1aa",
                "font-weight": "bold",
                "color": "black"
            }
        }
    )

# ------------------------- Fitur Deteksi -------------------------
if selected == "Deteksi Huruf":
    st.markdown("<h1 style='text-align: center; color: #2b6cb0;'>🖐️ Deteksi Bahasa Isyarat BISINDO</h1>", unsafe_allow_html=True)
    run = st.toggle("🎥 Aktifkan Kamera", value=False)
    st.markdown("---")

    FRAME_INTERVAL = 3.0
    prev_time = time.time()
    buffer_preds = []

    if run:
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        stframe = st.empty()
        result_text = st.empty()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                st.warning("Tidak bisa mengakses kamera.")
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)
            img_h, img_w, _ = frame.shape

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
                if current_time - prev_time > FRAME_INTERVAL:
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

                        buffer_preds.extend([label1, label2, label3])

                        vote = Counter(buffer_preds).most_common(1)[0]
                        label_final = vote[0].upper()
                        count = vote[1]
                        confidence = (count / len(buffer_preds)) * 100

                        result_text.markdown(
                            f"<h2 style='text-align: center;'>✋ Huruf Terdeteksi: <span style='color: green;'>{label_final}</span> ({confidence:.1f}%)</h2>",
                            unsafe_allow_html=True
                        )

                        components.html(f"""
                            <script>
                                function speak(text) {{
                                    const synth = window.speechSynthesis;
                                    const utterance = new SpeechSynthesisUtterance(text);
                                    utterance.lang = "id-ID";
                                    utterance.rate = 1;
                                    synth.cancel();
                                    synth.speak(utterance);
                                }}
                                speak("Huruf {label_final}");
                            </script>
                        """, height=0)

                        prev_time = current_time
                        buffer_preds.clear()

                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (255, 204, 0), 2)

            stframe.image(frame, channels="BGR", use_container_width=True)
            time.sleep(0.02)

        cap.release()
        stframe.empty()
    else:
        st.info("Aktifkan kamera untuk memulai deteksi gesture.")

# ------------------------- Tentang BISINDO -------------------------
elif selected == "Tentang BISINDO":
    st.subheader("📘 Tentang BISINDO")
    st.markdown("""
**Bahasa Isyarat Indonesia (BISINDO)** adalah bahasa isyarat yang digunakan oleh komunitas Tuli di berbagai daerah di Indonesia. BISINDO merupakan bentuk komunikasi visual yang menggunakan gerakan tangan, ekspresi wajah, dan posisi tubuh untuk menyampaikan makna.

### 🕰️ Sejarah Singkat BISINDO
Bahasa isyarat telah digunakan oleh komunitas Tuli di Indonesia sejak lama, jauh sebelum ada pengakuan resmi dari pemerintah. Namun, istilah **"BISINDO"** sendiri mulai dikenal secara luas pada awal tahun 2000-an. Bahasa ini berkembang secara alami dalam komunitas Tuli melalui interaksi sehari-hari di berbagai kota seperti Jakarta, Bandung, Yogyakarta, dan kota-kota lainnya.

Sebelum munculnya BISINDO sebagai istilah populer, komunitas Tuli di Indonesia telah lama menggunakan bahasa isyarat lokal yang berbeda-beda. Sayangnya, pada masa lalu, pendekatan pendidikan formal cenderung menggunakan metode oral atau bahasa isyarat buatan seperti SIBI (Sistem Isyarat Bahasa Indonesia), yang tidak berkembang secara alami di komunitas.

Seiring meningkatnya kesadaran akan hak-hak penyandang disabilitas, komunitas Tuli mulai memperjuangkan pengakuan terhadap bahasa isyarat yang benar-benar mencerminkan budaya dan cara berkomunikasi mereka. BISINDO lahir sebagai hasil dari perjuangan tersebut, dan hingga kini menjadi simbol penting bagi identitas, budaya, dan hak berbahasa komunitas Tuli di Indonesia.

---
📸 **Catatan Dataset**  
Data gambar isyarat tangan yang digunakan dalam pelatihan model ini berasal dari sesi pembelajaran langsung bersama **Ce Lucia**, Beliau adalah *CEO & Founder IBIC* (Insert BISINDO into Conversations).
""")

    st.image("Bersama.PNG", caption="Sesi pembelajaran BISINDO bersama Ce Lucia", use_container_width=True)

    st.markdown("""
🙏 Terima kasih atas kontribusi IBIC dalam menyediakan sumber pembelajaran yang bermakna dan autentik bagi komunitas serta pengembang teknologi.
""")


# ------------------------- Tentang Pengembang -------------------------
elif selected == "Tentang Pengembang":
    st.subheader("👨‍💻 Tentang Pengembang")
    st.markdown("""
Halo! Saya adalah seorang mahasiswa yang mengembangkan aplikasi ini sebagai bagian dari proyek akhir dan kontribusi nyata untuk mendukung inklusivitas bagi penyandang disabilitas, khususnya komunitas Tuli pengguna BISINDO (Bahasa Isyarat Indonesia).

### 🔍 Teknologi di Balik Aplikasi
Aplikasi ini menggabungkan beberapa teknologi modern dalam pengolahan citra dan machine learning:

- 📱 **MobileNetV3-Small** Model utama yang digunakan dalam sistem ini adalah **MobileNetV3-Small**, yaitu model convolutional neural network (CNN) yang dirancang secara khusus agar ringan dan efisien. MobileNetV3 memanfaatkan dua teknik utama:
    > **Depthwise Separable Convolution**, yang membagi proses konvolusi menjadi dua tahap (depthwise & pointwise) untuk mengurangi jumlah parameter.
    > **Squeeze-and-Excitation (SE) Module**, yang menambahkan perhatian (attention) ke saluran-saluran penting dalam jaringan.
    > **Hard-Swish Activation**, yang lebih efisien dibanding fungsi aktivasi ReLU atau Swish biasa.
- ✋ **MediaPipe** digunakan untuk mendeteksi dan mengekstraksi koordinat tangan secara real-time. Framework ini memudahkan pemrosesan awal dengan menghasilkan keypoints (titik-titik penting) dari tangan.
- 🌳 **Decision Tree sebagai Post-Processor**  
  Di tahap akhir, setelah MobileNetV3 melakukan prediksi pada frame-frame video secara berurutan, kita mengimplementasikan **Decision Tree** sebagai algoritma pemungutan keputusan berdasarkan **kemunculan prediksi terbanyak (mode class)**.  
  Secara teoritis, ini disebut teknik **majority voting** atau **mode estimation**, di mana:
  > Decision Tree di sini tidak digunakan untuk pelatihan dari awal, melainkan sebagai alat logika untuk menentukan hasil akhir prediksi dari sekumpulan output model.  
- 🖼️ **Augmentasi Gambar** seperti rotasi, translasi, noise, dan pencahayaan digunakan dalam pelatihan model untuk meningkatkan akurasi dan generalisasi terhadap berbagai kondisi pencahayaan dan latar belakang.

### 🙋‍♂️ Tentang Saya
📧 Email: Padmavati.tanuwijaya@student.ukdc.ac.id  
🏫 Institusi: Universitas Katolik Darma Cendika

Terima kasih telah menggunakan aplikasi ini. Semoga teknologi ini bisa menjadi jembatan komunikasi yang lebih baik bagi semua pihak!
    """)
