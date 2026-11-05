
# ============================================================
# FACIAL EMOTION RECOGNITION
# CNN MODEL WITH STREAMLIT DASHBOARD
# ============================================================

import os
import json
import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf

import matplotlib.pyplot as plt
import seaborn as sns

from PIL import Image
from pathlib import Path


# ============================================================
# 1. KONFIGURASI APLIKASI
# ============================================================

st.set_page_config(
    page_title="Facial Emotion Recognition",
    page_icon="😊",
    layout="wide"
)

BASE_DIR = Path(__file__).resolve().parent

MODEL_DIR = BASE_DIR / "models"
OUTPUT_DIR = BASE_DIR / "outputs"

MODEL_PATH = MODEL_DIR / "best_emotion_cnn.keras"
CLASS_PATH = MODEL_DIR / "class_names.json"
CONFIG_PATH = MODEL_DIR / "model_config.json"

DISTRIBUTION_PATH = OUTPUT_DIR / "dataset_distribution.csv"
EVALUATION_PATH = OUTPUT_DIR / "evaluation_results.csv"
REPORT_PATH = OUTPUT_DIR / "classification_report.csv"
HISTORY_PATH = OUTPUT_DIR / "training_history.csv"

CONFUSION_PATH = OUTPUT_DIR / "confusion_matrix.png"

IMG_SIZE = (96, 96)

EMOTION_EMOJI = {
    "angry": "😠",
    "disgust": "🤢",
    "fear": "😨",
    "happy": "😊",
    "neutral": "😐",
    "sad": "😢",
    "surprise": "😲"
}


# ============================================================
# 2. CUSTOM STYLE
# ============================================================

st.markdown("""
<style>

.main {
    background-color: #f5f7fb;
}

.hero {
    padding: 28px;
    border-radius: 16px;
    background: linear-gradient(
        135deg,
        #172554,
        #4338ca,
        #7c3aed
    );
    color: white;
    text-align: center;
    margin-bottom: 25px;
}

.hero h1 {
    color: white;
    font-size: 36px;
}

.hero p {
    color: #e0e7ff;
    font-size: 16px;
}

.metric-card {
    background-color: white;
    padding: 18px;
    border-radius: 12px;
    border: 1px solid #e2e8f0;
}

.result-card {
    padding: 25px;
    border-radius: 15px;
    background: linear-gradient(
        135deg,
        #4338ca,
        #7c3aed
    );
    color: white;
    text-align: center;
}

.result-card h1 {
    color: white;
    font-size: 38px;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# 3. LOAD MODEL DAN ASET
# ============================================================

@st.cache_resource
def load_assets():

    model = tf.keras.models.load_model(
        MODEL_PATH,
        compile=False
    )

    with open(CLASS_PATH, "r", encoding="utf-8") as f:
        class_names = json.load(f)

    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        config = json.load(f)

    return model, class_names, config


# Memeriksa ketersediaan file
required_files = [
    MODEL_PATH,
    CLASS_PATH,
    CONFIG_PATH
]

missing_files = [
    str(path)
    for path in required_files
    if not path.exists()
]

if missing_files:

    st.error("Beberapa file model tidak ditemukan.")

    for file in missing_files:
        st.write(file)

    st.stop()


model, class_names, config = load_assets()

IMG_SIZE = tuple(config["image_size"])


# ============================================================
# 4. LOAD DATA HASIL EKSPERIMEN
# ============================================================

def load_csv(path):

    if path.exists():
        return pd.read_csv(path)

    return pd.DataFrame()


df_distribution = load_csv(DISTRIBUTION_PATH)
df_evaluation = load_csv(EVALUATION_PATH)
if REPORT_PATH.exists():
    df_report = pd.read_csv(
        REPORT_PATH,
        index_col=0
    )
else:
    df_report = pd.DataFrame()
df_history = load_csv(HISTORY_PATH)


# ============================================================
# 5. HEADER DASHBOARD
# ============================================================

# HEADER DASHBOARD

st.markdown("""
<style>
.hero-container {
    background: linear-gradient(135deg, #172554, #4338ca, #7c3aed);
    padding: 35px;
    border-radius: 16px;
    text-align: center;
    color: white;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero-container">
    <h1 style="color:white;">
        😊 Facial Emotion Recognition
    </h1>
    <p>
        Klasifikasi Emosi Wajah Menggunakan
        Convolutional Neural Network (CNN)
    </p>
    <p>
        Dashboard Analisis Big Data dan Deep Learning
    </p>
</div>
""", unsafe_allow_html=True)

# ============================================================
# 6. SIDEBAR
# ============================================================

with st.sidebar:

    st.title("📊 Informasi Model")

    st.write("**Arsitektur:** CNN")
    st.write(f"**Input:** {IMG_SIZE[0]} × {IMG_SIZE[1]}")
    st.write(f"**Jumlah kelas:** {len(class_names)}")

    st.divider()

    st.subheader("Kelas Emosi")

    for emotion in class_names:

        emoji = EMOTION_EMOJI.get(emotion, "😊")

        st.write(
            f"{emoji} {emotion.capitalize()}"
        )

    st.divider()

    st.caption(
        "Facial Emotion Recognition menggunakan TensorFlow dan Streamlit."
    )


# ============================================================
# 7. RINGKASAN DATASET
# ============================================================

st.header("📁 Ringkasan Dataset")

total_images = 0

if not df_distribution.empty:

    total_images = int(
        df_distribution["Jumlah Gambar"].sum()
    )

col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.metric(
        "Total Gambar",
        f"{total_images:,}"
    )

with col2:
    st.metric(
        "Jumlah Kelas",
        len(class_names)
    )

with col3:
    st.metric(
        "Training",
        "80%"
    )

with col4:
    st.metric(
        "Validation",
        "10%"
    )

with col5:
    st.metric(
        "Testing",
        "10%"
    )

# ============================================================
# 8. DISTRIBUSI DATASET
# ============================================================

st.subheader("Distribusi Jumlah Gambar per Kelas")

if not df_distribution.empty:

    fig, ax = plt.subplots(figsize=(11, 5))

    sns.barplot(
        data=df_distribution,
        x="Kelas Emosi",
        y="Jumlah Gambar",
        hue="Kelas Emosi",
        palette="viridis",
        legend=False,
        ax=ax
    )

    ax.set_title("Distribusi Dataset Facial Emotion")
    ax.set_xlabel("Kelas Emosi")
    ax.set_ylabel("Jumlah Gambar")

    plt.xticks(rotation=30)
    plt.tight_layout()

    st.pyplot(fig)
    plt.close(fig)

    st.dataframe(
        df_distribution,
        use_container_width=True,
        hide_index=True
    )

else:

    st.warning("Data distribusi dataset belum ditemukan.")


# ============================================================
# 9. HASIL EVALUASI MODEL
# ============================================================

st.header("📈 Evaluasi Performa Model CNN")

if not df_evaluation.empty:

    metrics = {
        row["Metrik"]: float(row["Nilai"])
        for _, row in df_evaluation.iterrows()
    }

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.metric(
            "Accuracy",
            f"{metrics.get('Accuracy', 0) * 100:.2f}%"
        )

    with c2:
        st.metric(
            "Precision",
            f"{metrics.get('Precision', 0) * 100:.2f}%"
        )

    with c3:
        st.metric(
            "Recall",
            f"{metrics.get('Recall', 0) * 100:.2f}%"
        )

    with c4:
        st.metric(
            "F1-Score",
            f"{metrics.get('F1-Score', 0) * 100:.2f}%"
        )

    st.write(
        f"**Test Loss:** {metrics.get('Test Loss', 0):.4f}"
    )

else:

    st.warning("File evaluasi belum ditemukan.")


# ============================================================
# 10. GRAFIK TRAINING HISTORY
# ============================================================

st.header("📉 Grafik Training dan Validation")

if not df_history.empty:

    if "epoch" in df_history.columns:

        chart_data = df_history.set_index("epoch")

        if all(
            col in chart_data.columns
            for col in ["accuracy", "val_accuracy"]
        ):

            st.subheader("Accuracy")

            st.line_chart(
                chart_data[
                    ["accuracy", "val_accuracy"]
                ]
            )

        if all(
            col in chart_data.columns
            for col in ["loss", "val_loss"]
        ):

            st.subheader("Loss")

            st.line_chart(
                chart_data[
                    ["loss", "val_loss"]
                ]
            )

else:

    st.warning("Training history belum tersedia.")


# ============================================================
# 11. CLASSIFICATION REPORT
# ============================================================

st.header("📋 Classification Report")

if not df_report.empty:

    st.dataframe(
        df_report,
        use_container_width=True
    )

else:

    st.warning("Classification report belum ditemukan.")


# ============================================================
# 12. CONFUSION MATRIX
# ============================================================

st.header("🔍 Confusion Matrix")

if CONFUSION_PATH.exists():

    st.image(
        str(CONFUSION_PATH),
        caption="Confusion Matrix Model CNN",
        use_container_width=True
    )

else:

    st.warning("Confusion Matrix belum tersedia.")



# ============================================================
# 13. PREDIKSI GAMBAR WAJAH
# ============================================================

st.header("😊 Prediksi Emosi Wajah")

st.write(
    "Upload gambar wajah untuk mengetahui emosi "
    "yang diprediksi oleh model CNN."
)

uploaded_file = st.file_uploader(
    "Pilih gambar wajah",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    try:

        image = Image.open(uploaded_file).convert("RGB")

        col1, col2 = st.columns(2, gap="large")

        # PREPROCESSING GAMBAR
        resized_image = image.resize(IMG_SIZE)

        image_array = np.array(
            resized_image,
            dtype=np.float32
        )

        image_array = image_array / 255.0

        image_array = np.expand_dims(
            image_array,
            axis=0
        )

        # PREDIKSI MODEL
        predictions = model.predict(
            image_array,
            verbose=0
        )[0]

        predicted_index = int(np.argmax(predictions))

        predicted_emotion = class_names[predicted_index]

        confidence = float(
            predictions[predicted_index]
        ) * 100

        emoji = EMOTION_EMOJI.get(
            predicted_emotion,
            "😊"
        )

        # KOLOM KIRI: GAMBAR INPUT
        with col1:

            st.subheader("Gambar Input")

            st.image(
                image,
                use_container_width=True
            )

        # KOLOM KANAN: HASIL PREDIKSI
        with col2:

            st.subheader("Hasil Prediksi")

            st.markdown("""
            <style>
            .prediction-card {
                background: linear-gradient(
                    135deg, #4338ca, #7c3aed
                );
                padding: 30px;
                border-radius: 15px;
                text-align: center;
                color: white;
            }

            .prediction-card h1,
            .prediction-card h2,
            .prediction-card p {
                color: white;
            }
            </style>
            """, unsafe_allow_html=True)

            st.markdown(
                f"""
                <div class="prediction-card">
                    <h1>{emoji}</h1>
                    <h2>{predicted_emotion.upper()}</h2>
                    <p>Confidence Score</p>
                    <h1>{confidence:.2f}%</h1>
                </div>
                """,
                unsafe_allow_html=True
            )

            if confidence < 50:

                st.warning(
                    "Confidence relatif rendah. "
                    "Hasil prediksi perlu ditafsirkan "
                    "dengan hati-hati."
                )

            else:

                st.success(
                    "Model berhasil menghasilkan prediksi."
                )

        # PROBABILITAS SETIAP KELAS
        st.divider()

        st.subheader("Probabilitas Setiap Kelas")

        probability_df = pd.DataFrame({
            "Emosi": class_names,
            "Probabilitas": predictions * 100
        })

        probability_df = probability_df.sort_values(
            "Probabilitas",
            ascending=False
        )

        for _, row in probability_df.iterrows():

            emotion = row["Emosi"]

            probability = float(
                row["Probabilitas"]
            )

            st.write(
                f"{EMOTION_EMOJI.get(emotion, '😊')} "
                f"**{emotion.capitalize()}** "
                f"({probability:.2f}%)"
            )

            st.progress(
                float(probability / 100)
            )

    except Exception as e:

        st.error(
            f"Terjadi kesalahan saat memproses gambar: {e}"
        )

# ============================================================
# 14. FOOTER
# ============================================================

st.divider()

st.caption(
    "Facial Emotion Recognition | "
    "Convolutional Neural Network | "
    "TensorFlow & Streamlit"
)

st.caption(
    "Aplikasi ini dibuat untuk keperluan Praktikum Analisis Big Data."
)