from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps, UnidentifiedImageError


# ============================================================
# KONFIGURASI APLIKASI
# ============================================================
st.set_page_config(
    page_title="Klasifikasi Emosi Wajah",
    page_icon="🙂",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"

# Urutan kelas HARUS sama dengan urutan saat model dilatih.
CLASS_NAMES = ["marah", "sedih", "senang"]
IMAGE_SIZE = (224, 224)

MODEL_FILES: Dict[str, Path] = {
    "MobileNetV2 Transfer Learning": MODEL_DIR / "mobilenetv2_model.keras",
    "CNN From Scratch": MODEL_DIR / "cnn_model.keras",
}

# Hasil evaluasi berdasarkan notebook yang diunggah.
# Ubah angka ini jika model dilatih ulang.
MODEL_METRICS = {
    "CNN From Scratch": {
        "Accuracy": 0.3043,
        "Precision Macro": 0.1886,
        "Recall Macro": 0.2917,
        "F1-Score Macro": 0.2037,
    },
    "MobileNetV2 Transfer Learning": {
        "Accuracy": 0.4348,
        "Precision Macro": 0.2949,
        "Recall Macro": 0.4464,
        "F1-Score Macro": 0.3519,
    },
}


# ============================================================
# STYLE
# ============================================================
st.markdown(
    """
    <style>
        .block-container {
            max-width: 1180px;
            padding-top: 2rem;
            padding-bottom: 3rem;
        }
        .main-title {
            font-size: 2.25rem;
            font-weight: 750;
            line-height: 1.2;
            margin-bottom: 0.2rem;
        }
        .subtitle {
            color: #6b7280;
            margin-bottom: 1.5rem;
        }
        .result-box {
            border: 1px solid rgba(128,128,128,0.25);
            border-radius: 14px;
            padding: 1.1rem 1.2rem;
            margin-top: 0.5rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# FUNGSI MODEL DAN PREPROCESSING
# ============================================================
@st.cache_resource(show_spinner="Memuat model...")
def load_model(model_path: str) -> tf.keras.Model:
    """
    Memuat model Keras tanpa optimizer/training state.
    compile=False cukup untuk kebutuhan inferensi dan lebih ringan.
    """
    return tf.keras.models.load_model(model_path, compile=False)


def prepare_image(uploaded_file) -> Tuple[Image.Image, np.ndarray]:
    """
    Membaca gambar, memperbaiki orientasi EXIF, mengubah ke RGB,
    resize 224x224, lalu membuat batch berukuran (1, 224, 224, 3).

    Normalisasi tidak dilakukan di sini karena preprocessing sudah
    tertanam di dalam model CNN dan MobileNetV2.
    """
    image = Image.open(uploaded_file)
    image = ImageOps.exif_transpose(image).convert("RGB")
    resized = image.resize(IMAGE_SIZE, Image.Resampling.LANCZOS)

    array = np.asarray(resized, dtype=np.float32)
    batch = np.expand_dims(array, axis=0)

    return image, batch


def predict(model: tf.keras.Model, image_batch: np.ndarray):
    """Menghasilkan label, confidence, dan probabilitas setiap kelas."""
    probabilities = model.predict(image_batch, verbose=0)[0]

    if probabilities.shape[0] != len(CLASS_NAMES):
        raise ValueError(
            "Jumlah output model tidak sama dengan jumlah kelas. "
            f"Output model: {probabilities.shape[0]}, "
            f"jumlah kelas: {len(CLASS_NAMES)}."
        )

    predicted_index = int(np.argmax(probabilities))
    predicted_label = CLASS_NAMES[predicted_index]
    confidence = float(probabilities[predicted_index])

    probability_df = pd.DataFrame(
        {
            "Emosi": [name.capitalize() for name in CLASS_NAMES],
            "Probabilitas": probabilities.astype(float),
        }
    ).sort_values("Probabilitas", ascending=False)

    return predicted_label, confidence, probability_df


def available_models() -> Dict[str, Path]:
    """Mengembalikan hanya model yang benar-benar ada di repository."""
    return {
        name: path
        for name, path in MODEL_FILES.items()
        if path.exists()
    }


# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.header("Pengaturan")

    available = available_models()

    if not available:
        st.error("Model belum ditemukan.")
        st.code(
            "models/\n"
            "├── mobilenetv2_model.keras\n"
            "└── cnn_model.keras"
        )
        st.stop()

    default_index = 0
    model_names = list(available.keys())

    selected_model_name = st.selectbox(
        "Pilih model",
        options=model_names,
        index=default_index,
        help="MobileNetV2 memiliki hasil test lebih tinggi pada notebook saat ini.",
    )

    st.divider()
    st.caption("Ukuran input: 224 × 224 piksel")
    st.caption("Kelas: marah, sedih, senang")
    st.caption("Aplikasi ini dibuat untuk demonstrasi proyek akademik.")


# ============================================================
# HEADER
# ============================================================
st.markdown(
    '<div class="main-title">Klasifikasi Ekspresi Emosi Wajah</div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="subtitle">'
    'Perbandingan CNN From Scratch dan MobileNetV2 Transfer Learning'
    '</div>',
    unsafe_allow_html=True,
)

st.info(
    "Unggah satu foto wajah. Sistem akan menampilkan prediksi emosi "
    "beserta probabilitas untuk kelas marah, sedih, dan senang."
)


# ============================================================
# AREA PREDIKSI
# ============================================================
left_col, right_col = st.columns([1, 1], gap="large")

with left_col:
    st.subheader("1. Unggah Gambar")

    uploaded_file = st.file_uploader(
        "Format yang didukung: JPG, JPEG, PNG, atau WEBP",
        type=["jpg", "jpeg", "png", "webp"],
        accept_multiple_files=False,
    )

    if uploaded_file is None:
        st.warning("Belum ada gambar yang diunggah.")
    else:
        try:
            original_image, image_batch = prepare_image(uploaded_file)
            st.image(
                original_image,
                caption="Gambar masukan",
                use_container_width=True,
            )
        except (UnidentifiedImageError, OSError, ValueError) as error:
            st.error(f"Gambar tidak dapat dibaca: {error}")
            st.stop()

with right_col:
    st.subheader("2. Hasil Prediksi")

    if uploaded_file is None:
        st.write("Hasil akan muncul setelah gambar diunggah.")
    else:
        try:
            model_path = available[selected_model_name]
            model = load_model(str(model_path))

            predicted_label, confidence, probability_df = predict(
                model,
                image_batch,
            )

            st.markdown('<div class="result-box">', unsafe_allow_html=True)

            metric_col1, metric_col2 = st.columns(2)

            with metric_col1:
                st.metric(
                    "Prediksi Emosi",
                    predicted_label.capitalize(),
                )

            with metric_col2:
                st.metric(
                    "Confidence",
                    f"{confidence:.2%}",
                )

            st.markdown("</div>", unsafe_allow_html=True)

            st.write("#### Probabilitas Setiap Kelas")

            chart_df = probability_df.set_index("Emosi")
            st.bar_chart(chart_df["Probabilitas"])

            table_df = probability_df.copy()
            table_df["Probabilitas"] = table_df["Probabilitas"].map(
                lambda value: f"{value:.2%}"
            )
            st.dataframe(
                table_df,
                use_container_width=True,
                hide_index=True,
            )

            if confidence < 0.60:
                st.warning(
                    "Confidence masih rendah. Prediksi ini perlu "
                    "ditafsirkan dengan hati-hati."
                )

        except Exception as error:
            st.error(
                "Prediksi gagal. Periksa kecocokan versi TensorFlow, "
                "nama model, dan struktur file."
            )
            st.exception(error)


# ============================================================
# HASIL EVALUASI NOTEBOOK
# ============================================================
st.divider()
st.subheader("Perbandingan Hasil Evaluasi")

metrics_df = pd.DataFrame(MODEL_METRICS).reset_index()
metrics_df = metrics_df.rename(columns={"index": "Metrik"})

display_metrics_df = metrics_df.copy()
for column in display_metrics_df.columns[1:]:
    display_metrics_df[column] = display_metrics_df[column].map(
        lambda value: f"{value:.2%}"
    )

st.dataframe(
    display_metrics_df,
    use_container_width=True,
    hide_index=True,
)

st.caption(
    "Nilai tersebut berasal dari data test pada notebook saat ini. "
    "Perbarui MODEL_METRICS di app.py apabila model dilatih ulang."
)

with st.expander("Cara membaca hasil"):
    st.write(
        "MobileNetV2 menghasilkan nilai test lebih tinggi daripada CNN "
        "pada seluruh metrik utama. Namun, akurasi 43,48% masih rendah "
        "untuk sistem klasifikasi tiga kelas, sehingga aplikasi ini lebih "
        "tepat diposisikan sebagai demonstrasi akademik, bukan alat "
        "penilaian emosi yang andal."
    )
