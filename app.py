
import streamlit as st
import pandas as pd
import joblib
import os
import re
import matplotlib.pyplot as plt
import seaborn as sns

# ==========================================
# 1. KONFIGURASI HALAMAN
# ==========================================

st.set_page_config(
    page_title="Dashboard Analisis Sentimen",
    page_icon="💬",
    layout="wide"
)

st.title("💬 Dashboard Analisis Sentimen")
st.write(
    "Klasifikasi sentimen ulasan menggunakan "
    "metode Neural Network dengan Word + Character TF-IDF."
)

# ==========================================
# 2. LOKASI FILE
# ==========================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(
    BASE_DIR, "model", "model_neural_network_word_char.pkl"
)

DATA_PATH = os.path.join(
    BASE_DIR, "dataset_sentimen.csv"
)

EVALUASI_PATH = os.path.join(
    BASE_DIR, "output", "evaluasi_nn_word_char.csv"
)

CM_PATH = os.path.join(
    BASE_DIR, "output", "confusion_matrix_nn_word_char.png"
)

# ==========================================
# 3. MEMUAT MODEL DAN DATASET
# ==========================================

@st.cache_resource
def load_model_bundle():
    return joblib.load(MODEL_PATH)

@st.cache_data
def load_dataset():
    return pd.read_csv(DATA_PATH)

if not os.path.exists(MODEL_PATH):
    st.error("File model tidak ditemukan. Periksa folder model.")
    st.stop()

if not os.path.exists(DATA_PATH):
    st.error("Dataset tidak ditemukan.")
    st.stop()

# Model terbaru berbentuk dictionary
model_data = load_model_bundle()

model = model_data["model"]
vectorizer = model_data["vectorizer"]

df = load_dataset()

# ==========================================
# 4. PREPROCESSING TEKS
# ==========================================

def bersihkan_teks(teks):
    teks = str(teks).lower()
    teks = re.sub(r"http\S+|www\S+", " ", teks)
    teks = re.sub(r"@\w+", " ", teks)
    teks = re.sub(r"#", " ", teks)
    teks = re.sub(r"\d+", " ", teks)
    teks = re.sub(r"[^a-zA-Z\s]", " ", teks)
    teks = re.sub(r"\s+", " ", teks)
    return teks.strip()

# ==========================================
# 5. SIDEBAR MENU
# ==========================================

st.sidebar.title("📌 Menu Dashboard")

menu = st.sidebar.radio(
    "Pilih Halaman:",
    [
        "Dashboard",
        "Prediksi Sentimen",
        "Evaluasi Model",
        "Dataset"
    ]
)

# ==========================================
# 6. DASHBOARD
# ==========================================

if menu == "Dashboard":

    st.header("📊 Ringkasan Dataset")

    total_data = len(df)
    distribusi = df["sentimen"].value_counts()

    negatif = distribusi.get("negatif", 0)
    netral = distribusi.get("netral", 0)
    positif = distribusi.get("positif", 0)

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Total Data", total_data)
    col2.metric("Negatif", negatif)
    col3.metric("Netral", netral)
    col4.metric("Positif", positif)

    st.divider()

    st.subheader("Distribusi Sentimen")

    fig, ax = plt.subplots(figsize=(8, 4))

    urutan = ["negatif", "netral", "positif"]
    nilai = [distribusi.get(x, 0) for x in urutan]

    sns.barplot(
        x=urutan,
        y=nilai,
        ax=ax
    )

    ax.set_xlabel("Kategori Sentimen")
    ax.set_ylabel("Jumlah Data")
    ax.set_title("Distribusi Data Sentimen")

    st.pyplot(fig)
    plt.close(fig)

    st.subheader("Informasi Model")

    col1, col2 = st.columns(2)

    col1.metric("Akurasi Model", "74,44%")
    col2.metric("Macro F1-Score", "0,6676")

    st.info(
        "Model menggunakan Neural Network (MLPClassifier) "
        "dengan Word TF-IDF, Character TF-IDF, "
        "dan Random Oversampling."
    )

# ==========================================
# 7. PREDIKSI SENTIMEN
# ==========================================

elif menu == "Prediksi Sentimen":

    st.header("🔍 Prediksi Sentimen Ulasan")

    st.write(
        "Masukkan teks ulasan untuk mengetahui "
        "kategori sentimennya."
    )

    teks = st.text_area(
        "Masukkan teks ulasan:",
        placeholder="Contoh: Aplikasi ini sangat membantu dan mudah digunakan..."
    )

    if st.button("Prediksi Sentimen", type="primary"):

        if not teks.strip():
            st.warning("Silakan masukkan teks terlebih dahulu.")

        else:
            teks_bersih = bersihkan_teks(teks)

            if not teks_bersih:
                st.warning(
                    "Teks tidak memiliki kata yang dapat diproses."
                )

            else:
                # Mengubah teks menjadi fitur TF-IDF
                teks_vector = vectorizer.transform([teks_bersih])

                # Melakukan prediksi
                prediksi = model.predict(teks_vector)[0]

                st.subheader("Hasil Prediksi")

                if prediksi == "positif":
                    st.success("Sentimen: POSITIF 😊")

                elif prediksi == "negatif":
                    st.error("Sentimen: NEGATIF 😠")

                else:
                    st.info("Sentimen: NETRAL 😐")

                # Menampilkan probabilitas prediksi
                if hasattr(model, "predict_proba"):

                    probabilitas = model.predict_proba(
                        teks_vector
                    )[0]

                    hasil_prob = pd.DataFrame({
                        "Sentimen": model.classes_,
                        "Probabilitas": probabilitas
                    })

                    hasil_prob["Persentase"] = (
                        hasil_prob["Probabilitas"] * 100
                    ).round(2)

                    st.subheader(
                        "Probabilitas Setiap Sentimen"
                    )

                    st.bar_chart(
                        hasil_prob.set_index("Sentimen")["Persentase"]
                    )

                    st.dataframe(
                        hasil_prob[["Sentimen", "Persentase"]],
                        use_container_width=True,
                        hide_index=True
                    )

# ==========================================
# 8. EVALUASI MODEL
# ==========================================

elif menu == "Evaluasi Model":

    st.header("📈 Evaluasi Model Neural Network")

    if os.path.exists(EVALUASI_PATH):

        evaluasi = pd.read_csv(EVALUASI_PATH)

        st.subheader("Hasil Evaluasi Model")

        st.dataframe(
            evaluasi,
            use_container_width=True,
            hide_index=True
        )

    else:
        st.warning("File evaluasi belum ditemukan.")

    st.divider()

    st.subheader("Confusion Matrix")

    if os.path.exists(CM_PATH):

        st.image(
            CM_PATH,
            caption="Confusion Matrix Neural Network Word + Character TF-IDF",
            use_container_width=True
        )

    else:
        st.warning("Gambar Confusion Matrix belum ditemukan.")

# ==========================================
# 9. DATASET
# ==========================================

elif menu == "Dataset":

    st.header("📁 Dataset Analisis Sentimen")

    st.write(
        "Berikut data ulasan yang digunakan "
        "dalam proses training."
    )

    st.metric("Jumlah Data", len(df))

    st.dataframe(
        df,
        use_container_width=True
    )

    csv = df.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="⬇️ Download Dataset CSV",
        data=csv,
        file_name="dataset_sentimen.csv",
        mime="text/csv"
    )

# ==========================================
# 10. FOOTER
# ==========================================

st.sidebar.divider()
st.sidebar.caption("Analisis Big Data - Test 4")
st.sidebar.caption(
    "Neural Network | Word + Character TF-IDF"
)
