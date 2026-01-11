import streamlit as st
import pandas as pd
import joblib
from PIL import Image

# Konfigurasi Halaman (Harus di paling atas)
st.set_page_config(
    page_title="Student Performance AI",
    page_icon="🎓",
    layout="wide"
)

# Custom CSS untuk mempercantik tampilan
st.markdown("""
    <style>
    .main {
        background-color: #f5f7f9;
    }
    .stButton>button {
        width: 100%;
        border-radius: 20px;
        height: 3em;
        background-color: #007bff;
        color: white;
        font-weight: bold;
    }
    .stMetric {
        background-color: #ffffff;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    </style>
    """, unsafe_allow_html=True)

def prediction_app():
    # Sidebar untuk informasi tambahan
    with st.sidebar:
        st.image("https://cdn-icons-png.flaticon.com/512/3413/3413535.png", width=100)
        st.title("About")
        st.info("""
        Aplikasi ini menggunakan **Machine Learning** untuk memprediksi performa akademik berdasarkan kebiasaan belajar siswa.
        """)
        st.write("---")
        st.caption("Model: Regression & Logistic Regression")

    # Header Utama
    st.title("🎓 Student Performance Predictor")
    st.markdown("Sistem cerdas untuk memprediksi indeks prestasi dan klasifikasi performa siswa.")
    
    # Load Model dengan caching agar cepat
    @st.cache_resource
    def load_models():
        reg = joblib.load("regression_student_performance.pkl")
        log = joblib.load("logistic_regression_student_performance.pkl")
        return reg, log

    try:
        reg, log = load_models()
    except Exception as e:
        st.error(f"Gagal memuat model. Pastikan file .pkl tersedia. Error: {e}")
        return

    # Form Input dengan Container dan Columns
    with st.container():
        st.write("### 📝 Input Data Siswa")
        col1, col2 = st.columns(2)
        
        with col1:
            sh = st.slider("Hours Studied", 0, 24, 7, help="Jumlah jam belajar per hari")
            ps = st.number_input("Previous Score", 0, 100, 80)
            
        with col2:
            sleep_h = st.slider("Sleep Hours", 0, 24, 6)
            practiced = st.number_input("Question Papers Practiced", 0, 100, 5)

    # Tombol Prediksi di Tengah
    st.write("---")
    if st.button("✨ Analisis Performa Sekarang"):
        # Menyiapkan data
        input_data = pd.DataFrame([{
            'Hours Studied': sh,
            'Previous Scores': ps,
            'Sleep Hours': sleep_h,
            'Sample Question Papers Practiced': practiced
        }])

        # Proses Prediksi
        with st.spinner('Menghitung hasil...'):
            pred_index = reg.predict(input_data)[0]
            pred_class = log.predict(input_data)[0]

        # Menampilkan Hasil dengan Layout Menarik
        st.subheader("📊 Hasil Analisis")
        
        res_col1, res_col2 = st.columns(2)
        
        with res_col1:
            st.metric(label="Predicted Performance Index", value=f"{pred_index:.2f}")
        
        with res_col2:
            # Memberikan warna berbeda berdasarkan kelas
            class_color = "Green" if pred_class in ["High", "A", 1] else "Orange"
            st.markdown(f"""
                <div class='stMetric'>
                    <p style='color:gray; font-size:14px; margin:0;'>Performance Class</p>
                    <h2 style='color:{class_color}; margin:0;'>{pred_class}</h2>
                </div>
            """, unsafe_allow_html=True)

        # Feedback Visual Terpadu
        st.write("")
        if pred_index > 80:
            st.success("🌟 **Luar Biasa!** Siswa diprediksi memiliki performa yang sangat tinggi. Pertahankan ritme belajarnya.")
        elif pred_index > 50:
            st.warning("📈 **Performa Stabil.** Siswa berada di jalur yang benar, namun masih ada ruang untuk peningkatan.")
        else:
            st.error("⚠️ **Perhatian.** Siswa memerlukan bantuan tambahan atau penyesuaian strategi belajar.")

if __name__ == "__main__":
    prediction_app()