import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import time

# Konfigurasi halaman dengan tema yang lebih menarik
st.set_page_config(
    page_title="Obesity Level Classification",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="collapsed"  # Collapse sidebar
)

# Custom CSS untuk dark theme yang lebih menarik
st.markdown("""
<style>
    /* Main app background */
    .stApp {
        background-color: #1e1e1e;
        color: #ffffff;
    }
    
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.37);
    }
    
    .input-card {
        background: #2d2d2d;
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.3);
        margin: 1rem 0;
        border-left: 4px solid #667eea;
        color: #ffffff;
    }
    
    .result-card {
        background: #2d2d2d;
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.3);
        margin: 1rem 0;
        border-top: 4px solid #28a745;
        color: #ffffff;
    }
    
    .section-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        text-align: center;
        font-weight: bold;
    }
    
    .metric-card {
        background: #3a3a3a;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #667eea;
        margin: 0.5rem 0;
        color: #ffffff;
    }
    
    .bmi-indicator {
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        font-weight: bold;
        margin: 1rem 0;
    }
    
    .bmi-normal {
        background: #1e4620;
        border: 2px solid #27ae60;
        color: #4ade80;
    }
    
    .bmi-underweight {
        background: #1e3a8a;
        border: 2px solid #3498db;
        color: #60a5fa;
    }
    
    .bmi-overweight {
        background: #451a03;
        border: 2px solid #f39c12;
        color: #fbbf24;
    }
    
    .bmi-obese {
        background: #450a0a;
        border: 2px solid #e74c3c;
        color: #f87171;
    }
    
    .progress-container {
        background: #2d2d2d;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.3);
        margin: 1rem 0;
    }
    
    .step {
        display: inline-block;
        padding: 0.5rem 1rem;
        margin: 0.25rem;
        border-radius: 20px;
        font-size: 0.9rem;
        font-weight: 500;
    }
    
    .step-completed {
        background: #28a745;
        color: white;
    }
    
    .step-active {
        background: #667eea;
        color: white;
    }
    
    .step-pending {
        background: #4a4a4a;
        color: #a0a0a0;
        border: 2px dashed #666666;
    }
    
    .warning-box {
        background: #44403c;
        border: 1px solid #a16207;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
        border-left: 4px solid #f39c12;
        color: #fbbf24;
    }
    
    .success-box {
        background: #0f2419;
        border: 1px solid #059669;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
        border-left: 4px solid #10b981;
        color: #4ade80;
    }
    
    .info-box {
        background: #1e293b;
        border: 1px solid #0284c7;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
        border-left: 4px solid #0ea5e9;
        color: #38bdf8;
    }
    
    /* Hide default streamlit elements */
    .stDeployButton {display:none;}
    footer {visibility: hidden;}
    .stApp > header {visibility: hidden;}
    
    /* Custom form styling for dark theme */
    .stSelectbox > div > div {
        background-color: #3a3a3a !important;
        border: 1px solid #555555 !important;
        border-radius: 8px;
        color: #ffffff !important;
    }
    
    .stSelectbox > div > div > div {
        color: #ffffff !important;
    }
    
    .stNumberInput > div > div > input {
        background-color: #3a3a3a !important;
        border: 1px solid #555555 !important;
        border-radius: 8px;
        color: #ffffff !important;
    }
    
    .stSlider > div > div > div {
        background-color: #667eea;
    }
    
    .stSlider > div > div > div > div {
        color: #ffffff !important;
    }
    
    /* Form label styling */
    .stApp label {
        color: #ffffff !important;
    }
    
    /* Metric styling */
    .metric-container > div {
        background-color: #2d2d2d !important;
        border: 1px solid #444444;
        border-radius: 8px;
        padding: 1rem;
    }
    
    .metric-container label {
        color: #a0a0a0 !important;
    }
    
    .metric-container > div > div:first-child {
        color: #ffffff !important;
        font-size: 1.5rem !important;
        font-weight: bold !important;
    }
    
    /* Form button styling */
    .stFormSubmitButton > button {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 25px !important;
        padding: 1rem 3rem !important;
        font-size: 1.2rem !important;
        font-weight: bold !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4) !important;
        width: 100% !important;
    }
    
    .stFormSubmitButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.6) !important;
    }
    
    /* Text color overrides */
    .stMarkdown, .stText, p, span, div {
        color: #ffffff;
    }
    
    h1, h2, h3, h4, h5, h6 {
        color: #ffffff !important;
    }
</style>
""", unsafe_allow_html=True)

# Fungsi untuk load model dan preprocessing objects dengan fallback
@st.cache_resource
def load_model_and_preprocessors():
    """Load model dan preprocessing objects dengan error handling yang lebih baik"""
    model_dir = "model"
    
    # Check if models directory exists
    if not os.path.exists(model_dir):
        st.error(f"❌ Folder '{model_dir}' tidak ditemukan!")
        st.info("💡 Pastikan folder 'models' ada dan berisi file-file model yang diperlukan.")
        return None, None, None, None
    
    # List of required files
    required_files = {
        "model": "Random_Forest_Tuned_best_model.joblib",
        "scaler": "scaler.joblib", 
        "label_encoder": "label_encoder.joblib",
        "feature_columns": "feature_columns.joblib"
    }
    
    # Check file existence
    missing_files = []
    for name, filename in required_files.items():
        filepath = os.path.join(model_dir, filename)
        if not os.path.exists(filepath):
            missing_files.append(filename)
    
    if missing_files:
        st.error(f"❌ File model tidak ditemukan: {', '.join(missing_files)}")
        st.info("💡 Jalankan 'python fix_model_compatibility.py' untuk membuat file yang diperlukan.")
        return None, None, None, None
    
    try:
        # Try loading with different methods for compatibility
        model_file = os.path.join(model_dir, required_files["model"])
        
        # Method 1: Standard joblib
        try:
            model = joblib.load(model_file)
            st.success("✅ Model berhasil dimuat!")
        except Exception as e1:
            st.warning(f"⚠️ Standard loading gagal: {e1}")
            
            # Method 2: Try fallback model
            fallback_file = os.path.join(model_dir, "fallback_model.joblib")
            if os.path.exists(fallback_file):
                st.info("🔄 Menggunakan fallback model...")
                model = joblib.load(fallback_file)
                st.warning("⚠️ Menggunakan model fallback (hanya untuk demo)")
            else:
                raise Exception("Model utama dan fallback tidak tersedia")
        
        # Load preprocessing objects
        scaler = joblib.load(os.path.join(model_dir, required_files["scaler"]))
        label_encoder = joblib.load(os.path.join(model_dir, required_files["label_encoder"]))
        feature_columns = joblib.load(os.path.join(model_dir, required_files["feature_columns"]))
        
        return model, scaler, label_encoder, feature_columns
        
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        
        # Provide detailed troubleshooting
        with st.expander("🔧 Troubleshooting Guide"):
            st.markdown("""
            **Kemungkinan penyebab:**
            1. **Scikit-learn version mismatch** - Model disimpan dengan versi sklearn berbeda
            2. **File model corrupt** - File rusak atau tidak lengkap
            3. **Python version incompatibility** - Perbedaan versi Python
            
            **Solusi:**
            1. **Re-install dependencies:**
               ```bash
               conda activate obesity-app
               pip uninstall scikit-learn -y
               pip install scikit-learn>=1.0.0
               ```
            
            2. **Jalankan compatibility fixer:**
               ```bash
               python fix_model_compatibility.py
               ```
            
            3. **Re-create model dari notebook training**
            
            4. **Gunakan model dummy untuk testing**
            """)
        
        return None, None, None, None

# Fungsi validasi input
def validate_input(input_data):
    """Validasi input data"""
    errors = []
    warnings = []
    
    # Validasi BMI
    bmi = input_data['Weight'] / (input_data['Height'] ** 2)
    if bmi < 10 or bmi > 50:
        errors.append("BMI tidak realistis. Periksa kembali tinggi dan berat badan.")
    
    # Validasi usia
    if input_data['Age'] < 10 or input_data['Age'] > 100:
        warnings.append("Usia di luar rentang normal (10-100 tahun).")
    
    # Validasi tinggi badan
    if input_data['Height'] < 1.0 or input_data['Height'] > 2.5:
        errors.append("Tinggi badan harus antara 1.0 - 2.5 meter.")
    
    # Validasi berat badan
    if input_data['Weight'] < 20 or input_data['Weight'] > 300:
        errors.append("Berat badan harus antara 20 - 300 kg.")
    
    return errors, warnings

# Fungsi untuk preprocessing input
def preprocess_input(input_data, scaler, feature_columns):
    """Preprocess input data"""
    try:
        df = pd.DataFrame([input_data])
        categorical_cols = ['Gender', 'CALC', 'FAVC', 'SCC', 'SMOKE', 'family_history_with_overweight', 'CAEC', 'MTRANS']
        
        for col in categorical_cols:
            if col in df.columns:
                dummies = pd.get_dummies(df[col], prefix=col, drop_first=True)
                df = pd.concat([df, dummies], axis=1)
                df.drop(col, axis=1, inplace=True)
        
        for col in feature_columns:
            if col not in df.columns:
                df[col] = 0
        
        df = df.reindex(columns=feature_columns, fill_value=0)
        scaled_data = scaler.transform(df)
        return scaled_data
    except Exception as e:
        st.error(f"❌ Error in preprocessing: {e}")
        return None

# Fungsi untuk membuat interpretasi hasil
def interpret_result(prediction, probabilities, label_encoder):
    """Interpretasi hasil prediksi dengan informasi lengkap"""
    
    class_descriptions = {
        'Insufficient_Weight': {
            'label': '📉 Berat Badan Kurang',
            'description': 'BMI di bawah normal. Perlu penambahan berat badan dengan pola makan sehat.',
            'color': '#3498db',
            'risk_level': 'Rendah-Sedang',
            'recommendations': [
                '🥗 Konsultasi dengan ahli gizi profesional',
                '💪 Tingkatkan asupan kalori dengan makanan bergizi',
                '🏋️ Lakukan latihan pembentukan massa otot',
                '📊 Monitor kesehatan dan berat badan rutin'
            ]
        },
        'Normal_Weight': {
            'label': '✅ Berat Badan Normal',
            'description': 'BMI dalam rentang ideal. Pertahankan gaya hidup sehat saat ini.',
            'color': '#27ae60',
            'risk_level': 'Rendah',
            'recommendations': [
                '🥘 Pertahankan pola makan seimbang',
                '🏃 Rutin berolahraga 150 menit per minggu',
                '😴 Tidur cukup 7-9 jam setiap hari',
                '🧘 Kelola stress dengan meditasi atau yoga'
            ]
        },
        'Overweight_Level_I': {
            'label': '⚠️ Kelebihan Berat Badan Tingkat I',
            'description': 'BMI sedikit di atas normal. Waktunya memperhatikan pola hidup.',
            'color': '#f39c12',
            'risk_level': 'Sedang',
            'recommendations': [
                '🍽️ Kurangi porsi makan secara bertahap',
                '🚶 Tingkatkan aktivitas fisik harian',
                '🚫 Batasi makanan tinggi kalori dan gula',
                '🥬 Perbanyak konsumsi sayur dan buah-buahan'
            ]
        },
        'Overweight_Level_II': {
            'label': '⚠️ Kelebihan Berat Badan Tingkat II',
            'description': 'BMI mendekati obesitas. Diperlukan perubahan gaya hidup signifikan.',
            'color': '#e67e22',
            'risk_level': 'Sedang-Tinggi',
            'recommendations': [
                '👨‍⚕️ Konsultasi dengan dokter atau ahli gizi',
                '📋 Ikuti program penurunan berat badan terstruktur',
                '🏃‍♂️ Olahraga rutin minimal 5x seminggu',
                '📈 Monitor progres dengan aplikasi kesehatan'
            ]
        },
        'Obesity_Type_I': {
            'label': '🔴 Obesitas Tipe I',
            'description': 'BMI kategori obesitas ringan. Memerlukan intervensi medis.',
            'color': '#e74c3c',
            'risk_level': 'Tinggi',
            'recommendations': [
                '🏥 Konsultasi medis segera dengan dokter',
                '📋 Program diet ketat dengan supervisi ahli',
                '💊 Evaluasi kemungkinan terapi medis',
                '🔍 Screening risiko penyakit penyerta'
            ]
        },
        'Obesity_Type_II': {
            'label': '🔴 Obesitas Tipe II',
            'description': 'BMI kategori obesitas sedang. Risiko kesehatan tinggi.',
            'color': '#c0392b',
            'risk_level': 'Sangat Tinggi',
            'recommendations': [
                '🚨 Perawatan medis intensif segera',
                '📊 Program penurunan berat badan ketat',
                '🔬 Monitoring kesehatan komprehensif',
                '⚕️ Evaluasi opsi intervensi bedah'
            ]
        },
        'Obesity_Type_III': {
            'label': '🚨 Obesitas Tipe III (Ekstrem)',
            'description': 'BMI kategori obesitas berat. Risiko kesehatan sangat tinggi.',
            'color': '#8e44ad',
            'risk_level': 'Kritis',
            'recommendations': [
                '🚨 Perawatan medis darurat diperlukan',
                '⚕️ Konsultasi bedah bariatrik',
                '🏥 Monitoring intensif berkelanjutan',
                '👥 Terapi multidisiplin (dokter, psikolog, ahli gizi)'
            ]
        }
    }
    
    class_name = label_encoder.inverse_transform([prediction])[0]
    class_info = class_descriptions.get(class_name, {})
    return class_name, class_info, probabilities

# Progress tracker untuk user experience
def show_progress_steps(current_step):
    """Menampilkan progress steps"""
    steps = ["📝 Input Data", "🔍 Validasi", "🤖 Prediksi", "📊 Hasil"]
    
    st.markdown('<div class="progress-container">', unsafe_allow_html=True)
    
    progress_html = '<div style="text-align: center; margin: 1rem 0;">'
    for i, step in enumerate(steps):
        if i < current_step:
            progress_html += f'<span class="step step-completed">✅ {step}</span>'
        elif i == current_step:
            progress_html += f'<span class="step step-active">🔄 {step}</span>'
        else:
            progress_html += f'<span class="step step-pending">⏳ {step}</span>'
    progress_html += '</div>'
    
    st.markdown(progress_html, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

def calculate_bmi_status(height, weight):
    """Calculate BMI and return status"""
    if height > 0 and weight > 0:
        bmi = weight / (height ** 2)
        if bmi < 18.5:
            return bmi, "Underweight 📉", "bmi-underweight"
        elif bmi < 25:
            return bmi, "Normal ✅", "bmi-normal"
        elif bmi < 30:
            return bmi, "Overweight ⚠️", "bmi-overweight"
        else:
            return bmi, "Obese 🔴", "bmi-obese"
    return 0, "Invalid", "bmi-normal"

# Main App dengan UI/UX yang lebih baik
def main():
    # Header dengan design yang menarik
    st.markdown("""
    <div class="main-header">
        <h1>🏥 Sistem Klasifikasi Tingkat Obesitas</h1>
        <p>Powered by Machine Learning • Akurasi 96% • Random Forest Algorithm</p>
        <small>Universitas Dian Nuswantoro - Teknik Informatika</small>
    </div>
    """, unsafe_allow_html=True)
    
    # Load model dengan feedback yang lebih baik
    with st.spinner("🔄 Memuat model AI..."):
        model, scaler, label_encoder, feature_columns = load_model_and_preprocessors()
    
    if model is None:
        st.stop()
    
    # Initialize session state untuk tracking progress
    if 'step' not in st.session_state:
        st.session_state.step = 0
    if 'input_complete' not in st.session_state:
        st.session_state.input_complete = False
    if 'prediction_made' not in st.session_state:
        st.session_state.prediction_made = False
    
    # Progress indicator
    show_progress_steps(st.session_state.step)
    
    # Main form area dengan layout yang lebih baik
    st.markdown('<div class="section-header"><h2>📋 Form Input Data Kesehatan</h2></div>', unsafe_allow_html=True)
    st.markdown("Silakan isi semua informasi dengan akurat untuk hasil prediksi terbaik.")
    
    # Create form layout dengan columns
    with st.form("health_data_form"):
        # Personal Information Section
        st.markdown('<div class="input-card">', unsafe_allow_html=True)
        st.subheader("👤 Informasi Pribadi")
        
        col1, col2 = st.columns(2)
        with col1:
            age = st.slider("🎂 Usia (tahun)", 14, 100, 25, help="Masukkan usia Anda dalam tahun")
            height = st.number_input("📏 Tinggi Badan (meter)", 1.0, 2.5, 1.70, 0.01, 
                                   help="Contoh: 1.70 untuk tinggi 170 cm")
        with col2:
            gender = st.selectbox("⚧ Jenis Kelamin", ["Male", "Female"], help="Pilih jenis kelamin")
            weight = st.number_input("⚖️ Berat Badan (kg)", 30.0, 300.0, 70.0, 0.1,
                                   help="Masukkan berat badan dalam kilogram")
        
        # Real-time BMI calculation
        bmi, bmi_status, bmi_class = calculate_bmi_status(height, weight)
        st.markdown(f"""
        <div class="bmi-indicator {bmi_class}">
            <strong>BMI Anda: {bmi:.1f}</strong><br>
            <small>Status: {bmi_status}</small>
        </div>
        """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Family History Section
        st.markdown('<div class="input-card">', unsafe_allow_html=True)
        st.subheader("👨‍👩‍👧‍👦 Riwayat Keluarga")
        family_history = st.selectbox("Riwayat Keluarga Obesitas", ["no", "yes"],
                                    help="Apakah ada anggota keluarga dengan riwayat kelebihan berat badan?")
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Eating Habits Section
        st.markdown('<div class="input-card">', unsafe_allow_html=True)
        st.subheader("🍽️ Kebiasaan Makan")
        
        col1, col2 = st.columns(2)
        with col1:
            favc = st.selectbox("Sering Makan Tinggi Kalori?", ["no", "yes"],
                              help="Apakah Anda sering mengonsumsi makanan tinggi kalori?")
            fcvc = st.slider("🥗 Konsumsi Sayuran (per hari)", 0.0, 3.0, 2.0, 0.1,
                           help="Seberapa sering Anda makan sayuran setiap hari")
        with col2:
            ncp = st.slider("🍽️ Makanan Utama (per hari)", 1.0, 4.0, 3.0, 0.1,
                          help="Berapa kali Anda makan besar dalam sehari")
            caec = st.selectbox("🍿 Frekuensi Cemilan", ["no", "Sometimes", "Frequently", "Always"],
                              help="Seberapa sering Anda makan cemilan di antara waktu makan?")
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Lifestyle Section
        st.markdown('<div class="input-card">', unsafe_allow_html=True)
        st.subheader("🏃‍♂️ Gaya Hidup")
        
        col1, col2 = st.columns(2)
        with col1:
            smoke = st.selectbox("🚬 Merokok?", ["no", "yes"])
            ch2o = st.slider("💧 Konsumsi Air (liter/hari)", 1.0, 3.0, 2.0, 0.1,
                           help="Berapa liter air yang Anda minum setiap hari")
            scc = st.selectbox("📊 Monitor Kalori?", ["no", "yes"],
                             help="Apakah Anda memantau asupan kalori harian?")
        with col2:
            faf = st.slider("🏋️ Aktivitas Fisik (per minggu)", 0.0, 3.0, 1.0, 0.1,
                          help="Seberapa sering Anda melakukan aktivitas fisik per minggu")
            tue = st.slider("📱 Penggunaan Teknologi (jam/hari)", 0.0, 2.0, 1.0, 0.1,
                          help="Berapa jam per hari Anda menggunakan perangkat teknologi")
            calc = st.selectbox("🍷 Konsumsi Alkohol", ["no", "Sometimes", "Frequently", "Always"])
        
        mtrans = st.selectbox("🚗 Transportasi Utama", 
                            ["Public_Transportation", "Automobile", "Walking", "Motorbike", "Bike"],
                            help="Jenis transportasi apa yang biasa Anda gunakan?")
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Submit button
        st.markdown('<br>', unsafe_allow_html=True)
        submitted = st.form_submit_button("🔮 Analisis Tingkat Obesitas", use_container_width=True)
        
        if submitted:
            st.session_state.step = 1
            
            # Prepare input data
            input_data = {
                'Age': age, 'Height': height, 'Weight': weight, 'FCVC': fcvc,
                'NCP': ncp, 'CH2O': ch2o, 'FAF': faf, 'TUE': tue,
                'Gender': gender, 'CALC': calc, 'FAVC': favc, 'SCC': scc,
                'SMOKE': smoke, 'family_history_with_overweight': family_history,
                'CAEC': caec, 'MTRANS': mtrans
            }
            
            # Validasi input
            errors, warnings = validate_input(input_data)
            
            if errors:
                st.session_state.step = 1
                st.markdown("""
                <div class="warning-box">
                    <h4>❌ Error dalam Input</h4>
                </div>
                """, unsafe_allow_html=True)
                for error in errors:
                    st.error(error)
            else:
                st.session_state.step = 2
                
                if warnings:
                    st.markdown("""
                    <div class="warning-box">
                        <h4>⚠️ Peringatan</h4>
                    </div>
                    """, unsafe_allow_html=True)
                    for warning in warnings:
                        st.warning(warning)
                
                # Processing dengan animasi
                with st.spinner("🔄 AI sedang menganalisis data Anda..."):
                    time.sleep(1)  # Simulasi processing time untuk UX yang lebih baik
                    
                    processed_data = preprocess_input(input_data, scaler, feature_columns)
                    
                    if processed_data is not None:
                        prediction = model.predict(processed_data)[0]
                        probabilities = model.predict_proba(processed_data)[0]
                        class_name, class_info, probs = interpret_result(prediction, probabilities, label_encoder)
                        
                        st.session_state.step = 3
                        st.session_state.prediction_made = True
                        
                        # Hasil prediksi dengan design yang menarik
                        st.markdown("---")
                        
                        # Main result card
                        st.markdown(f"""
                        <div class="result-card">
                            <div style="text-align: center;">
                                <h2 style="color: {class_info.get('color', '#3498db')}; margin-bottom: 1rem;">
                                    {class_info.get('label', class_name)}
                                </h2>
                                <p style="font-size: 1.2rem; margin-bottom: 1rem;">
                                    {class_info.get('description', 'Tidak ada deskripsi tersedia.')}
                                </p>
                                <div style="background: {class_info.get('color', '#3498db')}20; 
                                          padding: 1rem; border-radius: 10px; margin: 1rem 0;">
                                    <strong>Tingkat Risiko: {class_info.get('risk_level', 'Unknown')}</strong>
                                </div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Metrics row
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            confidence = max(probabilities) * 100
                            st.metric("🎯 Confidence Score", f"{confidence:.1f}%")
                        
                        with col2:
                            bmi = weight / (height ** 2)
                            st.metric("📏 BMI Anda", f"{bmi:.1f}")
                        
                        with col3:
                            if bmi < 18.5:
                                ideal_weight = 18.5 * (height ** 2)
                            elif bmi > 25:
                                ideal_weight = 25 * (height ** 2)
                            else:
                                ideal_weight = weight
                            st.metric("⚖️ Berat Ideal", f"{ideal_weight:.1f} kg")
                        
                        with col4:
                            weight_diff = weight - ideal_weight
                            if weight_diff > 0:
                                st.metric("📊 Selisih Berat", f"+{weight_diff:.1f} kg", delta=f"{weight_diff:.1f}")
                            else:
                                st.metric("📊 Selisih Berat", f"{weight_diff:.1f} kg", delta=f"{weight_diff:.1f}")
                        
                        # Probability distribution dengan chart yang lebih menarik
                        st.markdown("### 📈 Distribusi Probabilitas Prediksi")
                        
                        class_names = label_encoder.classes_
                        prob_df = pd.DataFrame({
                            'Kelas': [name.replace('_', ' ') for name in class_names],
                            'Probabilitas': probs * 100,
                            'Color': ['#ff6b6b' if name == class_name else '#4ecdc4' for name in class_names]
                        })
                        
                        fig = px.bar(prob_df, x='Probabilitas', y='Kelas', orientation='h',
                                   title="Confidence Score untuk Setiap Kategori Obesitas",
                                   color='Probabilitas',
                                   color_continuous_scale='viridis',
                                   text='Probabilitas')
                        
                        fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
                        fig.update_layout(height=400, showlegend=False)
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Recommendations section
                        st.markdown("### 💡 Rekomendasi Kesehatan Personal")
                        recommendations = class_info.get('recommendations', [])
                        
                        if recommendations:
                            for rec in recommendations:
                                st.markdown(f"""
                                <div class="metric-card">
                                    {rec}
                                </div>
                                """, unsafe_allow_html=True)
                        
                        # BMI Visualization yang lebih interaktif
                        st.markdown("### 📊 Analisis BMI Komprehensif")
                        
                        # Create BMI gauge chart
                        fig_gauge = go.Figure(go.Indicator(
                            mode = "gauge+number+delta",
                            value = bmi,
                            domain = {'x': [0, 1], 'y': [0, 1]},
                            title = {'text': "BMI Anda"},
                            delta = {'reference': 22.5},  # Middle of normal range
                            gauge = {
                                'axis': {'range': [None, 40]},
                                'bar': {'color': class_info.get('color', '#3498db')},
                                'steps': [
                                    {'range': [0, 18.5], 'color': "#3498db"},
                                    {'range': [18.5, 25], 'color': "#27ae60"},
                                    {'range': [25, 30], 'color': "#f39c12"},
                                    {'range': [30, 40], 'color': "#e74c3c"}
                                ],
                                'threshold': {
                                    'line': {'color': "red", 'width': 4},
                                    'thickness': 0.75,
                                    'value': bmi
                                }
                            }
                        ))
                        
                        fig_gauge.update_layout(height=400)
                        st.plotly_chart(fig_gauge, use_container_width=True)
                        
                        # Summary dan action items
                        st.markdown("### 📋 Ringkasan & Langkah Selanjutnya")
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.markdown(f"""
                            <div class="success-box">
                                <h4>📊 Ringkasan Hasil</h4>
                                <ul>
                                    <li><strong>Kategori:</strong> {class_info.get('label', class_name)}</li>
                                    <li><strong>BMI:</strong> {bmi:.1f}</li>
                                    <li><strong>Confidence:</strong> {max(probabilities) * 100:.1f}%</li>
                                    <li><strong>Risk Level:</strong> {class_info.get('risk_level', 'Unknown')}</li>
                                </ul>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        with col2:
                            st.markdown("""
                            <div class="info-box">
                                <h4>⚠️ Disclaimer</h4>
                                <p>Hasil prediksi ini hanya untuk referensi dan tidak menggantikan konsultasi medis profesional. 
                                Selalu konsultasikan dengan dokter untuk diagnosis yang akurat.</p>
                            </div>
                            """, unsafe_allow_html=True)

    # Footer dengan informasi lengkap
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: gray; padding: 2rem;">
        <h4>🎓 Capstone Project - Bengkel Koding Data Science</h4>
        <p>📚 <strong>Universitas Dian Nuswantoro</strong> | Program Studi Teknik Informatika</p>
        <p>🤖 Powered by <strong>Random Forest Algorithm</strong> with <strong>96.02% Accuracy</strong></p>
        <p>🔬 Model trained on 2,091 samples with 23 features</p>
        <hr style="margin: 1rem 0;">
        <small>
            <strong>Technology Stack:</strong> Python • Scikit-learn • Streamlit • Plotly<br>
            <strong>Data Source:</strong> Obesity Dataset (Mexico, Peru, Colombia)<br>
            <strong>Last Updated:</strong> June 2025
        </small>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()