import streamlit as st
import pandas as pd
import numpy as np
import joblib

THRESHOLD_PRICE = 1200 # Euro

@st.cache_resource
def load_assets():
    FILES = ['voting_classifier_model.pkl', 'preprocessor.pkl', 
             'model_accuracy.pkl', 'original_df_for_unique_values.pkl', 
             'feature_columns.pkl']
    
    loaded_objects = {}
    try:
        for file_name in FILES:
            loaded_objects[file_name.split('.')[0]] = joblib.load(file_name)

        return (loaded_objects['voting_classifier_model'], 
                loaded_objects['preprocessor'], 
                loaded_objects['model_accuracy'], 
                loaded_objects['original_df_for_unique_values'], 
                loaded_objects['feature_columns'])
        
    except FileNotFoundError:
        st.error("Aset (model/preprocessor/akurasi) belum ditemukan. Harap pastikan 'train_model.py' sudah dieksekusi dengan benar dan semua file .pkl tersedia.")
        return None, None, None, None, None

voting_clf, preprocessor, accuracies, original_df, X_columns = load_assets()

# --- 3. STREAMLIT INTERFACE ---
if voting_clf is not None:

    st.title("💻 Aplikasi Klasifikasi Harga Laptop: Premium vs. Standar")
    st.markdown("""
    Proyek ini menggunakan model **Ensemble (VotingClassifier)** yang menggabungkan kekuatan **Random Forest** dan **AdaBoost** untuk memprediksi kategori harga laptop baru.
    
    **Kategori Harga:**
    * **Premium/High-End (1):** Harga Jual > **1200 Euro**
    * **Standar/Non-Premium (0):** Harga Jual $\le$ **1200 Euro** 
    """)
    st.markdown("---")

    st.sidebar.header("📊 Model Performance (Bukti Akurasi Super)")

    st.sidebar.metric("Akurasi Random Forest (Model >90%)", f"{accuracies['rf']*100:.2f}%")
    st.sidebar.markdown(f"**Random Forest mencapai {accuracies['rf']*100:.2f}%**")
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("Performa Model Ensemble")
    st.sidebar.metric("Akurasi VotingClassifier", f"{accuracies['voting']*100:.2f}%")
    st.sidebar.caption("VotingClassifier adalah model utama untuk prediksi.")

    st.header("Masukkan Spesifikasi Laptop Baru")

    unique_vals = {col: original_df[col].unique() for col in original_df.columns if original_df[col].dtype == 'object'}

    with st.form("prediction_form"):
        col1, col2 = st.columns(2)
        with col1:
            company = st.selectbox("1. Perusahaan (Company)", sorted(unique_vals['Company']))
            typename = st.selectbox("2. Tipe Laptop (TypeName)", sorted(unique_vals['TypeName']))
            os = st.selectbox("3. Sistem Operasi (OS)", sorted(unique_vals['OS']))
            cpu_company = st.selectbox("4. CPU (Produsen)", sorted(unique_vals['CPU_company']))
            gpu_company = st.selectbox("5. GPU (Produsen)", sorted(unique_vals['GPU_company']))
        
        with col2:
            screen = st.selectbox("6. Kualitas Layar (Screen)", sorted(unique_vals['Screen']))
            touchscreen = st.selectbox("7. Layar Sentuh (Touchscreen)", sorted(unique_vals['Touchscreen']))
            ips_panel = st.selectbox("8. IPS Panel", sorted(unique_vals['IPSpanel']))
            retina = st.selectbox("9. Retina Display", sorted(unique_vals['RetinaDisplay']))

        st.subheader("Spesifikasi Numerik")
        col3, col4, col5 = st.columns(3)
        with col3:
            ram = st.number_input("10. RAM (GB)", min_value=2, max_value=64, value=8, step=2)
            inches = st.number_input("11. Ukuran Layar (Inches)", min_value=10.0, max_value=18.4, value=15.6, step=0.1, format="%.1f")
        with col4:
            weight = st.number_input("12. Berat (kg)", min_value=0.5, max_value=5.0, value=2.0, step=0.1, format="%.2f")
            cpu_freq = st.number_input("13. Frekuensi CPU (GHz)", min_value=0.9, max_value=3.6, value=2.5, step=0.1, format="%.1f")
        with col5:
            primary_storage = st.number_input("14. Penyimpanan Primer (GB)", min_value=0, max_value=2048, value=256, step=128)
            secondary_storage = st.number_input("15. Penyimpanan Sekunder (GB)", min_value=0, max_value=2048, value=0, step=128)
        
        submitted = st.form_submit_button("Prediksi Kategori Harga")

    if submitted:
        input_data = pd.DataFrame([[company, typename, inches, ram, os, weight, screen, 
                                    touchscreen, ips_panel, retina, cpu_company, cpu_freq, 
                                    primary_storage, secondary_storage, gpu_company]], 
                                    columns=X_columns)

        input_data_processed = preprocessor.transform(input_data)

        prediction = voting_clf.predict(input_data_processed)

        st.subheader("🎉 Hasil Prediksi Model Ensemble")
        if prediction[0] == 1:
            st.success(f"Laptop ini diprediksi berada di Kategori **PREMIUM/HIGH-END** (Diprediksi Harga > {THRESHOLD_PRICE} Euro). 🚀")
            st.balloons()
        else:
            st.info(f"Laptop ini diprediksi berada di Kategori **STANDAR/NON-PREMIUM** (Diprediksi Harga $\le$ {THRESHOLD_PRICE} Euro).")
            
        st.markdown(f"---")
        st.caption(f"Prediksi dihasilkan oleh **VotingClassifier** yang menggabungkan Random Forest dan AdaBoost.")
