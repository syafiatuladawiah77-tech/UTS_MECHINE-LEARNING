import streamlit as st
import joblib
import pandas as pd

# ================== LOAD ==================
model = joblib.load("pipeline.pkl")
encoders = joblib.load("encoder_dict.pkl")

# ================== TITLE ==================
st.title("Klasifikasi Tingkat Pengalaman Profesional")
st.markdown("---")

# ================== INPUT ==================
st.subheader("Input Data")

col1, col2 = st.columns(2)

# ===== KOLOM KIRI =====
with col1:
    years_experience = st.number_input("Years Experience", 0, 20, 1)
    salary = st.number_input("Salary", 0, 200000, 5000)

    skills_python = st.selectbox("Python Skill", [0, 1])
    skills_sql = st.selectbox("SQL Skill", [0, 1])
    skills_ml = st.selectbox("ML Skill", [0, 1])

# ===== KOLOM KANAN =====
with col2:
    skills_deep_learning = st.selectbox("Deep Learning", [0, 1])
    skills_cloud = st.selectbox("Cloud Skill", [0, 1])

    job_openings = st.number_input("Job Openings", 0, 50, 1)
    job_posting_month = st.selectbox("Posting Month", list(range(1, 13)))
    job_posting_year = st.number_input("Posting Year", 2020, 2030, 2025)

st.markdown("---")

# ================== INPUT KATEGORI ==================
st.subheader("Informasi Perusahaan")

col3, col4 = st.columns(2)

with col3:
    company_size = st.selectbox("Company Size", encoders['company_size'].classes_)
    company_industry = st.selectbox("Industry", encoders['company_industry'].classes_)
    country = st.selectbox("Country", encoders['country'].classes_)

with col4:
    remote_type = st.selectbox("Remote Type", encoders['remote_type'].classes_)
    hiring_urgency = st.selectbox("Hiring Urgency", encoders['hiring_urgency'].classes_)
    education_level = st.selectbox("Education Level", encoders['education_level'].classes_)

st.markdown("---")

# ================== SUSUN DATA ==================
input_data = {
    'years_experience': years_experience,
    'salary': salary,
    'skills_python': skills_python,
    'skills_sql': skills_sql,
    'skills_ml': skills_ml,
    'skills_deep_learning': skills_deep_learning,
    'skills_cloud': skills_cloud,
    'job_openings': job_openings,
    'job_posting_month': job_posting_month,
    'job_posting_year': job_posting_year,
    'company_size': company_size,
    'company_industry': company_industry,
    'country': country,
    'remote_type': remote_type,
    'hiring_urgency': hiring_urgency,
    'education_level': education_level
}

df = pd.DataFrame([input_data])

# ================== ENCODING ==================
try:
    for col, encoder in encoders.items():
        df[col] = encoder.transform(df[col])
except Exception as e:
    st.error(f"Terjadi kesalahan encoding: {e}")
    st.stop()

# ================== URUTAN KOLOM ==================
try:
    df = df[model.feature_names_in_]
except:
    pass

# ================== PREDIKSI ==================
if st.button("Prediksi"):
    hasil = model.predict(df)

    mapping = {
        0: "Entry",
        1: "Mid",
        2: "Senior"
    }

    st.success(f"🎯 Hasil Klasifikasi: {mapping[hasil[0]]}")

    # ===== PROBABILITAS =====
    try:
        proba = model.predict_proba(df)[0]
        st.subheader("Probabilitas:")
        st.write({
            "Entry": round(proba[0], 3),
            "Mid": round(proba[1], 3),
            "Senior": round(proba[2], 3)
        })
    except:
        pass