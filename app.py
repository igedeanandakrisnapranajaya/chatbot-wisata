import streamlit as st
import pandas as pd
import requests
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ==========================================
# ⚙️ KONFIGURASI HALAMAN & API
# ==========================================
st.set_page_config(page_title="Konco Plesir", page_icon="✈️")
st.title("✈️ Konco Plesir: Chatbot Wisata Jawa")
st.write("Tanya rekomendasi wisata dan kuliner di sini!")

# Ambil API Key dari Streamlit Secrets (Akan disetting nanti di dashboard)
try:
    API_KEY = st.secrets["GEMINI_API_KEY"]
except:
    st.error("API Key belum disetting di Streamlit Secrets!")
    st.stop()

FILE_DATASET = 'datasetpariwisata_jawa_makanan.csv'
COL_NAMA     = 'place_name'
COL_RATING   = 'rating'
COL_ALAMAT   = 'address'
COL_KOTA     = 'city'
COL_PROVINSI = 'province'
COL_MAKANAN  = 'makanan_khas'

# --- FUNGSI LOAD DATA (Dicache biar tidak berat) ---
@st.cache_resource
def load_data_and_model():
    try:
        df = pd.read_csv(FILE_DATASET, sep=';')
        # Cleaning data
        df['rating_num'] = pd.to_numeric(df[COL_RATING].astype(str).str.replace(',', '.'), errors='coerce')
        df[COL_MAKANAN] = df[COL_MAKANAN].fillna('Kuliner Lokal')
        df['search_content'] = (df[COL_NAMA].astype(str) + " " + df[COL_KOTA].astype(str)).str.lower()
        
        # TF-IDF Setup
        tfidf = TfidfVectorizer()
        tfidf_matrix = tfidf.fit_transform(df['search_content'].fillna(''))
        return df, tfidf, tfidf_matrix
    except Exception as e:
        return None, None, None

df, tfidf, tfidf_matrix = load_data_and_model()

if df is None:
    st.error(f"Gagal memuat dataset: {FILE_DATASET}. Pastikan file ada di folder yang sama.")
    st.stop()

# --- FUNGSI CHAT KE GEMINI ---
def get_gemini_response(user_input, context_data):
    # Setup Model (Hardcoded ke v1beta/flash agar stabil)
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent?key={API_KEY}"
    
    prompt = {
        "contents": [{
            "parts": [{"text": f"Kamu adalah 'Konco Plesir'. Bantu user ini: '{user_input}'. {context_data} Jawab santai, singkat, dan pake bahasa gaul."}]
        }]
    }
    
    try:
        r = requests.post(url, json=prompt)
        if r.status_code == 200:
            return r.json()['candidates'][0]['content']['parts'][0]['text']
        else:
            return f"⚠️ Error API: {r.status_code}"
    except Exception as e:
        return f"⚠️ Error Koneksi: {e}"

# --- LOGIKA REKOMENDASI ---
def get_recommendation_context(user_input):
    try:
        vec = tfidf.transform([user_input.lower()])
        sim = cosine_similarity(vec, tfidf_matrix).flatten()
        top_idx = sim.argsort()[-2:][::-1]
        
        context_data = ""
        if sim[top_idx[0]] > 0.1:
            context_data = "Data Rekomendasi dari Database:\n"
            for i in top_idx:
                row = df.iloc[i]
                context_data += f"- {row[COL_NAMA]} ({row[COL_KOTA]}), Kuliner: {row[COL_MAKANAN]}\n"
        return context_data
    except:
        return ""

# --- INTERFACE CHAT ---
# Simpan history chat di session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Tampilkan chat sebelumnya
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Input user baru
if prompt := st.chat_input("Mau liburan ke mana bro?"):
    # Tampilkan pesan user
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Proses jawaban
    context = get_recommendation_context(prompt)
    response = get_gemini_response(prompt, context)

    # Tampilkan balasan bot
    with st.chat_message("assistant"):
        st.markdown(response)

    st.session_state.messages.append({"role": "assistant", "content": response})
