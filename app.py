import streamlit as st
import pandas as pd
import google.generativeai as genai
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ==========================================
# ⚙️ KONFIGURASI HALAMAN
# ==========================================
st.set_page_config(page_title="Konco Plesir", page_icon="✈️")
st.title("✈️ Konco Plesir: Chatbot Wisata Jawa")

# ==========================================
# 🔑 SETUP API & AUTO-DETECT MODEL
# ==========================================
ACTIVE_MODEL = "gemini-1.5-flash" # Default sementara

try:
    # 1. Konfigurasi API
    api_key = st.secrets["GEMINI_API_KEY"]
    genai.configure(api_key=api_key)
    
    # 2. DETEKSI MODEL OTOMATIS (Biar tidak error 404)
    st.sidebar.header("🔧 Status Sistem")
    
    valid_models = []
    # Cari semua model yang mendukung generateContent
    for m in genai.list_models():
        if 'generateContent' in m.supported_generation_methods:
            valid_models.append(m.name)
            
    if valid_models:
        # Ambil model pertama yang ditemukan
        ACTIVE_MODEL = valid_models[0]
        st.sidebar.success(f"✅ Terhubung ke: {ACTIVE_MODEL}")
        # Tampilkan list model di sidebar (untuk debugging)
        with st.sidebar.expander("Lihat semua model tersedia"):
            st.write(valid_models)
    else:
        st.sidebar.error("⚠️ API Key valid, tapi tidak ada model yang tersedia.")

except Exception as e:
    st.sidebar.error(f"⚠️ Masalah API: {str(e)}")
    st.error("Cek API Key di Streamlit Secrets!")
    st.stop()

# ==========================================
# 📂 SETUP DATASET
# ==========================================
FILE_DATASET = 'datasetpariwisata_jawa_makanan.csv'

@st.cache_resource
def load_data():
    try:
        df = pd.read_csv(FILE_DATASET, sep=';')
        df['makanan_khas'] = df['makanan_khas'].fillna('Kuliner Lokal')
        df['search_content'] = (df['place_name'].astype(str) + " " + df['city'].astype(str)).str.lower()
        tfidf = TfidfVectorizer()
        matrix = tfidf.fit_transform(df['search_content'].fillna(''))
        return df, tfidf, matrix
    except:
        return None, None, None

df, tfidf, tfidf_matrix = load_data()

if df is None:
    st.error(f"❌ File '{FILE_DATASET}' tidak ditemukan!")
    st.stop()

# ==========================================
# 🤖 LOGIKA AI
# ==========================================
def chat_with_gemini(user_text):
    # Context Retrieval
    vec = tfidf.transform([user_text.lower()])
    sim = cosine_similarity(vec, tfidf_matrix).flatten()
    top_idx = sim.argsort()[-3:][::-1]
    
    context_info = ""
    if sim[top_idx[0]] > 0.15:
        context_info = "Data Wisata:\n"
        for i in top_idx:
            row = df.iloc[i]
            context_info += f"- {row['place_name']} ({row['city']}), Kuliner: {row['makanan_khas']}\n"

    try:
        # PENTING: Pakai model hasil deteksi otomatis tadi
        # Hapus prefix 'models/' jika ada, karena library kadang nambahin sendiri
        clean_model_name = ACTIVE_MODEL.replace("models/", "")
        model = genai.GenerativeModel(clean_model_name)
        
        prompt = f"""
        Kamu 'Konco Plesir'. Jawab santai.
        User: {user_text}
        {context_info}
        """
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"⚠️ Error Chat: {str(e)}"

# ==========================================
# 💬 TAMPILAN
# ==========================================
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if user_input := st.chat_input("Mau jalan-jalan ke mana?"):
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner(f"Nanya ke {ACTIVE_MODEL}..."):
            balasan = chat_with_gemini(user_input)
            st.markdown(balasan)
    
    st.session_state.messages.append({"role": "assistant", "content": balasan})
