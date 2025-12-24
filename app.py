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
# 🔑 SETUP API GOOGLE (Cara Resmi)
# ==========================================
try:
    # Mengambil key dari Secrets
    api_key = st.secrets["GEMINI_API_KEY"]
    genai.configure(api_key=api_key)
except Exception as e:
    st.error("⚠️ Kunci API belum diatur di Streamlit Secrets.")
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
        # Buat kolom pencarian
        df['search_content'] = (df['place_name'].astype(str) + " " + df['city'].astype(str)).str.lower()
        
        tfidf = TfidfVectorizer()
        matrix = tfidf.fit_transform(df['search_content'].fillna(''))
        return df, tfidf, matrix
    except Exception as e:
        return None, None, None

df, tfidf, tfidf_matrix = load_data()

if df is None:
    st.error(f"❌ File '{FILE_DATASET}' tidak ditemukan di GitHub!")
    st.stop()

# ==========================================
# 🤖 LOGIKA AI & REKOMENDASI
# ==========================================
def chat_with_gemini(user_text):
    # Cari rekomendasi dulu dari CSV
    vec = tfidf.transform([user_text.lower()])
    sim = cosine_similarity(vec, tfidf_matrix).flatten()
    top_idx = sim.argsort()[-3:][::-1] # Ambil top 3
    
    context_info = ""
    if sim[top_idx[0]] > 0.15: # Kalau kemiripan > 15%
        context_info = "Data Wisata Terkait:\n"
        for i in top_idx:
            row = df.iloc[i]
            context_info += f"- Nama: {row['place_name']}, Lokasi: {row['city']}, Kuliner: {row['makanan_khas']}\n"

    # Kirim ke Gemini pakai Library Resmi
    try:
        # Menggunakan model flash yang cepat dan gratis
        model = genai.GenerativeModel('gemini-1.5-flash')
        
        prompt = f"""
        Peran: Kamu adalah 'Konco Plesir', asisten wisata yang asik dan gaul.
        Pertanyaan User: {user_text}
        
        {context_info}
        
        Instruksi:
        1. Jawab pertanyaan user berdasarkan data di atas jika ada.
        2. Kalau tidak ada data, jawab pakai pengetahuan umum tapi tetap sopan.
        3. Gaya bahasa santai.
        """
        
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"⚠️ Maaf, sistem lagi sibuk. Error: {str(e)}"

# ==========================================
# 💬 TAMPILAN CHAT
# ==========================================
if "messages" not in st.session_state:
    st.session_state.messages = []

# Tampilkan history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Input user
if user_input := st.chat_input("Mau jalan-jalan ke mana?"):
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Proses AI
    with st.chat_message("assistant"):
        with st.spinner("Mikiri sik..."):
            balasan = chat_with_gemini(user_input)
            st.markdown(balasan)
    
    st.session_state.messages.append({"role": "assistant", "content": balasan})
