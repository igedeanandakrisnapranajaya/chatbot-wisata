import streamlit as st
import pandas as pd
import google.generativeai as genai
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import time

# ==========================================
# 1. KONFIGURASI HALAMAN (Wajib Paling Atas)
# ==========================================
st.set_page_config(
    page_title="Konco Plesir",
    page_icon="✈️",
    layout="centered", # Bisa ganti 'wide' kalau mau lebar
    initial_sidebar_state="expanded"
)

# Custom CSS biar tampilan lebih bersih
st.markdown("""
<style>
    .stChatFloatingInputContainer {bottom: 20px;}
    h1 {color: #FF4B4B;} /* Warna judul merah streamlit */
    .stMarkdown {font-size: 1.1rem;}
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. SETUP API & MODEL
# ==========================================
ACTIVE_MODEL = "gemini-1.5-flash" 

try:
    api_key = st.secrets["GEMINI_API_KEY"]
    genai.configure(api_key=api_key)
    
    # Auto-detect model
    valid_models = [m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
    if valid_models:
        ACTIVE_MODEL = valid_models[0]
except Exception as e:
    st.error(f"⚠️ Error API Key: {e}")
    st.stop()

# ==========================================
# 3. SIDEBAR (MENU KIRI)
# ==========================================
with st.sidebar:
    st.title("🧳 Konco Plesir")
    st.markdown("Asisten wisata pintarmu untuk keliling Pulau Jawa!")
    st.divider()
    
    # Tombol Reset Chat
    if st.button("🗑️ Hapus Chat", type="primary"):
        st.session_state.messages = []
        st.rerun()
        
    st.info(f"🤖 Model: {ACTIVE_MODEL.replace('models/', '')}")
    st.caption("© 2025 Project Pariwisata")

# ==========================================
# 4. LOAD DATASET (Logika Pencarian Cerdas)
# ==========================================
FILE_DATASET = 'datasetpariwisata_jawa_makanan.csv'

@st.cache_resource
def load_data():
    try:
        df = pd.read_csv(FILE_DATASET, sep=';')
        df['makanan_khas'] = df['makanan_khas'].fillna('Kuliner Lokal')
        
        # Gabungkan semua kolom penting
        df['search_content'] = (
            df['place_name'].astype(str) + " " + 
            df['city'].astype(str) + " " + 
            df['province'].astype(str) + " " + 
            df['makanan_khas'].astype(str)
        ).str.lower()
        
        tfidf = TfidfVectorizer()
        matrix = tfidf.fit_transform(df['search_content'].fillna(''))
        return df, tfidf, matrix
    except:
        return None, None, None

df, tfidf, tfidf_matrix = load_data()

if df is None:
    st.error("❌ Database wisata tidak ditemukan!")
    st.stop()

# ==========================================
# 5. HEADER UTAMA
# ==========================================
st.title("✈️ Halo, Mau Kemana?")
st.markdown("Tanya rekomendasi wisata, kuliner, atau oleh-oleh di Jawa. Santai aja nanyanya!")
st.divider()

# ==========================================
# 6. ENGINE CHATBOT
# ==========================================
def chat_with_gemini(user_text):
    # 1. Cari Data di CSV
    vec = tfidf.transform([user_text.lower()])
    sim = cosine_similarity(vec, tfidf_matrix).flatten()
    top_idx = sim.argsort()[-5:][::-1] # Ambil 5 data teratas
    
    context_info = ""
    # Threshold 0.1 agar lebih sensitif menangkap data
    if sim[top_idx[0]] > 0.1:
        context_info = "Data Database:\n"
        for i in top_idx:
            row = df.iloc[i]
            context_info += f"- {row['place_name']} ({row['city']}, {row['province']}). Kuliner: {row['makanan_khas']}\n"

    # 2. Kirim ke AI
    try:
        clean_model = ACTIVE_MODEL.replace("models/", "")
        model = genai.GenerativeModel(clean_model)
        
        prompt = f"""
        Peran: Kamu adalah 'Konco Plesir', tour guide lokal yang asik, ramah, dan gaul.
        Data Tersedia: 
        {context_info}
        
        Pertanyaan User: {user_text}
        
        Instruksi:
        - Jawab berdasarkan 'Data Tersedia' dulu.
        - Kalau data ada, jelaskan dengan menarik seolah kamu pernah ke sana.
        - Kalau data tidak ada, jawab pakai pengetahuanmu sendiri tapi beri tahu kalau itu rekomendasi umum.
        - Gunakan emoji yang relevan 🏖️🍜.
        - Bahasa santai tapi sopan.
        """
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Waduh, koneksi putus nih. Error: {e}"

# ==========================================
# 7. INTERFACE CHAT
# ==========================================
if "messages" not in st.session_state:
    # Pesan pembuka otomatis
    st.session_state.messages = [
        {"role": "assistant", "content": "Halo Bestie! Mau cari wisata alam atau kulineran nih? 🎒"}
    ]

# Tampilkan Chat
for msg in st.session_state.messages:
    # Set Avatar: Bot pake robot/pesawat, User pake orang
    icon = "🤖" if msg["role"] == "assistant" else "🧑‍💻"
    with st.chat_message(msg["role"], avatar=icon):
        st.markdown(msg["content"])

# Input User
if user_input := st.chat_input("Ketik pertanyaanmu di sini..."):
    # Tampilkan pesan user
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user", avatar="🧑‍💻"):
        st.markdown(user_input)

    # Animasi loading (biar kelihatan mikir)
    with st.chat_message("assistant", avatar="🤖"):
        message_placeholder = st.empty()
        with st.spinner("Sedang mencari info terbaik..."):
            balasan = chat_with_gemini(user_input)
            
            # Efek mengetik (Typing effect)
            full_response = ""
            for chunk in balasan.split():
                full_response += chunk + " "
                time.sleep(0.05) # Kecepatan ngetik
                message_placeholder.markdown(full_response + "▌")
            message_placeholder.markdown(full_response)
    
    st.session_state.messages.append({"role": "assistant", "content": balasan})
