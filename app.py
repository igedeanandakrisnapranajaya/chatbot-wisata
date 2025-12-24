import streamlit as st
import pandas as pd
import google.generativeai as genai
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import time
import os # Import os untuk cek keberadaan file

# ==========================================
# KONFIGURASI NAMA FILE GAMBAR
# ==========================================
# GANTI tulisan 'logo.png' di bawah ini jika nama filemu berbeda!
NAMA_FILE_LOGO = "logo.png"

# Cek apakah file gambar benar-benar ada di GitHub
gambar_tersedia = os.path.exists(NAMA_FILE_LOGO)

# ==========================================
# 1. KONFIGURASI HALAMAN & FAVICON
# ==========================================
st.set_page_config(
    page_title="Konco Plesir",
    # Jika gambar ada, pakai jadi icon tab browser. Jika tidak, pakai emoji.
    page_icon=NAMA_FILE_LOGO if gambar_tersedia else "🏖️",
    layout="centered",
    initial_sidebar_state="expanded"
)

# ==========================================
# 2. DEKORASI CSS (Tema Cerah)
# ==========================================
st.markdown("""
<style>
    /* Mengubah warna background header chat */
    .stAppHeader {
        background-color: #FFFFFF;
        opacity: 0.9;
    }
    
    /* Membuat Judul Gradasi Warna */
    .gradient-text {
        background: -webkit-linear-gradient(left, #0072ff, #00c6ff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: bold;
        font-size: 3em;
        padding-bottom: 10px;
    }
    
    /* Mengubah tampilan chat bubble user */
    .stChatMessage[data-testid="stChatMessage"]:nth-child(odd) {
        background-color: #E3F2FD;
        border-radius: 10px;
    }
    
    /* Agar gambar logo di tengah tidak terlalu nempel ke atas */
    .main-logo {
        margin-top: -50px;
        margin-bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 3. SETUP API & MODEL
# ==========================================
ACTIVE_MODEL = "gemini-2.5-pro" 
try:
    api_key = st.secrets["GEMINI_API_KEY"]
    genai.configure(api_key=api_key)
    valid_models = [m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
    if valid_models: ACTIVE_MODEL = valid_models[0]
except Exception as e:
    st.error(f"⚠️ Error API Key: {e}")
    st.stop()

# ==========================================
# 4. LOAD DATASET
# ==========================================
FILE_DATASET = 'datasetpariwisata_jawa_makanan.csv'
@st.cache_resource
def load_data():
    try:
        df = pd.read_csv(FILE_DATASET, sep=';')
        df['makanan_khas'] = df['makanan_khas'].fillna('Kuliner Lokal')
        df['search_content'] = (
            df['place_name'].astype(str) + " " + df['city'].astype(str) + " " + 
            df['province'].astype(str) + " " + df['makanan_khas'].astype(str)
        ).str.lower()
        tfidf = TfidfVectorizer()
        matrix = tfidf.fit_transform(df['search_content'].fillna(''))
        return df, tfidf, matrix
    except: return None, None, None

df, tfidf, tfidf_matrix = load_data()
if df is None:
    st.error("❌ Database wisata tidak ditemukan!"); st.stop()

# ==========================================
# 5. SIDEBAR DENGAN LOGO
# ==========================================
with st.sidebar:
    # --- LOGO DI SIDEBAR ---
    if gambar_tersedia:
        # Tampilkan logo. width=150 biar pas ukurannya.
        st.image(NAMA_FILE_LOGO, width=150)
    else:
        # Fallback kalau lupa upload gambar
        st.header("🧳")
        
    st.title("Konco Plesir")
    st.caption("Teman jalan-jalan keliling Jawa!")
    st.markdown("---")
    
    if st.button("🔄 Mulai Chat Baru", type="secondary"):
        st.session_state.messages = []
        st.rerun()
        
    st.markdown("---")
    st.info("💡 **Tips:** Coba tanya 'Makanan khas Jogja' atau 'Pantai di Malang'.")

# ==========================================
# 6. HALAMAN UTAMA DENGAN LOGO
# ==========================================
# --- LOGO DI TENGAH (OPSIONAL) ---
# Jika ingin ada logo besar di atas judul, aktifkan kode ini:
# if gambar_tersedia:
#    col1, col2, col3 = st.columns([1,2,1])
#    with col2: # Taruh di kolom tengah biar rapi
#        st.image(NAMA_FILE_LOGO, use_column_width=True, className="main-logo")

# Judul dengan efek gradasi
st.markdown('<p class="gradient-text">✈️ Konco Plesir</p>', unsafe_allow_html=True)
st.markdown("##### Asisten wisata AI yang siap nemenin liburanmu!")

# ==========================================
# 7. LOGIKA CHAT
# ==========================================
def chat_with_gemini(user_text):
    # 1. Cari Data di CSV (RAG)
    vec = tfidf.transform([user_text.lower()])
    sim = cosine_similarity(vec, tfidf_matrix).flatten()
    top_idx = sim.argsort()[-5:][::-1]
    
    context_info = ""
    if sim[top_idx[0]] > 0.1:
        context_info = "Data Database Wisata:\n"
        for i in top_idx:
            row = df.iloc[i]
            context_info += f"- {row['place_name']} di {row['city']}, {row['province']}. Kuliner Khas: {row['makanan_khas']}\n"

    # 2. Kirim ke AI dengan Instruksi yang LEBIH LENGKAP
    try:
        clean_model = ACTIVE_MODEL.replace("models/", "")
        model = genai.GenerativeModel(clean_model)
        
        prompt = f"""
        Peran: Kamu adalah 'Konco Plesir', travel consultant profesional yang detail tapi bahasanya santai & gaul.
        
        Data Database (Gunakan sebagai referensi utama lokasi): 
        {context_info}
        
        Pertanyaan User: {user_text}
        
        INSTRUKSI WAJIB (PENTING):
        1. Jawab pertanyaan user dengan ramah.
        2. Jika user bertanya rekomendasi wisata, JANGAN HANYA SEBUT NAMA.
        3. WAJIB BERIKAN 'ESTIMASI BIAYA' (Gunakan pengetahuan umummu/internet knowledge):
           - 🎟️ Tiket Masuk (Kira-kira berapa Rupiah)
           - 🍜 Harga Makanan di sana (Range harga)
           - 🚗 Transportasi (Opsi ke sana naik apa)
           - 🏨 Penginapan/Hotel terdekat (Sebutkan nama daerah atau kisaran harga hotel bintang 2-3)
        4. Berikan disclaimer bahwa harga bisa berubah sewaktu-waktu.
        5. Gunakan format List/Poin biar enak dibaca.
        
        Contoh gaya bicara:
        "Wah, kalau ke Malioboro wajib coba Gudeg Yu Djum! Tiket masuk gratis kok, cuma bayar parkir ceban (Rp 10.000). Penginapan banyak di sosrowijayan mulai 200rb-an."
        """
        
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Waduh, koneksi putus nih. Error: {e}"

# ==========================================
# 8. INTERFACE CHAT
# ==========================================
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Halo Kak! 👋 Mau liburan ke mana kita hari ini? Aku siap bantu cariin tempat asik!"}
    ]

for msg in st.session_state.messages:
    icon = "🤖" if msg["role"] == "assistant" else "😎"
    with st.chat_message(msg["role"], avatar=icon):
        st.markdown(msg["content"])

if user_input := st.chat_input("Ketik pertanyaanmu di sini..."):
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user", avatar="😎"):
        st.markdown(user_input)

    with st.chat_message("assistant", avatar="🤖"):
        message_placeholder = st.empty()
        with st.spinner("Sedang mencari info terbaik..."):
            balasan = chat_with_gemini(user_input)
            full_response = ""
            for chunk in balasan.split():
                full_response += chunk + " "
                time.sleep(0.05)
                message_placeholder.markdown(full_response + "▌")
            message_placeholder.markdown(full_response)
    
    st.session_state.messages.append({"role": "assistant", "content": balasan})


