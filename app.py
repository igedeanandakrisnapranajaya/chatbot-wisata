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
# ==========================================
# 7. LOGIKA CHAT (DENGAN MEMORI)
# ==========================================
def chat_with_gemini(user_text, history_messages):
    # ... (Bagian 1: History & Bagian 2: CSV Search biarkan sama seperti sebelumnya) ...
    # Langsung copy dari baris "vec = ..." sampai "if sim[top_idx[0]]..." itu sama.
    # Kita fokus ganti bagian PROMPT di bawah ini:
    
    # --- MULAI COPY DARI SINI ---
    # 1. SIAPKAN MEMORI 
    history_str = "\nRiwayat Percakapan:\n"
    for msg in history_messages[-4:]: 
        role = "User" if msg["role"] == "user" else "Bot"
        history_str += f"{role}: {msg['content']}\n"

    # 2. CARI DATA DI CSV 
    vec = tfidf.transform([user_text.lower()])
    sim = cosine_similarity(vec, tfidf_matrix).flatten()
    top_idx = sim.argsort()[-5:][::-1]
    
    context_info = ""
    if sim[top_idx[0]] > 0.05:
        context_info = "Data Database Wisata (Verifikasi Lokasi):\n"
        for i in top_idx:
            row = df.iloc[i]
            context_info += f"- {row['place_name']} di {row['city']}, {row['province']}. Kuliner: {row['makanan_khas']}\n"
    else:
        context_info = "Data Database: Tidak ada. Gunakan Riwayat & Pengetahuan Umum."

    try:
        clean_model = ACTIVE_MODEL.replace("models/", "")
        model = genai.GenerativeModel(clean_model)
        
        prompt = f"""
        Peran: Kamu adalah 'Konco Plesir'.
        
        {history_str}
        {context_info}
        Pertanyaan: {user_text}
        
        INSTRUKSI UTAMA:
        1. Jawab ramah dan solutif.
        2. Jika tanya Estimasi Harga/Hotel/Transport -> Gunakan PENGETAHUAN UMUM (General Knowledge) untuk memberi kisaran harga (Rupiah).
        
        ATURAN FORMATTING (WAJIB DIPATUHI AGAR RAPI):
        - JANGAN menulis paragraf panjang. Pecah menjadi Poin-Poin.
        - Gunakan format **Markdown**.
        - Gunakan **Bullet Points** (-) untuk daftar nama tempat atau rincian biaya.
        - Gunakan **Bold** untuk menonjolkan Harga (Rp...) atau Nama Tempat.
        - WAJIB beri jarak antar baris (Enter) supaya tidak menumpuk.
        
        Contoh Format Output yang Benar:
        "Wah, ada beberapa opsi asik nih:
        
        - **Hotel A**: Sekitar 300rb (Dekat alun-alun)
        - **Hotel B**: Sekitar 500rb (Ada kolam renang)
        
        Untuk makanannya:
        - **Nasi Liwet**: 20rb-an per porsi."
        """
        
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Waduh, error nih: {e}"

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

# ... (Bagian atas kode input user sama)

if user_input := st.chat_input("Ketik pertanyaanmu di sini..."):
    # Tampilkan user input
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user", avatar="😎"):
        st.markdown(user_input)

    # Tampilkan respon bot
    with st.chat_message("assistant", avatar="🤖"):
        message_placeholder = st.empty()
        with st.spinner("Sedang mencari info terbaik..."):
            
            # Panggil Fungsi Chat
            balasan = chat_with_gemini(user_input, st.session_state.messages)
            
            # --- PERBAIKAN LOGIKA TYPING EFFECT ---
            full_response = ""
            
            # Teknik 1: Tampilkan teks langsung (Tanpa efek ngetik) -> Paling aman buat format
            # message_placeholder.markdown(balasan) 
            
            # Teknik 2: Efek ngetik yang AMAN untuk Markdown (Per Karakter atau Per Baris)
            # Kita pakai split('\n') supaya ENTER-nya tidak hilang.
            lines = balasan.split('\n') 
            for line in lines:
                full_response += line + "\n" # Tambahkan enter manual
                time.sleep(0.05) # Atur kecepatan
                # Simbol kursor
                message_placeholder.markdown(full_response + "▌")
                
            # Tampilkan hasil akhir bersih tanpa kursor
            message_placeholder.markdown(full_response)
    
    # Simpan ke session state
    st.session_state.messages.append({"role": "assistant", "content": balasan})









