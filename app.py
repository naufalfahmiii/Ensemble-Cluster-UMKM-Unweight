import traceback
import streamlit as st
from modules.utils import init_state

st.set_page_config(
    page_title="Ensemble Clustering UMKM",
    page_icon="🧵",
    layout="wide",
    initial_sidebar_state="expanded",
)

init_state()

# --- Navigasi kustom: judul & ikon sidebar sengaja TIDAK mengikuti nama file
# (ketentuan #5) supaya lebih ramah dibaca orang awam yang membuka aplikasi. ---
page_defs = {
    "Mulai di Sini": [
        ("pages/1_upload.py", "Setup Data", "📂", True),
        ("pages/2_preprocessing.py", "Preprocessing", "🧹", False),
    ],
    "Klasterisasi": [
        ("pages/3_base_clustering.py", "Base Clustering", "🧩", False),
        ("pages/4_validation.py", "Multi-Index Validation", "📊", False),
        ("pages/5_alignment.py", "Label Alignment", "🔗", False),
        ("pages/6_ensemble.py", "Ensemble Voting", "🗳️", False),
        ("pages/7_final_validation.py", "Final Validation", "🏁", False),
        ("pages/8_sensitivity.py", "Sensitivity Check (Optional)", "🏁", False),
    ],
    "Hasil Akhir": [
        ("pages/9_visualization.py", "Visualization", "🖼️", False),
        ("pages/10_profiling.py", "Profiling & Recommendation", "🧬", False),
    ],
}
pages = {
    section: [st.Page(path, title=title, icon=icon, default=default) for path, title, icon, default in items]
    for section, items in page_defs.items()
}

nav = st.navigation(pages)

with st.sidebar:
    st.markdown("### 🧵 Ensemble Clustering UMKM")
    st.caption("Jalur Unweighted (Equal-Vote) — K-Means x K-Prototypes x Gower K-Medoids")
    st.divider()

try:
    nav.run()
except Exception as e:
    # Penanganan error ramah pengguna: tidak menampilkan traceback mentah
    # langsung di layar, tapi pesan yang jelas + langkah lanjutan + detail
    # teknis yang bisa dibuka manual bila perlu untuk melapor/debug.
    st.error(
        "😵 **Terjadi kendala saat memproses halaman ini.**\n\n"
        "Ini biasanya terjadi karena: (1) urutan tahap dilompati — coba jalankan ulang "
        "dari tahap sebelumnya secara berurutan lewat sidebar, atau (2) data/kolom yang "
        "dipilih tidak cocok dengan tahap ini (mis. kolom kategorikal berubah setelah "
        "tahap sebelumnya dijalankan).",
        icon="🚨",
    )
    c1, c2 = st.columns(2)
    with c1:
        if st.button("🔄 Muat Ulang Halaman Ini", use_container_width=True):
            st.rerun()
    with c2:
        if st.button("♻️ Reset Seluruh Sesi & Mulai dari Awal", use_container_width=True):
            st.session_state.clear()
            st.rerun()
    with st.expander("🔧 Detail teknis (untuk pelaporan/debug)"):
        st.code(f"{type(e).__name__}: {e}\n\n{traceback.format_exc()}", language="text")
