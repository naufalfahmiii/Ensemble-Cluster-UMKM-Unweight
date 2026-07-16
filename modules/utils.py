"""
Kumpulan fungsi bantu lintas halaman: tombol unduh Excel, penjaga alur
antar-halaman (memastikan tahap sebelumnya sudah dikerjakan), dan
sedikit pemanis tampilan.
"""
import io
import pandas as pd
import streamlit as st


def df_to_excel_bytes(df: pd.DataFrame, sheet_name: str = "Sheet1") -> bytes:
    """Mengubah DataFrame menjadi bytes .xlsx siap-unduh (tanpa menulis ke disk)."""
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="xlsxwriter") as writer:
        df.to_excel(writer, index=False, sheet_name=sheet_name[:31])
    return buffer.getvalue()


def download_button(df: pd.DataFrame, filename: str, label: str = None, key: str = None):
    """Tombol unduh seragam untuk setiap output tahap (memenuhi ketentuan #6)."""
    if df is None or len(df) == 0:
        st.info("Belum ada data untuk diunduh pada tahap ini.")
        return
    label = label or f"⬇️ Unduh {filename}"
    st.download_button(
        label=label,
        data=df_to_excel_bytes(df, sheet_name=filename.replace(".xlsx", "")),
        file_name=filename,
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        key=key or f"dl_{filename}",
        use_container_width=True,
    )


def require_state(*keys, message=None, back_hint=None):
    """
    Menghentikan render halaman jika salah satu 'keys' belum tersedia di
    session_state — mencegah user meloncat ke halaman lanjutan sebelum
    tahap sebelumnya selesai dijalankan.
    """
    missing = [k for k in keys if st.session_state.get(k) is None]
    if missing:
        msg = message or "Tahap sebelumnya belum lengkap."
        st.warning(f"⚠️ {msg}")
        if back_hint:
            st.info(back_hint)
        st.stop()
        
def require_feature_mapping():
    """
    Menghentikan render halaman jika belum ada kolom yang dipetakan sebagai
    Numerik atau Kategorikal — mencegah halaman turunan (Preprocessing, dst.)
    memproses data dengan feature set kosong akibat user lompat halaman
    lewat sidebar sebelum menyelesaikan pemetaan kolom di Setup Data.
    """
    numeric_cols = st.session_state.get("numeric_cols") or []
    categorical_cols = st.session_state.get("categorical_cols") or []
    if not numeric_cols and not categorical_cols:
        st.warning("⚠️ Belum ada kolom yang dipetakan sebagai Numerik atau Kategorikal.")
        st.info(
            "Buka halaman **Pintu Masuk Data**, pilih kolom-kolom fitur di kotak "
            "*Kolom numerik* / *Kolom kategorikal*, baru kembali ke halaman ini."
        )
        st.stop()

def init_state():
    """Menyiapkan semua key session_state yang dipakai lintas halaman."""
    defaults = {
        # Bagian 1 — data & pemetaan kolom
        "raw_df": None,
        "raw_filename": None,
        "identity_cols": [],
        "numeric_cols": [],
        "categorical_cols": [],
        "cat_encoding": {},
        # Bagian 2 — preprocessing
        "df_clean": None,
        "df_identity": None,
        "df_typeaware": None,
        "df_baseline": None,
        "dedup_info": None,
        # Parameter global
        "k_min": 2,
        "k_max": 10,
        "random_state": 42,
        # Bagian 3 — base clustering
        "kmeans_result": None,
        "evaluation_kmeans": None,
        "X_kmeans": None,
        "feature_order_baseline": None,
        "kproto_result": None,
        "evaluation_kproto": None,
        "gamma_kproto": None,
        "kmedoids_result": None,
        "evaluation_kmedoids": None,
        "gower_matrix": None,
        "n_init_gower": 10,
        # Bagian 4 — validasi per metode
        "ranked_kmeans": None,
        "ranked_kproto": None,
        "ranked_kmedoids": None,
        "table2_summary": None,
        # Bagian 5 — alignment
        "aligned_result": None,
        # Bagian 6 — ensemble
        "X_eval_combined": None,
        "gower_matrix_eval": None,
        "ensemble_result": None,
        "evaluation_ensemble": None,
        # Bagian 7 — validasi konsensus & stabilitas
        "stability_df": None,
        "evaluation_final": None,
        "best_k": None,
        "n_runs_stability": 30,
        # Bagian 8 — sensitivitas
        "sensitivity_result": None,
        # Bagian 9/10 — profil
        "hasil_cluster": None,
        "tabel6_profil": None,
        "rekomendasi_template": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def step_header(emoji: str, title: str, subtitle: str = ""):
    st.markdown(f"## {emoji} {title}")
    if subtitle:
        st.caption(subtitle)
    st.divider()


def reset_downstream(from_step: int):
    """
    Menghapus hasil tahap-tahap SETELAH `from_step` ketika user mengubah
    parameter di tahap awal (mis. ganti kolom fitur) — supaya tidak ada
    hasil basi yang tertinggal di session_state.
    """
    groups = {
        1: ["df_clean", "df_identity", "df_typeaware", "df_baseline", "dedup_info",
            "feature_order_baseline"],
        2: ["kmeans_result", "evaluation_kmeans", "X_kmeans",
            "kproto_result", "evaluation_kproto", "gamma_kproto",
            "kmedoids_result", "evaluation_kmedoids", "gower_matrix"],
        3: ["ranked_kmeans", "ranked_kproto", "ranked_kmedoids", "table2_summary"],
        4: ["aligned_result"],
        5: ["X_eval_combined", "gower_matrix_eval", "ensemble_result", "evaluation_ensemble"],
        6: ["stability_df", "evaluation_final", "best_k"],
        7: ["sensitivity_result"],
        8: ["hasil_cluster", "tabel6_profil", "rekomendasi_template"],
    }
    for step in range(from_step, 9):
        for key in groups.get(step, []):
            st.session_state[key] = None
