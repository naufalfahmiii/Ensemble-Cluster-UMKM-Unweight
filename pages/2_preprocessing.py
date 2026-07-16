import streamlit as st

from modules.utils import step_header, require_state, require_feature_mapping, download_button, reset_downstream
from modules.preprocessing import drop_duplicates, split_identity, build_typeaware, build_baseline
from modules.visualization import numeric_distribution, categorical_distribution

step_header("🧹", "Dapur Data",
            "Data mentah dibersihkan dan diramu menjadi dua representasi: satu untuk K-Prototypes & "
            "Gower K-Medoids (skala asli), satu lagi matriks siap-angka untuk K-Means.")

require_state("raw_df", message="Belum ada data yang diunggah.",
               back_hint="Buka halaman **Setup Data** terlebih dahulu.")
require_feature_mapping()

df_raw = st.session_state.raw_df
numeric_cols = st.session_state.numeric_cols
categorical_cols = st.session_state.categorical_cols
identity_cols = st.session_state.identity_cols
cat_encoding = st.session_state.cat_encoding

st.markdown("### 🧾 Pengecekan Duplikat")
st.caption(
    "Seperti pada notebook (kolom `No.` dikecualikan karena hanya nomor urut entri), kolom "
    "identitas yang bersifat unik per baris (mis. ID, nomor urut, nama) sebaiknya **tidak** ikut "
    "dibandingkan saat mencari baris duplikat — jika ikut dibandingkan, baris yang sebenarnya "
    "kembar pada seluruh fitur tidak akan terdeteksi karena ID/nomornya pasti berbeda."
)
dedup_exclude_default = [
    c for c in identity_cols
    if df_raw[c].nunique(dropna=False) == len(df_raw)
] or identity_cols
dedup_exclude_cols = st.multiselect(
    "Kolom yang dikecualikan dari pengecekan duplikat",
    options=identity_cols,
    default=dedup_exclude_default,
    help="Default: kolom identitas yang nilainya unik di setiap baris (kandidat ID/nomor urut).",
)

run = st.button("🚀 Jalankan Pembersihan & Preprocessing", type="primary", use_container_width=True)

if run or st.session_state.df_baseline is not None:
    if run:
        with st.spinner("Membersihkan duplikat dan menyusun representasi data..."):
            df_clean, dedup_info = drop_duplicates(df_raw, exclude_cols=dedup_exclude_cols)
            df_identity = split_identity(df_clean, identity_cols)
            df_typeaware = build_typeaware(df_clean, numeric_cols, categorical_cols)
            df_baseline, feature_order = build_baseline(df_typeaware, numeric_cols, categorical_cols, cat_encoding)

        st.session_state.df_clean = df_clean
        st.session_state.dedup_info = dedup_info
        st.session_state.df_identity = df_identity
        st.session_state.df_typeaware = df_typeaware
        st.session_state.df_baseline = df_baseline
        st.session_state.feature_order_baseline = feature_order
        reset_downstream(2)

    dedup_info = st.session_state.dedup_info
    m1, m2, m3 = st.columns(3)
    m1.metric("Baris sebelum dedup", dedup_info["n_before"])
    m2.metric("Baris sesudah dedup", dedup_info["n_after"])
    m3.metric("Baris duplikat dibuang", dedup_info["n_removed"])

    st.divider()
    tab1, tab2, tab3 = st.tabs(["🔬 Representasi Type-Aware", "📐 Representasi Baseline (K-Means)", "🗂️ Data Identitas"])

    with tab1:
        st.caption("Dipakai oleh K-Prototypes & Gower K-Medoids — skala nilai asli, kategori dirapikan (title case).")
        st.dataframe(st.session_state.df_typeaware.head(20), use_container_width=True)
        download_button(st.session_state.df_typeaware, "hasil_preprocessing_typeaware.xlsx")

    with tab2:
        st.caption("Dipakai oleh K-Means — numerik di-IQR-clip lalu MinMax [0,1], kategorikal di-encode sesuai pilihan Anda.")
        st.dataframe(st.session_state.df_baseline.head(20), use_container_width=True)
        download_button(st.session_state.df_baseline, "hasil_preprocessing_baseline.xlsx")

    with tab3:
        st.caption("Kolom yang tidak dipakai sebagai fitur — tetap disimpan untuk profiling di tahap akhir.")
        if st.session_state.df_identity.shape[1] == 0:
            st.info("Tidak ada kolom identitas (semua kolom dipakai sebagai fitur).")
        else:
            st.dataframe(st.session_state.df_identity.head(20), use_container_width=True)
            download_button(st.session_state.df_identity, "data_identitas.xlsx")

    st.divider()
    st.markdown("### 👁️ Sekilas Sebaran Data (agar mudah dipahami)")
    viz_cols = st.columns(2)
    if numeric_cols:
        with viz_cols[0]:
            pick_num = st.selectbox("Lihat sebaran kolom numerik:", numeric_cols, key="prep_num_pick")
            st.plotly_chart(numeric_distribution(st.session_state.df_typeaware, pick_num), use_container_width=True)
    if categorical_cols:
        with viz_cols[1]:
            pick_cat = st.selectbox("Lihat sebaran kolom kategorikal:", categorical_cols, key="prep_cat_pick")
            st.plotly_chart(categorical_distribution(st.session_state.df_typeaware, pick_cat), use_container_width=True)

    st.success("✅ Preprocessing selesai. Lanjut ke **Base Clustering** di sidebar.")
else:
    st.info("Klik tombol di atas untuk memulai pembersihan & penyusunan data.")