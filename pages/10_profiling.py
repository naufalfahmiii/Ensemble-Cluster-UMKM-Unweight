import streamlit as st
import pandas as pd

from modules.utils import step_header, require_state, download_button
from modules.profiling import (
    build_hasil_cluster, profil_ukuran, profil_numerik, profil_kategorikal,
    build_tabel6, build_rekomendasi_template,
)

step_header("🧬", "Profiling & Recommendation",
            "Melihat 'siapa' isi tiap cluster: ukuran, karakteristik numerik, dan kecenderungan "
            "kategorikalnya — termasuk kolom identitas yang tadi disisihkan di awal.")

require_state("evaluation_final", "best_k", message="K final belum ditentukan.",
               back_hint="Buka halaman **Final Validation** terlebih dahulu.")

best_k = st.session_state.best_k
numeric_cols = st.session_state.numeric_cols
categorical_cols = st.session_state.categorical_cols

if st.session_state.hasil_cluster is None:
    hasil_cluster = build_hasil_cluster(
        st.session_state.df_identity, st.session_state.df_typeaware,
        numeric_cols, categorical_cols, st.session_state.ensemble_result, best_k,
    )
    st.session_state.hasil_cluster = hasil_cluster

    p_ukuran = profil_ukuran(hasil_cluster)
    p_numerik = profil_numerik(hasil_cluster, numeric_cols)
    p_kategorikal = profil_kategorikal(hasil_cluster, categorical_cols, st.session_state.cat_encoding)
    tabel6_profil = build_tabel6(p_ukuran, p_numerik, p_kategorikal)

    st.session_state.tabel6_profil = tabel6_profil
    st.session_state.rekomendasi_template = build_rekomendasi_template(tabel6_profil)

st.markdown(f"### 📌 Data Lengkap + Label Cluster Final (K={best_k})")
st.caption("Termasuk kolom identitas yang tidak dipakai sebagai fitur (ketentuan #1) — tetap utuh di sini.")
st.dataframe(st.session_state.hasil_cluster.head(20), use_container_width=True)
download_button(st.session_state.hasil_cluster, "hasil_cluster_lengkap.xlsx")

st.divider()
st.markdown("### 🪪 Profil Ringkas per Cluster")
st.dataframe(st.session_state.tabel6_profil, use_container_width=True)
download_button(st.session_state.tabel6_profil.reset_index(), "tabel_profil_cluster.xlsx")

st.divider()
st.markdown("### 📝 Template Rekomendasi (isi manual sesuai konteks program Anda)")
st.caption("Kolom deskriptif dikosongkan secara sengaja — memerlukan interpretasi domain/pendampingan lapangan, "
            "bukan hasil algoritmik. Unduh lalu lengkapi sesuai kebutuhan.")
st.dataframe(st.session_state.rekomendasi_template, use_container_width=True, hide_index=True)
download_button(st.session_state.rekomendasi_template, "template_rekomendasi.xlsx")

st.divider()
st.markdown("### 🔍 Jelajahi Anggota Tiap Cluster")
cluster_options = sorted(st.session_state.hasil_cluster["Cluster"].unique())
pick = st.selectbox("Pilih cluster:", cluster_options, format_func=lambda c: f"Cluster C{c + 1}")
subset = st.session_state.hasil_cluster[st.session_state.hasil_cluster["Cluster"] == pick]
st.dataframe(subset, use_container_width=True)
download_button(subset, f"anggota_cluster_C{pick + 1}.xlsx")

st.success("🎉 Seluruh alur ensemble clustering (unweighted) selesai — dari data mentah hingga profil "
            "dan rekomendasi per cluster.")
