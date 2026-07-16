import streamlit as st

from modules.utils import step_header, require_state, download_button, reset_downstream
from modules.alignment import align_all, hungarian_correspondence_table

step_header("🔗", "Label Alignment",
            "Label cluster tiap metode bersifat arbitrer (mis. 'Cluster 0' di K-Means belum tentu sama "
            "artinya dengan 'Cluster 0' di K-Prototypes). Algoritma Hungarian menyelaraskan penomoran ini "
            "terhadap K-Means sebagai acuan, sebelum voting dilakukan.")

require_state("evaluation_kmeans", "evaluation_kproto", "evaluation_kmedoids",
               message="Base clustering belum dijalankan.",
               back_hint="Buka halaman **Base Clustering** terlebih dahulu.")

k_range = range(st.session_state.k_min, st.session_state.k_max + 1)

if st.session_state.aligned_result is None:
    with st.spinner("Menyelaraskan label antar metode untuk setiap nilai K..."):
        st.session_state.aligned_result = align_all(
            st.session_state.kmeans_result, st.session_state.kproto_result, st.session_state.kmedoids_result,
            st.session_state.df_identity, k_range,
        )
    reset_downstream(5)

st.dataframe(st.session_state.aligned_result.head(20), use_container_width=True)
download_button(st.session_state.aligned_result, "aligned_labels_all.xlsx")

st.divider()
st.markdown("### 🔍 Contoh Korespondensi Label")
st.caption("Menunjukkan bagaimana label asli tiap metode dipetakan ke label acuan K-Means, pada satu nilai K contoh.")

k_list = list(k_range)
k_example = st.selectbox("Pilih K untuk melihat contoh korespondensi:", k_list,
                            index=min(2, len(k_list) - 1))

km_labels = st.session_state.kmeans_result[f"Cluster_K{k_example}"].to_numpy()
kp_labels = st.session_state.kproto_result[f"Cluster_K{k_example}"].to_numpy()
gm_labels = st.session_state.kmedoids_result[f"Cluster_K{k_example}"].to_numpy()

import pandas as pd
table3 = pd.concat([
    hungarian_correspondence_table(km_labels, kp_labels, "K-Prototypes"),
    hungarian_correspondence_table(km_labels, gm_labels, "Gower K-Medoids"),
], ignore_index=True)
st.dataframe(table3, use_container_width=True, hide_index=True)

st.success("✅ Penyelarasan label selesai. Lanjut ke **Ensemble Voting** di sidebar.")
