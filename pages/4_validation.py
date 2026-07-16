import streamlit as st

from modules.utils import step_header, require_state, download_button, reset_downstream
from modules.validation import rank_multi_index, build_table2_summary
from modules.visualization import composite_rank_bar

step_header("📊", "Multi-Index Validation",
            "Tiga kriteria (Silhouette, Davies-Bouldin, Calinski-Harabasz) digabung jadi satu peringkat "
            "komposit untuk mencari K terbaik di masing-masing metode secara mandiri.")

require_state("evaluation_kmeans", "evaluation_kproto", "evaluation_kmedoids",
               message="Base clustering belum dijalankan.",
               back_hint="Buka halaman **Base Clustering** terlebih dahulu.")

if st.session_state.ranked_kmeans is None:
    st.session_state.ranked_kmeans = rank_multi_index(st.session_state.evaluation_kmeans)
    st.session_state.ranked_kproto = rank_multi_index(st.session_state.evaluation_kproto)
    st.session_state.ranked_kmedoids = rank_multi_index(st.session_state.evaluation_kmedoids)
    st.session_state.table2_summary = build_table2_summary(
        st.session_state.evaluation_kmeans, st.session_state.evaluation_kproto, st.session_state.evaluation_kmedoids,
        st.session_state.kmeans_result, st.session_state.kproto_result, st.session_state.kmedoids_result,
    )
    reset_downstream(4)

st.markdown("### 🏅 Ringkasan: K Terbaik di Dalam Tiap Metode")
st.caption("Catatan: ini **bukan** keputusan K final untuk ensemble — hanya gambaran per-metode. "
            "K final ditentukan setelah tahap voting & analisis stabilitas.")
st.dataframe(st.session_state.table2_summary, use_container_width=True, hide_index=True)
download_button(st.session_state.table2_summary, "tabel2_ringkasan_best_within_method.xlsx")

st.divider()
tab1, tab2, tab3 = st.tabs(["K-Means", "K-Prototypes", "Gower K-Medoids"])

for tab, name, df_key in [
    (tab1, "K-Means", "ranked_kmeans"),
    (tab2, "K-Prototypes", "ranked_kproto"),
    (tab3, "Gower K-Medoids", "ranked_kmedoids"),
]:
    with tab:
        ranked_df = st.session_state[df_key]
        best_k_method = int(ranked_df.iloc[0]["K"])
        st.plotly_chart(
            composite_rank_bar(ranked_df, best_k_method, title=f"Peringkat Komposit — {name}"),
            use_container_width=True,
        )
        st.dataframe(ranked_df, use_container_width=True, hide_index=True)
        download_button(ranked_df, f"multi_index_validation_{name.lower().replace(' ', '_').replace('-', '')}.xlsx")

st.success("✅ Validasi per metode selesai. Lanjut ke **Label Alignment** di sidebar.")
