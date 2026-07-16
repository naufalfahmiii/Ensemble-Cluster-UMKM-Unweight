import streamlit as st

from modules.utils import step_header, require_state
from modules.visualization import (
    line_metric_vs_k, compute_pca, pca_scatter_panel, cluster_size_bar,
)

step_header("🖼️", "Visualization",
            "Ringkasan visual dari seluruh proses — mulai dari performa tiap K hingga bentuk cluster "
            "hasil akhir dalam ruang dua dimensi.")

require_state("evaluation_final", "best_k", message="K final belum ditentukan.",
               back_hint="Buka halaman **Final Validation** terlebih dahulu.")

best_k = st.session_state.best_k
evaluation_ensemble = st.session_state.evaluation_ensemble

st.markdown("### 📉 Tren Kriteria Validasi terhadap Jumlah Cluster (K)")
c1, c2, c3 = st.columns(3)
with c1:
    st.plotly_chart(line_metric_vs_k(evaluation_ensemble, "Silhouette", "Silhouette Coefficient",
                                       "Silhouette (makin tinggi makin baik)", highlight_k=best_k),
                     use_container_width=True)
with c2:
    st.plotly_chart(line_metric_vs_k(evaluation_ensemble, "DBI", "Davies-Bouldin Index",
                                       "DBI (makin rendah makin baik)", highlight_k=best_k, color="#EF4444"),
                     use_container_width=True)
with c3:
    st.plotly_chart(line_metric_vs_k(evaluation_ensemble, "CHI", "Calinski-Harabasz Index",
                                       "CHI (makin tinggi makin baik)", highlight_k=best_k, color="#F97316"),
                     use_container_width=True)

st.divider()
st.markdown(f"### 🎨 Peta Sebaran Cluster (PCA 2D) pada K Final = {best_k}")
st.caption("Setiap titik mewakili satu UMKM. Tanda ✕ hitam menunjukkan pusat (centroid semu) tiap cluster.")

X_kmeans = st.session_state.X_kmeans
X_pca, var_ratio = compute_pca(X_kmeans, st.session_state.random_state)
st.caption(f"Total variansi data yang terwakili oleh 2 komponen utama: **{var_ratio.sum() * 100:.1f}%** "
            f"(PC1: {var_ratio[0]*100:.1f}%, PC2: {var_ratio[1]*100:.1f}%)")

km_labels_best = st.session_state.kmeans_result[f"Cluster_K{best_k}"].to_numpy()
kp_labels_best = st.session_state.aligned_result[f"KP_K{best_k}"].to_numpy()
gm_labels_best = st.session_state.aligned_result[f"GM_K{best_k}"].to_numpy()
ensemble_labels_best = st.session_state.ensemble_result[f"Cluster_K{best_k}"].to_numpy()

row1 = st.columns(2)
row2 = st.columns(2)
panels = [
    (row1[0], km_labels_best, f"K-Means (K={best_k})"),
    (row1[1], kp_labels_best, f"K-Prototypes — diselaraskan (K={best_k})"),
    (row2[0], gm_labels_best, f"Gower K-Medoids — diselaraskan (K={best_k})"),
    (row2[1], ensemble_labels_best, f"Ensemble Konsensus (K={best_k})"),
]
for col, labels, title in panels:
    with col:
        st.plotly_chart(pca_scatter_panel(X_pca, labels, title), use_container_width=True)

st.divider()
st.markdown("### 📦 Ukuran Tiap Cluster (Hasil Ensemble Final)")
st.plotly_chart(cluster_size_bar(ensemble_labels_best), use_container_width=True)

with st.expander("📎 Lampiran: Perbandingan pada K Terbaik Masing-Masing Metode (bukan K final)"):
    st.caption("Bersifat suplementer — bukan pembanding performa langsung karena jumlah cluster berbeda "
                "antar panel, hanya mengilustrasikan granularitas tiap representasi.")
    k_km_best = int(st.session_state.ranked_kmeans.iloc[0]["K"])
    k_kp_best = int(st.session_state.ranked_kproto.iloc[0]["K"])
    k_gm_best = int(st.session_state.ranked_kmedoids.iloc[0]["K"])

    km_labels_ownbest = st.session_state.kmeans_result[f"Cluster_K{k_km_best}"].to_numpy()
    kp_labels_ownbest = st.session_state.kproto_result[f"Cluster_K{k_kp_best}"].to_numpy()
    gm_labels_ownbest = st.session_state.kmedoids_result[f"Cluster_K{k_gm_best}"].to_numpy()

    r1 = st.columns(2)
    r2 = st.columns(2)
    appendix_panels = [
        (r1[0], km_labels_ownbest, f"K-Means (K terbaik metode = {k_km_best})"),
        (r1[1], kp_labels_ownbest, f"K-Prototypes (K terbaik metode = {k_kp_best})"),
        (r2[0], gm_labels_ownbest, f"Gower K-Medoids (K terbaik metode = {k_gm_best})"),
        (r2[1], ensemble_labels_best, f"Ensemble (K final = {best_k})"),
    ]
    for col, labels, title in appendix_panels:
        with col:
            st.plotly_chart(pca_scatter_panel(X_pca, labels, title), use_container_width=True)

st.success("✅ Lanjut ke **Profiling & Recommendation** di sidebar untuk melihat profil tiap cluster.")
