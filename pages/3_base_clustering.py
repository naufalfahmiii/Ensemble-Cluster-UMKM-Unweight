import streamlit as st

from modules.utils import step_header, require_state, download_button, reset_downstream
from modules.base_clustering import run_kmeans, run_kprototypes, run_gower_kmedoids

step_header("🧩", "Base Clustering",
            "Data yang sama dilihat dari tiga sudut pandang berbeda: K-Means (matriks baseline), "
            "K-Prototypes (campuran numerik+kategorikal), dan Gower K-Medoids (jarak Gower).")

require_state("df_baseline", "df_typeaware", message="Data belum diproses.",
               back_hint="Buka halaman **Preprocessing** terlebih dahulu.")

k_range = range(st.session_state.k_min, st.session_state.k_max + 1)
random_state = st.session_state.random_state

with st.expander("⚙️ Parameter Lanjutan"):
    gamma_kproto = st.number_input("Gamma K-Prototypes (bobot numerik vs kategorikal)",
                                     min_value=0.0, value=0.150004, step=0.01, format="%.6f")
    n_init_gower = st.slider("Jumlah inisialisasi Gower K-Medoids per K", min_value=3, max_value=20, value=10)

st.caption(f"Rentang K yang akan diuji: **{st.session_state.k_min} – {st.session_state.k_max}**")

run = st.button("🚀 Jalankan Base Clustering (3 metode)", type="primary", use_container_width=True)

if run:
    reset_downstream(2)
    df_baseline = st.session_state.df_baseline
    feature_order = st.session_state.feature_order_baseline
    df_typeaware = st.session_state.df_typeaware
    df_identity = st.session_state.df_identity
    numeric_cols = st.session_state.numeric_cols
    categorical_cols = st.session_state.categorical_cols

    with st.status("Menjalankan K-Means...", expanded=True) as status:
        kmeans_result, evaluation_kmeans, X_kmeans = run_kmeans(
            df_baseline, feature_order, df_identity, k_range, random_state
        )
        st.session_state.kmeans_result = kmeans_result
        st.session_state.evaluation_kmeans = evaluation_kmeans
        st.session_state.X_kmeans = X_kmeans
        status.update(label="✅ K-Means selesai")

        status.update(label="Menjalankan K-Prototypes (bisa memakan waktu)...", state="running")
        kproto_result, evaluation_kproto, categorical_index = run_kprototypes(
            df_typeaware, numeric_cols, categorical_cols, df_identity, k_range, gamma_kproto, random_state
        )
        st.session_state.kproto_result = kproto_result
        st.session_state.evaluation_kproto = evaluation_kproto
        st.session_state.gamma_kproto = gamma_kproto
        status.update(label="✅ K-Prototypes selesai")

        status.update(label="Menjalankan Gower K-Medoids (bisa memakan waktu)...", state="running")
        progress_bar = st.progress(0.0)

        def _cb(frac, k):
            progress_bar.progress(frac, text=f"K = {k}")

        kmedoids_result, evaluation_kmedoids, gower_matrix = run_gower_kmedoids(
            df_typeaware, numeric_cols, categorical_cols, df_identity, k_range,
            n_init_gower, random_state, progress_cb=_cb
        )
        st.session_state.kmedoids_result = kmedoids_result
        st.session_state.evaluation_kmedoids = evaluation_kmedoids
        st.session_state.gower_matrix = gower_matrix
        st.session_state.n_init_gower = n_init_gower
        progress_bar.empty()
        status.update(label="✅ Base clustering selesai untuk ketiga metode", state="complete")

if st.session_state.evaluation_kmedoids is not None:
    st.divider()
    tab1, tab2, tab3 = st.tabs(["K-Means", "K-Prototypes", "Gower K-Medoids"])

    with tab1:
        st.caption("Matriks baseline 8 fitur (atau sesuai pemetaan kolom Anda), MinMax + encoded.")
        st.dataframe(st.session_state.evaluation_kmeans, use_container_width=True, hide_index=True)
        c1, c2 = st.columns(2)
        with c1:
            download_button(st.session_state.kmeans_result, "hasil_kmeans_all.xlsx")
        with c2:
            download_button(st.session_state.evaluation_kmeans, "evaluasi_kmeans.xlsx")

    with tab2:
        st.caption("Huang init, numerik distandarisasi z-score, kategorikal apa adanya.")
        st.dataframe(st.session_state.evaluation_kproto, use_container_width=True, hide_index=True)
        c1, c2 = st.columns(2)
        with c1:
            download_button(st.session_state.kproto_result, "hasil_kprototype_all.xlsx")
        with c2:
            download_button(st.session_state.evaluation_kproto, "evaluasi_kprototype.xlsx")

    with tab3:
        st.caption("Matriks jarak Gower, bobot fitur setara, dipilih hasil terbaik dari beberapa inisialisasi.")
        st.dataframe(st.session_state.evaluation_kmedoids, use_container_width=True, hide_index=True)
        c1, c2 = st.columns(2)
        with c1:
            download_button(st.session_state.kmedoids_result, "hasil_kmedoids_all.xlsx")
        with c2:
            download_button(st.session_state.evaluation_kmedoids, "evaluasi_kmedoids.xlsx")

    st.success("✅ Base clustering selesai. Lanjut ke **Multi-Index Validation** di sidebar.")
else:
    st.info("Klik tombol di atas untuk menjalankan ketiga metode klasterisasi dasar.")
