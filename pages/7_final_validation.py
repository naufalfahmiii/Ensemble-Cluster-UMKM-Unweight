import streamlit as st

from modules.utils import step_header, require_state, download_button, reset_downstream
from modules.stability import build_reference_consensus, run_stability_analysis, build_final_evaluation
from modules.visualization import errorbar_metric_vs_k, composite_rank_bar

step_header("🏁", "Final Validation",
            "Sebelum menetapkan K final, hasil konsensus diuji ulang berkali-kali (stability analysis) "
            "untuk memastikan pengelompokan tidak berubah-ubah drastis akibat faktor acak semata.")

require_state("evaluation_ensemble", message="Voting ensemble belum dijalankan.",
               back_hint="Buka halaman **Ensemble Voting** terlebih dahulu.")

k_range = range(st.session_state.k_min, st.session_state.k_max + 1)

st.session_state.n_runs_stability = st.slider(
    "Jumlah pengulangan uji stabilitas per K", min_value=5, max_value=30,
    value=st.session_state.n_runs_stability, step=5,
    help="Notebook asli memakai 30 pengulangan. Nilai lebih kecil mempercepat proses namun kurang teliti."
)
n_k = st.session_state.k_max - st.session_state.k_min + 1
st.caption(f"Total simulasi: {n_k} nilai K × {st.session_state.n_runs_stability} pengulangan × 3 metode "
            f"= **{n_k * st.session_state.n_runs_stability * 3}** proses fit. Bisa memakan waktu beberapa menit.")

run = st.button("🚀 Jalankan Uji Stabilitas & Tetapkan K Final", type="primary", use_container_width=True)

if run:
    reset_downstream(6)
    with st.status("Membangun referensi konsensus (n_init=10)...", expanded=True) as status:
        # X_kproto & categorical_index perlu dibangun ulang dari df_typeaware (tidak disimpan di Bagian 3)
        from sklearn.preprocessing import StandardScaler
        df_typeaware = st.session_state.df_typeaware
        numeric_cols = st.session_state.numeric_cols
        categorical_cols = st.session_state.categorical_cols

        scaler = StandardScaler()
        data_kproto = df_typeaware.copy()
        if numeric_cols:
            data_kproto[numeric_cols] = scaler.fit_transform(data_kproto[numeric_cols])
        X_kproto = data_kproto[numeric_cols + categorical_cols].to_numpy()
        categorical_index = [X_kproto.shape[1] - len(categorical_cols) + i for i in range(len(categorical_cols))]

        reference_result = build_reference_consensus(
            st.session_state.X_kmeans, X_kproto, categorical_index, st.session_state.gower_matrix,
            k_range, st.session_state.gamma_kproto, st.session_state.random_state,
        )
        status.update(label="✅ Referensi konsensus siap")

        status.update(label="Menjalankan pengulangan uji stabilitas...", state="running")
        progress_bar = st.progress(0.0)

        def _cb(frac, k, seed):
            progress_bar.progress(frac, text=f"K = {k}, seed = {seed}")

        stability_df = run_stability_analysis(
            st.session_state.X_kmeans, X_kproto, categorical_index, st.session_state.gower_matrix,
            reference_result, k_range, st.session_state.gamma_kproto,
            n_runs=st.session_state.n_runs_stability, random_state=st.session_state.random_state,
            progress_cb=_cb,
        )
        progress_bar.empty()
        st.session_state.stability_df = stability_df

        evaluation_final = build_final_evaluation(st.session_state.evaluation_ensemble, stability_df)
        st.session_state.evaluation_final = evaluation_final
        st.session_state.best_k = int(evaluation_final.iloc[0]["K"])
        status.update(label="✅ Uji stabilitas & validasi akhir selesai", state="complete")

if st.session_state.evaluation_final is not None:
    best_k = st.session_state.best_k
    st.divider()
    st.markdown(f"## 🎯 K Final Terpilih: **{best_k}**")
    st.caption("Ditentukan dari peringkat komposit 5 kriteria: Silhouette, DBI, CHI, Mean ARI, dan Mean NMI.")

    st.plotly_chart(
        composite_rank_bar(st.session_state.evaluation_final, best_k,
                             title="Peringkat Komposit Akhir (5 Kriteria)"),
        use_container_width=True,
    )

    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(
            errorbar_metric_vs_k(st.session_state.stability_df, "Mean_ARI", "Std_ARI",
                                    "Stabilitas — Adjusted Rand Index", "Mean ARI", highlight_k=best_k),
            use_container_width=True,
        )
    with c2:
        st.plotly_chart(
            errorbar_metric_vs_k(st.session_state.stability_df, "Mean_NMI", "Std_NMI",
                                    "Stabilitas — Normalized Mutual Information", "Mean NMI", highlight_k=best_k,
                                    color="#A855F7"),
            use_container_width=True,
        )

    st.dataframe(st.session_state.evaluation_final, use_container_width=True, hide_index=True)
    c1, c2 = st.columns(2)
    with c1:
        download_button(st.session_state.evaluation_final, "evaluasi_final_5kriteria.xlsx")
    with c2:
        download_button(st.session_state.stability_df, "hasil_stability_analysis.xlsx")

    st.success(f"✅ K final = {best_k} siap dipakai. Lanjut ke **Sensitivity Check (Optional)"
                f"atau langsung ke **Visualization** di sidebar.")
else:
    st.info("Atur jumlah pengulangan lalu klik tombol di atas untuk menjalankan uji stabilitas.")
