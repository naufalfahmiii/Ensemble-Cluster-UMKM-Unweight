import streamlit as st

from modules.utils import step_header, require_state, download_button, reset_downstream
from modules.ensemble import build_eval_space, run_ensemble

step_header("🗳️", "Ensemble Voting",
            "Untuk setiap UMKM, tiga metode 'memberi suara' klaster mana yang paling cocok. "
            "Label yang muncul paling banyak (2 dari 3, atau ikut K-Means bila seri) menjadi label konsensus.")

require_state("aligned_result", message="Penyelarasan label belum dijalankan.",
               back_hint="Buka halaman **Label Alignment** terlebih dahulu.")

k_range = range(st.session_state.k_min, st.session_state.k_max + 1)

if st.session_state.ensemble_result is None:
    with st.spinner("Menghitung ruang evaluasi konsensus (Gower + z-score/one-hot gabungan)..."):
        X_eval_combined, gower_matrix_eval = build_eval_space(
            st.session_state.df_typeaware, st.session_state.numeric_cols, st.session_state.categorical_cols
        )
        st.session_state.X_eval_combined = X_eval_combined
        st.session_state.gower_matrix_eval = gower_matrix_eval

    with st.spinner("Melakukan voting mayoritas untuk setiap nilai K..."):
        ensemble_result, evaluation_ensemble = run_ensemble(
            st.session_state.aligned_result, st.session_state.df_identity, k_range,
            st.session_state.X_eval_combined, st.session_state.gower_matrix_eval,
        )
        st.session_state.ensemble_result = ensemble_result
        st.session_state.evaluation_ensemble = evaluation_ensemble
    reset_downstream(6)

st.markdown("### 📋 Hasil Voting per Nilai K")
st.dataframe(st.session_state.ensemble_result.head(20), use_container_width=True)
download_button(st.session_state.ensemble_result, "hasil_ensemble_all.xlsx")

st.divider()
st.markdown("### 📈 Evaluasi Hasil Konsensus")
st.caption("Dihitung pada ruang evaluasi gabungan (bukan ruang metode manapun) agar adil untuk ketiganya.")
st.dataframe(st.session_state.evaluation_ensemble, use_container_width=True, hide_index=True)
download_button(st.session_state.evaluation_ensemble, "evaluasi_ensemble.xlsx")

st.success("✅ Voting ensemble selesai. Lanjut ke **Final Validation** di sidebar untuk "
            "menentukan K final lewat uji stabilitas.")
