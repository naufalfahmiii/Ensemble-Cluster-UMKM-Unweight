import streamlit as st

from modules.utils import step_header, require_state, download_button
from modules.ensemble import run_sensitivity_check

step_header("⚖️", "Sensitivity Check (Optional)",
            "Voting utama memberi bobot setara ke tiga metode (equal-vote). Halaman ini menguji: "
            "'Apa hasilnya berubah banyak jika metode dengan skor validasi lebih baik diberi bobot lebih besar?'")

require_state("evaluation_final", "best_k", message="K final belum ditentukan.",
               back_hint="Buka halaman **Final Validation** terlebih dahulu.")

best_k = st.session_state.best_k
st.info(f"Pengecekan dilakukan pada K final = **{best_k}**.")

run = st.button("🚀 Jalankan Uji Sensitivitas", type="primary", use_container_width=True)

if run:
    with st.spinner("Menghitung bobot kualitas per metode dan voting berbobot..."):
        result = run_sensitivity_check(
            st.session_state.evaluation_kmeans, st.session_state.evaluation_kproto, st.session_state.evaluation_kmedoids,
            st.session_state.aligned_result, st.session_state.ensemble_result, best_k,
        )
        st.session_state.sensitivity_result = result

if st.session_state.sensitivity_result is not None:
    result = st.session_state.sensitivity_result
    st.divider()
    st.markdown("### ⚖️ Bobot Kualitas per Metode")
    weights_df = result["weights"].rename("Bobot").reset_index().rename(columns={"index": "Metode"})
    st.dataframe(weights_df, use_container_width=True, hide_index=True)

    c1, c2, c3 = st.columns(3)
    c1.metric("Adjusted Rand Index (ARI)", f"{result['ari']:.4f}")
    c2.metric("Normalized Mutual Info (NMI)", f"{result['nmi']:.4f}")
    c3.metric("UMKM berpindah cluster", f"{result['n_changed']} / {result['n_total']}")

    if result["ari"] > 0.9:
        st.success("✅ Hasil sangat mirip dengan voting setara (equal-vote) — pemilihan bobot tidak "
                    "banyak mengubah kesimpulan. Voting equal-vote yang menjadi hasil utama cukup robust.")
    else:
        st.warning("⚠️ Terdapat perbedaan cukup berarti dibanding voting equal-vote. Pertimbangkan untuk "
                    "meninjau kembali metode mana yang paling relevan dengan karakteristik data Anda.")

    import pandas as pd
    comparison_df = st.session_state.df_identity.copy()
    comparison_df["Cluster_Equal_Vote"] = result["equal_vote_labels"]
    comparison_df["Cluster_Weighted_Vote"] = result["weighted_vote_labels"]
    comparison_df["Berpindah"] = comparison_df["Cluster_Equal_Vote"] != comparison_df["Cluster_Weighted_Vote"]
    st.dataframe(comparison_df.head(20), use_container_width=True)
    download_button(comparison_df, "perbandingan_equal_vs_weighted.xlsx")
else:
    st.info("Klik tombol di atas untuk menjalankan uji sensitivitas (opsional).")
