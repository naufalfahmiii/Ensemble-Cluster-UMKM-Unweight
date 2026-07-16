"""Bagian 7 notebook — Multi-Index Validation Konsensus & Stability Analysis."""
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from kmodes.kprototypes import KPrototypes
import kmedoids
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from modules.alignment import hungarian_alignment
from modules.ensemble import majority_vote_with_tiebreak


def build_reference_consensus(X_kmeans, X_kproto, categorical_index, gower_matrix,
                                k_range, gamma_kproto, random_state=42):
    """Referensi konsensus (n_init=10, sama seperti Bagian 3 & 6) — pembanding stabilitas."""
    reference_result = {}
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=random_state, n_init=10, max_iter=300)
        km_labels = km.fit_predict(X_kmeans)

        kp = KPrototypes(n_clusters=k, init="Huang", n_init=10, max_iter=100,
                           gamma=gamma_kproto, random_state=random_state)
        kp_labels = kp.fit_predict(X_kproto, categorical=categorical_index)

        best_gm = None
        for init_i in range(10):
            candidate = kmedoids.fasterpam(gower_matrix, k, max_iter=100, random_state=random_state + init_i)
            if best_gm is None or candidate.loss < best_gm.loss:
                best_gm = candidate
        gm_labels = best_gm.labels

        kp_labels = hungarian_alignment(km_labels, kp_labels)
        gm_labels = hungarian_alignment(km_labels, gm_labels)

        reference_result[k] = majority_vote_with_tiebreak(km_labels, kp_labels, gm_labels)
    return reference_result


def run_stability_analysis(X_kmeans, X_kproto, categorical_index, gower_matrix,
                             reference_result, k_range, gamma_kproto,
                             n_runs=30, random_state=42, progress_cb=None):
    """
    30 (atau n_runs) pengulangan independen (n_init=1 per metode per seed),
    dibandingkan (ARI, NMI) terhadap referensi konsensus n_init=10.
    """
    seed_list = np.linspace(11000, 40000, n_runs, dtype=int)
    stability = []
    k_list = list(k_range)
    total_steps = len(k_list) * n_runs
    step = 0

    for k in k_list:
        reference = reference_result[k]
        ari_scores, nmi_scores = [], []

        for seed in seed_list:
            km = KMeans(n_clusters=k, random_state=int(seed), n_init=1, max_iter=300)
            km_labels = km.fit_predict(X_kmeans)

            kp = KPrototypes(n_clusters=k, init="Huang", n_init=1, max_iter=100,
                               gamma=gamma_kproto, random_state=int(seed))
            kp_labels = kp.fit_predict(X_kproto, categorical=categorical_index)

            gm = kmedoids.fasterpam(gower_matrix, k, max_iter=100, random_state=int(seed))
            gm_labels = gm.labels

            kp_labels = hungarian_alignment(km_labels, kp_labels)
            gm_labels = hungarian_alignment(km_labels, gm_labels)

            ensemble = majority_vote_with_tiebreak(km_labels, kp_labels, gm_labels)

            ari_scores.append(adjusted_rand_score(reference, ensemble))
            nmi_scores.append(normalized_mutual_info_score(reference, ensemble))

            step += 1
            if progress_cb:
                progress_cb(step / total_steps, k, int(seed))

        stability.append({
            "K": k,
            "Mean_ARI": float(np.mean(ari_scores)),
            "Std_ARI": float(np.std(ari_scores)),
            "Mean_NMI": float(np.mean(nmi_scores)),
            "Std_NMI": float(np.std(nmi_scores)),
        })

    return pd.DataFrame(stability)


def build_final_evaluation(evaluation_ensemble: pd.DataFrame, stability_df: pd.DataFrame) -> pd.DataFrame:
    """Gabungan 5 kriteria (SC, DBI, CHI, Mean ARI, Mean NMI) -> Mean_Rank -> K final."""
    evaluation_final = evaluation_ensemble.merge(stability_df, on="K")

    evaluation_final["Rank_SC"] = evaluation_final["Silhouette"].rank(ascending=False)
    evaluation_final["Rank_DBI"] = evaluation_final["DBI"].rank(ascending=True)
    evaluation_final["Rank_CHI"] = evaluation_final["CHI"].rank(ascending=False)
    evaluation_final["Rank_ARI"] = evaluation_final["Mean_ARI"].rank(ascending=False)
    evaluation_final["Rank_NMI"] = evaluation_final["Mean_NMI"].rank(ascending=False)

    evaluation_final["Total_Rank"] = (
        evaluation_final["Rank_SC"] + evaluation_final["Rank_DBI"] + evaluation_final["Rank_CHI"]
        + evaluation_final["Rank_ARI"] + evaluation_final["Rank_NMI"]
    )
    evaluation_final["Mean_Rank"] = evaluation_final["Total_Rank"] / 5

    return evaluation_final.sort_values("Mean_Rank").reset_index(drop=True)
