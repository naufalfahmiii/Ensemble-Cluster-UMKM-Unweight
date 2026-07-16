"""Bagian 6 & 8 notebook — Ensemble Voting (equal-vote) dan sensitivity check (quality-weighted)."""
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    silhouette_score, davies_bouldin_score, calinski_harabasz_score,
    adjusted_rand_score, normalized_mutual_info_score,
)
import gower


def majority_vote_with_tiebreak(km_labels, kp_labels, gm_labels):
    """
    Label konsensus = suara terbanyak dari 3 metode dasar yang sudah di-align.
    Jika seri (3 label berbeda), ikuti label referensi K-Means.
    """
    km_labels, kp_labels, gm_labels = map(np.asarray, (km_labels, kp_labels, gm_labels))
    n = len(km_labels)
    consensus = np.empty(n, dtype=int)
    votes = np.column_stack([km_labels, kp_labels, gm_labels])
    for i in range(n):
        values, counts = np.unique(votes[i], return_counts=True)
        max_count = counts.max()
        winners = values[counts == max_count]
        consensus[i] = winners[0] if len(winners) == 1 else km_labels[i]
    return consensus


def build_eval_space(df_typeaware: pd.DataFrame, numeric_cols: list, categorical_cols: list):
    """Ruang evaluasi konsensus: SC dari Gower, DBI/CHI dari (z-score numerik + one-hot kategorikal)."""
    transformers = []
    if numeric_cols:
        transformers.append(("num", StandardScaler(), numeric_cols))
    if categorical_cols:
        transformers.append(("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols))

    eval_preprocessor = ColumnTransformer(transformers=transformers)
    X_eval_combined = eval_preprocessor.fit_transform(df_typeaware)
    X_eval_combined = np.asarray(X_eval_combined.todense() if hasattr(X_eval_combined, "todense") else X_eval_combined)

    gower_cols = numeric_cols + categorical_cols
    X_gower = df_typeaware[gower_cols].copy()
    for c in categorical_cols:
        X_gower[c] = X_gower[c].astype(object)
    gower_matrix_eval = gower.gower_matrix(X_gower)

    return X_eval_combined, gower_matrix_eval


def run_ensemble(aligned_result, df_identity, k_range, X_eval_combined, gower_matrix_eval):
    ensemble_result = df_identity.copy()
    evaluation = []
    for k in k_range:
        labels = majority_vote_with_tiebreak(
            aligned_result[f"KM_K{k}"].to_numpy(),
            aligned_result[f"KP_K{k}"].to_numpy(),
            aligned_result[f"GM_K{k}"].to_numpy(),
        )
        ensemble_result[f"Cluster_K{k}"] = labels
        sizes = pd.Series(labels).value_counts()
        evaluation.append({
            "K": k,
            "Silhouette": silhouette_score(gower_matrix_eval, labels, metric="precomputed"),
            "DBI": davies_bouldin_score(X_eval_combined, labels),
            "CHI": calinski_harabasz_score(X_eval_combined, labels),
            "Min_Cluster_Size": int(sizes.min()), "Max_Cluster_Size": int(sizes.max()),
        })
    return ensemble_result, pd.DataFrame(evaluation)


def method_quality_weights(evaluation_kmeans, evaluation_kproto, evaluation_kmedoids, k):
    """q_m = 1 + mean(z_SC, z_DBI(reversed), z_CHI) ; w_m = q_m / sum(q_m)."""
    rows = {
        "K-Means": evaluation_kmeans.loc[evaluation_kmeans["K"] == k].iloc[0],
        "K-Prototypes": evaluation_kproto.loc[evaluation_kproto["K"] == k].iloc[0],
        "Gower K-Medoids": evaluation_kmedoids.loc[evaluation_kmedoids["K"] == k].iloc[0],
    }
    sc = pd.Series({m: r["Silhouette"] for m, r in rows.items()})
    dbi = pd.Series({m: r["DBI"] for m, r in rows.items()})
    chi = pd.Series({m: r["CHI"] for m, r in rows.items()})

    def minmax(s, reverse=False):
        if s.max() == s.min():
            return pd.Series(0.5, index=s.index)
        z = (s - s.min()) / (s.max() - s.min())
        return 1 - z if reverse else z

    q = 1 + (minmax(sc) + minmax(dbi, reverse=True) + minmax(chi)) / 3
    return q / q.sum()


def weighted_vote(km_labels, kp_labels, gm_labels, weights):
    km_labels, kp_labels, gm_labels = map(np.asarray, (km_labels, kp_labels, gm_labels))
    n = len(km_labels)
    consensus = np.empty(n, dtype=int)
    for i in range(n):
        tally = {}
        for lbl, w in zip(
            (km_labels[i], kp_labels[i], gm_labels[i]),
            (weights["K-Means"], weights["K-Prototypes"], weights["Gower K-Medoids"]),
        ):
            tally[lbl] = tally.get(lbl, 0.0) + w
        max_w = max(tally.values())
        winners = [lbl for lbl, w in tally.items() if np.isclose(w, max_w)]
        consensus[i] = winners[0] if len(winners) == 1 else km_labels[i]
    return consensus


def run_sensitivity_check(evaluation_kmeans, evaluation_kproto, evaluation_kmedoids,
                            aligned_result, ensemble_result, best_k):
    weights_best_k = method_quality_weights(evaluation_kmeans, evaluation_kproto, evaluation_kmedoids, best_k)
    equal_vote_best_k = ensemble_result[f"Cluster_K{best_k}"].to_numpy()
    weighted_vote_best_k = weighted_vote(
        aligned_result[f"KM_K{best_k}"].to_numpy(),
        aligned_result[f"KP_K{best_k}"].to_numpy(),
        aligned_result[f"GM_K{best_k}"].to_numpy(),
        weights_best_k,
    )
    ari = adjusted_rand_score(equal_vote_best_k, weighted_vote_best_k)
    nmi = normalized_mutual_info_score(equal_vote_best_k, weighted_vote_best_k)
    n_changed = int(np.sum(equal_vote_best_k != weighted_vote_best_k))
    return {
        "weights": weights_best_k,
        "ari": ari,
        "nmi": nmi,
        "n_changed": n_changed,
        "n_total": len(equal_vote_best_k),
        "equal_vote_labels": equal_vote_best_k,
        "weighted_vote_labels": weighted_vote_best_k,
    }
