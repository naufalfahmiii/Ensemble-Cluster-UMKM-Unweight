"""
Bagian 3 notebook — Base Clustering: K-Means, K-Prototypes, Gower K-Medoids.
Logika & parameter (n_init, gamma, max_iter) dipertahankan identik dengan
notebook; hanya kolom sumber yang digeneralisasi lewat parameter fungsi.
"""
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from kmodes.kprototypes import KPrototypes
import gower
import kmedoids


def run_kmeans(df_baseline: pd.DataFrame, feature_order: list, df_identity: pd.DataFrame,
                k_range, random_state=42):
    """3a — K-Means pada matriks baseline."""
    X_kmeans = df_baseline[feature_order].to_numpy(dtype=float)

    kmeans_result = df_identity.copy()
    evaluation = []
    for k in k_range:
        model = KMeans(n_clusters=k, init="k-means++", random_state=random_state, n_init=10, max_iter=300)
        labels = model.fit_predict(X_kmeans)
        kmeans_result[f"Cluster_K{k}"] = labels
        sizes = pd.Series(labels).value_counts()
        evaluation.append({
            "K": k, "Inertia": model.inertia_,
            "Silhouette": silhouette_score(X_kmeans, labels),
            "DBI": davies_bouldin_score(X_kmeans, labels),
            "CHI": calinski_harabasz_score(X_kmeans, labels),
            "Min_Cluster_Size": int(sizes.min()), "Max_Cluster_Size": int(sizes.max()),
        })
    return kmeans_result, pd.DataFrame(evaluation), X_kmeans


def run_kprototypes(df_typeaware: pd.DataFrame, numeric_cols: list, categorical_cols: list,
                      df_identity: pd.DataFrame, k_range, gamma=0.150004, random_state=42):
    """3b — K-Prototypes (Huang init, numerik distandarisasi z-score)."""
    scaler = StandardScaler()
    data_kproto = df_typeaware.copy()
    if numeric_cols:
        data_kproto[numeric_cols] = scaler.fit_transform(data_kproto[numeric_cols])

    X_kproto = data_kproto[numeric_cols + categorical_cols].to_numpy()
    categorical_index = [
        X_kproto.shape[1] - len(categorical_cols) + i for i in range(len(categorical_cols))
    ]

    kproto_result = df_identity.copy()
    evaluation = []
    X_numeric_kproto = data_kproto[numeric_cols] if numeric_cols else None

    for k in k_range:
        model = KPrototypes(n_clusters=k, init="Huang", n_init=10, max_iter=100,
                              gamma=gamma, random_state=random_state)
        labels = model.fit_predict(X_kproto, categorical=categorical_index)
        kproto_result[f"Cluster_K{k}"] = labels
        sizes = pd.Series(labels).value_counts()
        row = {
            "K": k, "Cost": model.cost_,
            "Min_Cluster_Size": int(sizes.min()), "Max_Cluster_Size": int(sizes.max()),
        }
        if X_numeric_kproto is not None and X_numeric_kproto.shape[1] > 0:
            row["Silhouette"] = silhouette_score(X_numeric_kproto, labels)
            row["DBI"] = davies_bouldin_score(X_numeric_kproto, labels)
            row["CHI"] = calinski_harabasz_score(X_numeric_kproto, labels)
        else:
            row["Silhouette"] = np.nan
            row["DBI"] = np.nan
            row["CHI"] = np.nan
        evaluation.append(row)

    return kproto_result, pd.DataFrame(evaluation), categorical_index


def run_gower_kmedoids(df_typeaware: pd.DataFrame, numeric_cols: list, categorical_cols: list,
                         df_identity: pd.DataFrame, k_range, n_init_gower=10, random_state=42,
                         progress_cb=None):
    """3c — Gower K-Medoids (fasterpam, multi-init pilih loss terbaik)."""
    gower_cols = numeric_cols + categorical_cols
    X_gower = df_typeaware[gower_cols].copy()
    # Hindari bug pandas 3.x StringDtype pada library gower: paksa dtype object.
    for c in categorical_cols:
        X_gower[c] = X_gower[c].astype(object)

    gower_matrix = gower.gower_matrix(X_gower)

    scaler_eval = StandardScaler()
    if numeric_cols:
        X_numeric_gower = pd.DataFrame(
            scaler_eval.fit_transform(df_typeaware[numeric_cols]), columns=numeric_cols
        )
    else:
        X_numeric_gower = None

    kmedoids_result = df_identity.copy()
    evaluation = []
    total = len(list(k_range))
    for i, k in enumerate(k_range):
        best_model = None
        for init_i in range(n_init_gower):
            candidate = kmedoids.fasterpam(gower_matrix, k, max_iter=100, random_state=random_state + init_i)
            if best_model is None or candidate.loss < best_model.loss:
                best_model = candidate
        labels = best_model.labels
        kmedoids_result[f"Cluster_K{k}"] = labels
        sizes = pd.Series(labels).value_counts()
        row = {
            "K": k, "Loss": best_model.loss,
            "Silhouette": silhouette_score(gower_matrix, labels, metric="precomputed"),
            "Min_Cluster_Size": int(sizes.min()), "Max_Cluster_Size": int(sizes.max()),
        }
        if X_numeric_gower is not None and X_numeric_gower.shape[1] > 0:
            row["DBI"] = davies_bouldin_score(X_numeric_gower, labels)
            row["CHI"] = calinski_harabasz_score(X_numeric_gower, labels)
        else:
            row["DBI"] = np.nan
            row["CHI"] = np.nan
        evaluation.append(row)
        if progress_cb:
            progress_cb((i + 1) / total, k)

    return kmedoids_result, pd.DataFrame(evaluation), gower_matrix
