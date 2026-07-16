"""Bagian 5 notebook — Hungarian Label Alignment."""
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import confusion_matrix


def hungarian_alignment(reference_labels, target_labels):
    """Menyelaraskan target_labels ke reference_labels via Hungarian assignment (maximum overlap)."""
    cm = confusion_matrix(reference_labels, target_labels)
    row_ind, col_ind = linear_sum_assignment(-cm)
    mapping = {col: row for row, col in zip(row_ind, col_ind)}
    return np.array([mapping[label] for label in target_labels])


def align_all(kmeans_result, kproto_result, kmedoids_result, df_identity, k_range) -> pd.DataFrame:
    aligned_result = df_identity.copy()
    for k in k_range:
        km_labels = kmeans_result[f"Cluster_K{k}"].to_numpy()
        kp_labels = kproto_result[f"Cluster_K{k}"].to_numpy()
        gm_labels = kmedoids_result[f"Cluster_K{k}"].to_numpy()

        kp_aligned = hungarian_alignment(km_labels, kp_labels)
        gm_aligned = hungarian_alignment(km_labels, gm_labels)

        aligned_result[f"KM_K{k}"] = km_labels
        aligned_result[f"KP_K{k}"] = kp_aligned
        aligned_result[f"GM_K{k}"] = gm_aligned
    return aligned_result


def hungarian_correspondence_table(reference_labels, target_labels, source_name) -> pd.DataFrame:
    """Tabel korespondensi label (mirip Tabel 3 paper) untuk satu nilai K contoh."""
    cm = confusion_matrix(reference_labels, target_labels)
    row_ind, col_ind = linear_sum_assignment(-cm)
    rows = []
    for r, c in zip(row_ind, col_ind):
        rows.append({
            "Source method": source_name,
            "Original source label": c,
            "Aligned K-Means label": f"C{r + 1}",
            "Maximum overlap, n": cm[r, c],
        })
    return pd.DataFrame(rows)
