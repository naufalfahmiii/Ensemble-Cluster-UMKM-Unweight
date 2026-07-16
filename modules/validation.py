"""Bagian 4 notebook — Multi-Index Validation per metode dasar."""
import pandas as pd


def rank_multi_index(evaluation_df: pd.DataFrame) -> pd.DataFrame:
    """Peringkat komposit multi-index (SC naik/maks baik, DBI turun/min baik, CHI naik/maks baik)."""
    out = evaluation_df.copy()
    out["Rank_SC"] = out["Silhouette"].rank(ascending=False)
    out["Rank_DBI"] = out["DBI"].rank(ascending=True)
    out["Rank_CHI"] = out["CHI"].rank(ascending=False)
    out["Total_Rank"] = out["Rank_SC"] + out["Rank_DBI"] + out["Rank_CHI"]
    out["Mean_Rank"] = out["Total_Rank"] / 3
    return out.sort_values("Mean_Rank").reset_index(drop=True)


def best_within_method(evaluation_df: pd.DataFrame, method_name: str, result_df: pd.DataFrame) -> dict:
    ranked = rank_multi_index(evaluation_df)
    best_row = ranked.iloc[0]
    best_k = int(best_row["K"])
    sizes = result_df[f"Cluster_K{best_k}"].value_counts()
    return {
        "Metode": method_name,
        "K Terbaik (dalam metode)": best_k,
        "Silhouette": round(best_row["Silhouette"], 4),
        "DBI": round(best_row["DBI"], 4),
        "CHI": round(best_row["CHI"], 2),
        "Ukuran Cluster (min-max)": f"{sizes.min()}-{sizes.max()}",
    }


def build_table2_summary(evaluation_kmeans, evaluation_kproto, evaluation_kmedoids,
                           kmeans_result, kproto_result, kmedoids_result) -> pd.DataFrame:
    rows = [
        best_within_method(evaluation_kmeans, "K-Means", kmeans_result),
        best_within_method(evaluation_kproto, "K-Prototypes", kproto_result),
        best_within_method(evaluation_kmedoids, "Gower K-Medoids", kmedoids_result),
    ]
    return pd.DataFrame(rows)
