"""Bagian 9/10 notebook — Profiling Cluster & Rekomendasi (generalized)."""
import pandas as pd


def build_hasil_cluster(df_identity: pd.DataFrame, df_typeaware: pd.DataFrame,
                          numeric_cols: list, categorical_cols: list,
                          ensemble_result: pd.DataFrame, best_k: int) -> pd.DataFrame:
    # Pengaman: identitas, fitur, dan hasil cluster harus berasal dari baris yang
    # sama persis dalam jumlah & urutan sebelum digabung secara posisional.
    n_identity, n_typeaware, n_ensemble = len(df_identity), len(df_typeaware), len(ensemble_result)
    if not (n_identity == n_typeaware == n_ensemble):
        raise ValueError(
            "Jumlah baris tidak sinkron saat menggabungkan identitas dengan hasil cluster "
            f"(identitas={n_identity}, fitur={n_typeaware}, hasil cluster={n_ensemble}). "
            "Jalankan ulang dari halaman 'Dapur Data' untuk membangun ulang seluruh tahap secara konsisten."
        )

    hasil_cluster = df_identity.copy()
    hasil_cluster["Cluster"] = ensemble_result[f"Cluster_K{best_k}"].to_numpy()
    for col in numeric_cols + categorical_cols:
        hasil_cluster[col] = df_typeaware[col].to_numpy()
    return hasil_cluster


def profil_ukuran(hasil_cluster: pd.DataFrame) -> pd.DataFrame:
    counts = hasil_cluster["Cluster"].value_counts().sort_index()
    props = (counts / counts.sum() * 100).round(1)
    return pd.DataFrame({
        "Cluster": [f"C{c + 1}" for c in counts.index],
        "n": counts.values,
        "Persentase (%)": props.values,
    })


def _median_iqr(series):
    q1, q3 = series.quantile(0.25), series.quantile(0.75)
    return f"{series.median():,.1f} ({q1:,.1f}-{q3:,.1f})"


def profil_numerik(hasil_cluster: pd.DataFrame, numeric_cols: list) -> pd.DataFrame:
    if not numeric_cols:
        return pd.DataFrame(index=[f"C{c + 1}" for c in sorted(hasil_cluster["Cluster"].unique())])
    out = hasil_cluster.groupby("Cluster")[numeric_cols].agg(_median_iqr)
    out.index = [f"C{c + 1}" for c in out.index]
    return out


def profil_kategorikal(hasil_cluster: pd.DataFrame, categorical_cols: list, cat_encoding: dict) -> pd.DataFrame:
    """
    Generalisasi STEP 4 Bagian 9 notebook:
    - 'binary'    : % baris = positive_value.
    - 'multivalue': % baris memakai minimal satu kanal (bukan kosong/"Tidak ada").
    - 'nominal'   : % baris pada kategori paling umum (modus) di seluruh data.
    """
    if not categorical_cols:
        return pd.DataFrame(index=[f"C{c + 1}" for c in sorted(hasil_cluster["Cluster"].unique())])

    result = {}
    for col in categorical_cols:
        enc = cat_encoding.get(col, {"type": "nominal"})
        series_title = hasil_cluster[col].astype(str).str.strip().str.title()

        if enc["type"] == "binary":
            positive = str(enc.get("positive_value", "")).strip().title()
            label = f"{col}: {positive} (%)"
            result[label] = hasil_cluster.groupby("Cluster")[col].apply(
                lambda s: (s.astype(str).str.strip().str.title().eq(positive).mean() * 100).round(1)
            )
        elif enc["type"] == "multivalue":
            label = f"{col}: minimal 1 kanal (%)"
            empties = {"tidak ada", "-", "nan", "none", ""}
            result[label] = hasil_cluster.groupby("Cluster")[col].apply(
                lambda s: (~s.astype(str).str.strip().str.lower().isin(empties)).mean() * 100
            ).round(1)
        else:
            modus = series_title.mode()
            modus_val = modus.iloc[0] if len(modus) else ""
            label = f"{col}: {modus_val} (%)"
            result[label] = hasil_cluster.groupby("Cluster")[col].apply(
                lambda s: (s.astype(str).str.strip().str.title().eq(modus_val).mean() * 100).round(1)
            )

    out = pd.DataFrame(result)
    out.index = [f"C{c + 1}" for c in out.index]
    return out


def build_tabel6(p_ukuran, p_numerik, p_kategorikal) -> pd.DataFrame:
    return p_ukuran.set_index("Cluster").join(p_numerik).join(p_kategorikal)


def build_rekomendasi_template(tabel6_profil: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({
        "Cluster": tabel6_profil.index,
        "n (%)": [f"{r['n']} ({r['Persentase (%)']}%)" for _, r in tabel6_profil.iterrows()],
        "Label Deskriptif (isi manual)": ["" for _ in tabel6_profil.index],
        "Profil Teramati (isi manual)": ["" for _ in tabel6_profil.index],
        "Rekomendasi Pendampingan (isi manual)": ["" for _ in tabel6_profil.index],
        "Validasi Pakar Sebelum Aksi (isi manual)": ["" for _ in tabel6_profil.index],
    })
