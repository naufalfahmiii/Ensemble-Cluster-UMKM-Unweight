"""
Bagian 1 & 2 notebook — Data Loading dan Cleaning & Type-Aware Preprocessing.

Digeneralisasi dari notebook asli agar tidak terikat nama kolom
"Jml. Tenaga Kerja", "Sosmed", dll. — user memilih sendiri kolom mana
yang numerik/kategorikal/identitas, dan bagaimana tiap kolom kategorikal
dikodekan untuk representasi baseline K-Means.
"""
import numpy as np
import pandas as pd


def clean_numeric_series(s: pd.Series) -> pd.Series:
    """
    Membersihkan satu kolom numerik: jika sudah bertipe numerik, langsung
    dipakai. Jika masih berupa teks (mis. format mata uang "Rp1.500.000"),
    simbol mata uang dan pemisah ribuan dibuang lalu dikonversi ke numerik
    — persis logika STEP 6 pada notebook, digeneralisasi untuk kolom apa pun.
    """
    if pd.api.types.is_numeric_dtype(s):
        return s.astype(float)
    cleaned = (
        s.astype(str)
        .str.replace("Rp", "", regex=False)
        .str.replace(r"[^\d,.\-]", "", regex=True)
        .str.replace(".", "", regex=False)
        .str.replace(",", "", regex=False)
        .str.strip()
    )
    return pd.to_numeric(cleaned, errors="coerce")


def clean_categorical_series(s: pd.Series) -> pd.Series:
    """Standarisasi spasi & kapitalisasi (title case) — STEP 6 notebook."""
    return s.astype(str).str.strip().str.title()


def drop_duplicates(df: pd.DataFrame, exclude_cols=None) -> tuple[pd.DataFrame, dict]:
    """STEP 4 notebook — hapus duplikat berdasarkan seluruh kolom kecuali kolom penomoran/ID."""
    exclude_cols = exclude_cols or []
    dedup_subset = [c for c in df.columns if c not in exclude_cols]
    n_before = len(df)
    df_clean = df.drop_duplicates(subset=dedup_subset).reset_index(drop=True)
    n_after = len(df_clean)
    info = {
        "n_before": n_before,
        "n_after": n_after,
        "n_removed": n_before - n_after,
    }
    return df_clean, info


def split_identity(df_clean: pd.DataFrame, identity_cols: list) -> pd.DataFrame:
    """STEP 5 notebook — memisahkan kolom identitas/tidak-dipakai (tetap disimpan, ketentuan #1)."""
    cols = [c for c in identity_cols if c in df_clean.columns]
    return df_clean[cols].copy().reset_index(drop=True)


def build_typeaware(df_clean: pd.DataFrame, numeric_cols: list, categorical_cols: list) -> pd.DataFrame:
    """
    STEP 6 notebook — representasi type-aware skala asli:
    dipakai oleh K-Prototypes & Gower K-Medoids.
    """
    out = df_clean[numeric_cols + categorical_cols].copy().reset_index(drop=True)
    for col in numeric_cols:
        out[col] = clean_numeric_series(out[col])
    for col in categorical_cols:
        out[col] = clean_categorical_series(out[col])
    return out


_POSITIVE_KEYWORDS = [
    "ada", "ya", "sendiri", "milik", "aktif", "terdaftar", "memiliki", "punya",
    "sudah", "yes", "true", "own", "owned", "active", "registered",
]
_NEGATIVE_KEYWORDS = [
    "tidak ada", "tidak", "belum", "tanpa", "kosong", "sewa", "no", "false",
    "none", "not registered", "unregistered", "rent", "rented",
]


def _guess_positive_by_keyword(unique_vals):
    """
    Menebak kategori 'positif' berdasarkan makna kata (mis. 'Ada' > 'Tidak Ada',
    'Milik Sendiri' > 'Sewa'), meniru konvensi umum penamaan kolom biner pada
    data survei/administratif Indonesia — dipakai sebagai default sebelum
    fallback ke heuristik frekuensi. Selalu bisa diubah manual oleh user.
    """
    lowered = {v: str(v).strip().lower() for v in unique_vals}
    for val, low in lowered.items():
        if any(neg in low for neg in _NEGATIVE_KEYWORDS):
            other = [v for v in unique_vals if v != val]
            if other:
                return other[0]
    for val, low in lowered.items():
        if any(pos in low for pos in _POSITIVE_KEYWORDS):
            return val
    return None


def detect_encoding_suggestion(series: pd.Series, sep: str = ",") -> dict:
    """
    Menebak strategi encoding kategorikal yang cocok untuk sebuah kolom,
    sebagai default yang bisa diubah manual oleh user di halaman Unggah Data:
    - 'binary'    : tepat 2 kategori unik -> 0/1
    - 'multivalue': sebagian nilai mengandung pemisah (mis. koma) -> hitung kanal
    - 'nominal'   : selain itu -> one-hot encoding
    """
    vals = series.astype(str).str.strip()
    unique_vals = vals[vals.str.lower().ne("nan")].unique()
    contains_sep = vals.str.contains(sep, regex=False).mean() > 0.05

    if contains_sep:
        return {"type": "multivalue", "separator": sep}
    if len(unique_vals) == 2:
        keyword_positive = _guess_positive_by_keyword(unique_vals)
        if keyword_positive is not None:
            return {"type": "binary", "positive_value": keyword_positive}
        # fallback: kategori yang lebih jarang muncul dianggap 'positif'
        # (biasanya lebih informatif, mis. "Ada" vs "Tidak Ada")
        counts = vals.value_counts()
        positive = counts.idxmin() if len(counts) == 2 else unique_vals[0]
        return {"type": "binary", "positive_value": positive}
    return {"type": "nominal"}


def _count_channels(value, sep=","):
    text = str(value).strip()
    if text == "" or text.lower() in ("tidak ada", "-", "nan", "none"):
        return 0
    return len([item for item in text.split(sep) if item.strip() != ""])


def build_baseline(df_typeaware: pd.DataFrame, numeric_cols: list,
                    categorical_cols: list, cat_encoding: dict) -> tuple[pd.DataFrame, list]:
    """
    STEP 7 notebook, digeneralisasi — matriks baseline siap-K-Means:
    - Numerik   : IQR-clip lalu MinMax-scale ke [0, 1] (identik notebook).
    - Kategorikal 'binary'    : positive_value -> 1, selain itu -> 0.
    - Kategorikal 'multivalue': jumlah kanal / kanal maksimum teramati (proxy ordinal).
    - Kategorikal 'nominal'   : one-hot encoding (agar tetap numerik untuk K-Means).

    Mengembalikan (df_baseline, feature_order) — feature_order dipakai
    konsisten di seluruh tahap agar urutan kolom X_kmeans stabil.
    """
    out = pd.DataFrame(index=df_typeaware.index)
    feature_order = []

    for col in numeric_cols:
        x = pd.to_numeric(df_typeaware[col], errors="coerce").astype(float)
        q1, q3 = x.quantile(0.25), x.quantile(0.75)
        iqr = q3 - q1
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        x_clipped = x.clip(lower, upper)
        x_min, x_max = x_clipped.min(), x_clipped.max()
        denom = (x_max - x_min) if (x_max - x_min) != 0 else 1.0
        out[col] = (x_clipped - x_min) / denom
        feature_order.append(col)

    for col in categorical_cols:
        enc = cat_encoding.get(col, {"type": "nominal"})
        series = df_typeaware[col].astype(str).str.strip().str.title()

        if enc["type"] == "binary":
            positive = str(enc.get("positive_value", "")).strip().title()
            out[col] = series.eq(positive).astype(int)
            feature_order.append(col)

        elif enc["type"] == "multivalue":
            sep = enc.get("separator", ",")
            counts = df_typeaware[col].apply(lambda v: _count_channels(v, sep))
            max_count = counts.max() if counts.max() > 0 else 1
            out[col] = counts / max_count
            feature_order.append(col)

        else:  # nominal -> one-hot
            dummies = pd.get_dummies(series, prefix=col).astype(int)
            for dcol in dummies.columns:
                out[dcol] = dummies[dcol]
                feature_order.append(dcol)

    return out, feature_order
