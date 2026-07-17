import pandas as pd
import streamlit as st

from modules.utils import step_header, reset_downstream
from modules.preprocessing import detect_encoding_suggestion

step_header("📂", "Setup Data",
            "Unggah data Anda, lalu tentukan kolom mana yang dipakai sebagai fitur klasterisasi.")

uploaded = st.file_uploader("Unggah file data (.xlsx atau .csv)", type=["xlsx", "xls", "csv"])

if uploaded is not None:
    try:
        if uploaded.name.lower().endswith(".csv"):
            df_raw = pd.read_csv(uploaded)
        else:
            df_raw = pd.read_excel(uploaded)
    except Exception as e:
        st.error(f"Gagal membaca file: {e}")
        st.stop()

    if st.session_state.get("raw_filename") != uploaded.name:
        # File baru diunggah -> reset seluruh tahap turunan + pemetaan kolom lama
        reset_downstream(1)
        st.session_state.raw_filename = uploaded.name
        st.session_state.numeric_cols = []
        st.session_state.categorical_cols = []
        st.session_state.identity_cols = []
        st.session_state.cat_encoding = {}

    st.session_state.raw_df = df_raw

if st.session_state.raw_df is None:
    st.info("Silakan unggah data terlebih dahulu untuk melanjutkan. Struktur kolom bebas — "
             "aplikasi ini tidak terpaku pada nama kolom dataset tertentu, jadi dataset UMKM "
             "apa pun bisa dipakai selama formatnya tabel (baris = 1 UMKM/entitas).")
    st.stop()

df_raw = st.session_state.raw_df

c1, c2, c3 = st.columns(3)
c1.metric("Jumlah Baris", f"{df_raw.shape[0]:,}")
c2.metric("Jumlah Kolom", df_raw.shape[1])
c3.metric("Nama File", st.session_state.raw_filename or "-")

with st.expander("👀 Pratinjau Data Mentah", expanded=True):
    st.dataframe(df_raw.head(20), use_container_width=True)

with st.expander("📋 Ringkasan Tipe & Nilai Kosong per Kolom"):
    info_df = pd.DataFrame({
        "Kolom": df_raw.columns,
        "Tipe Data": df_raw.dtypes.astype(str).values,
        "Nilai Kosong": df_raw.isnull().sum().values,
        "Nilai Unik": [df_raw[c].nunique() for c in df_raw.columns],
    })
    st.dataframe(info_df, use_container_width=True, hide_index=True)

st.divider()
st.markdown("### 🧭 Pemetaan Kolom")
st.caption(
    "Pilih kolom **numerik** dan **kategorikal** yang dipakai sebagai fitur klasterisasi. "
    "Kolom yang **tidak dipilih** otomatis disimpan sebagai kolom identitas — tetap ikut "
    "tersimpan hingga hasil akhir untuk keperluan profiling, tapi tidak memengaruhi perhitungan jarak/cluster."
)

all_cols = list(df_raw.columns)

if "numeric_cols_widget" not in st.session_state:
    st.session_state.numeric_cols_widget = [c for c in st.session_state.numeric_cols if c in all_cols]
if "categorical_cols_widget" not in st.session_state:
    st.session_state.categorical_cols_widget = [c for c in st.session_state.categorical_cols if c in all_cols]

col_a, col_b = st.columns(2)
with col_a:
    numeric_cols = st.multiselect(
        "Kolom numerik (mis. omset, aset, jumlah tenaga kerja)",
        options=all_cols,
        key="numeric_cols_widget",
        help="Kolom akan dibersihkan otomatis dari format mata uang/pemisah ribuan bila perlu.",
    )
with col_b:
    categorical_cols = st.multiselect(
        "Kolom kategorikal (mis. status legalitas, kanal pemasaran)",
        options=all_cols,
        key="categorical_cols_widget",
    )

# --- Penanganan overlap ---
overlap = [c for c in numeric_cols if c in categorical_cols]
if overlap:
    st.error(
        f"⚠️ Kolom berikut dipilih di **kedua** daftar sekaligus: "
        + ", ".join(f"`{c}`" for c in overlap)
        + ". Hapus dari salah satu daftar sebelum melanjutkan."
    )
    st.stop()

reserved_cols = [c for c in all_cols if c not in numeric_cols and c not in categorical_cols]

st.session_state.numeric_cols = numeric_cols
st.session_state.categorical_cols = categorical_cols

st.caption(f"🗂️ Kolom identitas / tidak dipakai sebagai fitur ({len(reserved_cols)}): "
            + (", ".join(f"`{c}`" for c in reserved_cols) if reserved_cols else "— tidak ada —"))

if not numeric_cols and not categorical_cols:
    st.warning("Pilih minimal satu kolom numerik atau kategorikal untuk melanjutkan.")
    st.stop()

# --- Strategi encoding tiap kolom kategorikal (untuk representasi baseline K-Means) ---
cat_encoding = dict(st.session_state.cat_encoding)
if categorical_cols:
    st.markdown("### 🏷️ Strategi Encoding Kolom Kategorikal")
    st.caption(
        "Menentukan bagaimana tiap kolom kategorikal diubah menjadi angka pada representasi "
        "baseline (dipakai K-Means). K-Prototypes & Gower K-Medoids selalu memakai nilai kategorikal aslinya."
    )
    for col in categorical_cols:
        suggestion = cat_encoding.get(col) or detect_encoding_suggestion(df_raw[col])
        with st.container(border=True):
            cc1, cc2 = st.columns([1, 2])
            with cc1:
                st.markdown(f"**`{col}`**")
                sample_vals = df_raw[col].dropna().astype(str).unique()[:5]
                st.caption("Contoh nilai: " + ", ".join(sample_vals))
            with cc2:
                type_options = ["binary", "multivalue", "nominal"]
                type_labels = {
                    "binary": "Biner (2 kategori -> 0/1)",
                    "multivalue": "Multi-nilai (dipisah koma, dihitung jumlah kanal)",
                    "nominal": "Nominal umum (one-hot encoding)",
                }
                chosen_type = st.selectbox(
                    f"Tipe encoding untuk `{col}`",
                    options=type_options,
                    index=type_options.index(suggestion["type"]),
                    format_func=lambda x: type_labels[x],
                    key=f"enc_type_{col}",
                    label_visibility="collapsed",
                )
                if chosen_type == "binary":
                    uniques = sorted(df_raw[col].dropna().astype(str).str.strip().str.title().unique())
                    default_positive = suggestion.get("positive_value", uniques[0] if uniques else "")
                    if default_positive not in uniques and uniques:
                        default_positive = uniques[0]
                    positive_value = st.selectbox(
                        f"Nilai yang dianggap 'positif' (=1) untuk `{col}`",
                        options=uniques if uniques else [""],
                        index=uniques.index(default_positive) if default_positive in uniques else 0,
                        key=f"enc_pos_{col}",
                    )
                    cat_encoding[col] = {"type": "binary", "positive_value": positive_value}
                elif chosen_type == "multivalue":
                    sep = st.text_input(f"Pemisah kanal untuk `{col}`", value=suggestion.get("separator", ","),
                                          key=f"enc_sep_{col}")
                    cat_encoding[col] = {"type": "multivalue", "separator": sep}
                else:
                    cat_encoding[col] = {"type": "nominal"}
    # Buang entri encoding untuk kolom yang sudah tidak lagi kategorikal
    cat_encoding = {k: v for k, v in cat_encoding.items() if k in categorical_cols}
else:
    cat_encoding = {}

changed = (
    numeric_cols != st.session_state.get("_prev_numeric_cols")
    or categorical_cols != st.session_state.get("_prev_categorical_cols")
    or reserved_cols != st.session_state.get("_prev_identity_cols")
    or cat_encoding != st.session_state.get("_prev_cat_encoding")
)

st.session_state.identity_cols = reserved_cols
st.session_state.cat_encoding = cat_encoding
st.session_state._prev_numeric_cols = numeric_cols
st.session_state._prev_categorical_cols = categorical_cols
st.session_state._prev_identity_cols = reserved_cols
st.session_state._prev_cat_encoding = cat_encoding

if changed:
    reset_downstream(1)

st.divider()
k1, k2, k3 = st.columns(3)
with k1:
    st.session_state.k_min = st.number_input("K minimum", min_value=2, max_value=8, value=st.session_state.k_min)
with k2:
    st.session_state.k_max = st.number_input("K maksimum", min_value=st.session_state.k_min + 1, max_value=15,
                                                value=max(st.session_state.k_max, st.session_state.k_min + 1))
with k3:
    st.session_state.random_state = st.number_input("Random state (agar hasil reproducible)",
                                                        min_value=0, value=st.session_state.random_state)

st.success(f"✅ Pemetaan kolom siap: {len(numeric_cols)} numerik, {len(categorical_cols)} kategorikal, "
            f"{len(reserved_cols)} kolom identitas tersimpan. Lanjut ke **Preprocessing** di sidebar.")
