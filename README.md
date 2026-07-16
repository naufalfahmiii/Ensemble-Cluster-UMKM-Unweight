# Ensemble Clustering UMKM — Aplikasi Streamlit (Jalur Unweighted)

Implementasi interaktif dari notebook `EnsembleClusterUMKM_Unweight_rev_FIX.ipynb`
(K-Means × K-Prototypes × Gower K-Medoids, Hungarian alignment, majority voting,
multi-index validation, dan stability analysis) — **digeneralisasi** agar bisa
dipakai untuk dataset UMKM lain, bukan cuma dataset spesifik yang dipakai saat
menulis notebook.

## 🚀 Cara Menjalankan di VS Code

1. Buka folder ini di VS Code.
2. Buat virtual environment (disarankan):
   ```bash
   python -m venv venv
   # Windows
   venv\Scripts\activate
   # Mac/Linux
   source venv/bin/activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Jalankan aplikasi:
   ```bash
   streamlit run app.py
   ```
5. Browser akan terbuka otomatis di `http://localhost:8501`.

> Jika `kmodes`, `gower`, atau `kmedoids` gagal terinstall, pastikan Python versi
> 3.9–3.12 dan `pip` sudah versi terbaru (`pip install --upgrade pip`).

## 🗂️ Struktur Proyek

```
umkm_app/
├── app.py                     # Entry point + navigasi sidebar kustom
├── requirements.txt
├── .streamlit/config.toml     # Tema warna aplikasi
├── modules/                   # Seluruh logika pipeline (tanpa Streamlit)
│   ├── utils.py                # session_state, tombol unduh
│   ├── preprocessing.py        # Bagian 1–2 notebook (digeneralisasi)
│   ├── base_clustering.py      # Bagian 3 notebook — K-Means/K-Prototypes/Gower K-Medoids
│   ├── validation.py           # Bagian 4 notebook — multi-index validation per metode
│   ├── alignment.py            # Bagian 5 notebook — Hungarian alignment
│   ├── ensemble.py             # Bagian 6 & 8 notebook — voting + sensitivity check
│   ├── stability.py            # Bagian 7 notebook — stability analysis
│   ├── profiling.py            # Bagian 9/10 notebook — profil & rekomendasi
│   └── visualization.py        # Semua chart Plotly (tanpa balloon/bubble plot)
└── pages/                     # Satu file = satu tahap alur (thin UI layer)
    ├── 1_upload.py
    ├── 2_preprocessing.py
    ├── 3_base_clustering.py
    ├── 4_validation.py
    ├── 5_alignment.py
    ├── 6_ensemble.py
    ├── 7_final_validation.py
    ├── 8_sensitivity.py
    ├── 9_visualization.py
    └── 10_profiling.py
```

Judul & ikon yang tampil di **sidebar** sengaja diatur manual lewat `st.Page(...)`
di `app.py` (bukan otomatis dari nama file) — supaya lebih ramah dibaca, mis.
`3_base_clustering.py` tampil sebagai **"🧩 Tiga Kacamata Klaster"**.

## 🔄 Alur Pemakaian

1. **Pintu Masuk Data** — unggah file (.csv/.xlsx), pilih kolom numerik & kategorikal,
   kolom yang tidak dipilih otomatis tersimpan sebagai identitas.
2. **Dapur Data** — pembersihan duplikat + dua representasi (type-aware & baseline).
3. **Tiga Kacamata Klaster** — jalankan K-Means, K-Prototypes, Gower K-Medoids untuk
   rentang K yang dipilih.
4. **Adu Performa Metode** — peringkat komposit (Silhouette, DBI, CHI) per metode.
5. **Menyamakan Bahasa Label** — Hungarian alignment terhadap K-Means.
6. **Musyawarah Mufakat** — voting mayoritas (equal-vote) antar 3 metode.
7. **Palang Pintu Akhir** — stability analysis (30 pengulangan default) → K final.
8. **Uji Sensitivitas** *(opsional)* — bandingkan equal-vote vs quality-weighted vote.
9. **Galeri Visual** — tren metrik vs K, sebaran PCA 2D per metode, ukuran cluster.
10. **Kartu Identitas Cluster** — profil tiap cluster + template rekomendasi kosong
    (diisi manual sesuai interpretasi domain, sesuai notebook asli).

Setiap tahap punya tombol **unduh Excel (.xlsx)** untuk hasil di tahap tersebut.

## ⚙️ Catatan Generalisasi (dibanding notebook asli)

Notebook asli memakai nama kolom & aturan encoding spesifik (mis. kolom
"NIB/SKU", "Sosmed", "Kepemilikan Lahan"). Karena dataset yang diunggah bisa
berbeda-beda, aplikasi ini meminta Anda memetakan:

- **Numerik** → dibersihkan dari format mata uang/pemisah ribuan (jika perlu),
  lalu IQR-clip + MinMax scaling (baseline) / z-score (K-Prototypes & Gower eval).
- **Kategorikal biner** (2 kategori, mis. "Ada"/"Tidak Ada") → 0/1.
- **Kategorikal multi-nilai** (dipisah koma, mis. daftar kanal pemasaran) →
  dihitung jumlah kanal, dinormalisasi.
- **Kategorikal nominal** (>2 kategori tanpa struktur multi-nilai) → one-hot
  encoding untuk representasi baseline; K-Prototypes & Gower K-Medoids tetap
  memakai nilai kategorikal aslinya (bukan one-hot).

Logika inti (IQR-clip+MinMax, Huang init K-Prototypes, gamma, fasterpam Gower,
Hungarian alignment, majority vote dengan tie-break ke K-Means, 5-kriteria
composite ranking pada tahap stabilitas) **dipertahankan identik** dengan notebook.

## ⏱️ Catatan Performa

Uji stabilitas (Bagian 7) adalah tahap paling berat: `jumlah K × jumlah
pengulangan × 3 metode` model fit. Untuk dataset besar (>2000 baris) atau
rentang K yang lebar, pertimbangkan menurunkan slider "Jumlah pengulangan"
di halaman **Palang Pintu Akhir** agar proses lebih cepat saat eksplorasi awal.
