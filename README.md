# Prediksi Tinggi Muka Air (TMA) Bengawan Solo
### Sebelas Maret Statistics & Data Science Competition 2026

**Ferliyana Ronnan, ig: @ron.nan7**

**Kampus:** Universitas Negeri Surabaya (UNESA)

---

## Gambaran Masalah

Notebook ini memprediksi **Tinggi Muka Air (`tma_mdpl`)** di 30 pos pemantauan DAS Bengawan Solo, berdasarkan data historis TMA dan fitur eksogen (curah hujan, kelembapan tanah, tekanan udara, indeks iklim global). Horizon prediksi adalah 241 hari (~8 bulan) ke depan, langsung menyambung dari akhir data train.

Data terdiri dari 4 sumber:
| File | Isi | Granularitas |
|---|---|---|
| `train.csv` | TMA aktual per pos | 3x sehari (06.00/12.00/18.00), Jan 2023 - Sep 2025 |
| `test.csv` | `id` yang perlu diprediksi | Sep 2025 - Mei 2026 |
| `data_lingkungan.csv` | Cuaca & fitur eksogen per pos | Per jam, Jan 2023 - Mei 2026 |
| `koordinat_pos.csv` | Lat/lon 30 pos pemantauan | Statis |

Notebook terbagi jadi 2 bagian besar:

## Bagian 1 EDA

Eksplorasi data sebelum modeling, isinya 18 poin analisis:

1. **Load data** load 4 sumber, parsing datetime, split `id` test.
2. **Kelengkapan data per pos** cek % slot waktu ideal (tiap 6 jam) yang beneran ada, ketemu 2 pos bermasalah: **Gunungsari** (~28%) dan **Floodway Bridge C** (~60%).
3. **Skala TMA beda-beda tiap pos** karena `tma_mdpl` adalah "meter di atas permukaan laut", elevasi lokasi pos ikut mempengaruhi nilainya jadi model harus per-pos aware.
4. **Deteksi outlier ekstrem** pakai IQR per pos (faktor 3x, bukan 1.5x standar, biar nggak overdeteksi variasi musiman wajar).
5. **Contoh time series pos stabil vs fluktuatif.**
6. **Pola musiman** cek korelasi TMA dengan musim hujan (Nov-Apr) vs kemarau (Mei-Okt) lewat z-score per pos.
7. **Korelasi TMA dengan fitur eksogen** instan (curah hujan, soil moisture, dll).
8. **Sebaran lokasi 30 pos** relasi kasar hulu-hilir dari lat/lon.
9. **Indeks iklim global** (Niño 3.4 & MJO) nilai global yang sudah tersedia sampai periode test, jadi valid dipakai langsung tanpa forecast sendiri.
10. **Skala TMA vs keadilan metric RMSE** cek pos mana yang paling dominan nyumbang ke RMSE gabungan kompetisi.
11. **Korelasi dengan lag curah hujan & soil moisture** (rolling 24h/72h) efek hujan ke TMA nggak instan.
12. **Autokorelasi (ACF)** cek sampai lag berapa TMA masih "nyambung".
13. **Pergeseran distribusi fitur eksogen train vs test** cek risiko ekstrapolasi buat model tree-based.
14. **Proxy hubungan hulu-hilir** pakai korelasi TMA antar pos + jarak geografis (haversine), sebagai alternatif ringan dari shapefile `HydroRIVERS` (~200MB).
15. **Pola bolongnya data** sistemik (bareng-bareng semua pos) atau acak.
16. **Missing value di `data_lingkungan.csv`**, termasuk periode test.
17. **Cek duplicate timestamp.**
18. **Cross-check spike TMA vs curah hujan** awalnya dikira "banjir asli", tapi setelah dicek manual polanya ternyata **sensor glitch** (naik ratusan meter dalam 1 bacaan lalu balik persis ke level semula) bukan hidrologi asli. Temuan ini yang jadi dasar `clean_sensor_glitches` di Bagian 2.

## Bagian 2 Modeling Pipeline

Pipeline final terdiri dari:

- **Data cleaning:** despike sensor glitch (`clean_sensor_glitches`) di `tma_mdpl` mentah, sebelum baseline/fitur dihitung.
- **Baseline musiman (Fourier):** dibandingkan order 1/2/3 lewat CV honest walk-forward order-1 menang (order-3 overfit parah, apalagi di pos berdata sedikit seperti Gunungsari).
- **Multi-fold time-based CV, horizon-matched:** validasi pakai panjang horizon yang sama seperti test asli (241 hari), plus perbandingan RMSE train vs val buat cek overfitting.
- **Exclude pos histori pendek** (`get_valid_pos_for_fold`) biar evaluasi nggak diganggu pos baru seperti Gunungsari.
- **Hyperparameter tuning (Optuna):** termasuk pilihan `objective` (L1/L2/Huber) ikut dituning.
- **Seed bagging:** 3 seed per model untuk mengurangi variance.
- **Stage A → Stage B → Persistence blend:**
  - Stage A: LightGBM dari fitur eksogen + kalender.
  - Stage B: tambah fitur upstream (peta hulu-hilir yang di-filter kualitas overlap & korelasi minimum, weighted top-3 kandidat), dihitung OOF biar nggak leakage.
  - Blend weight Stage A/B dan tau decay persistence di-grid-search bareng lintas semua fold.
- **Sensitivitas winsorize target:** dibandingkan threshold 1%/2%/5% winsorize tetap paling optimal.
- **Eksperimen tambahan:** quantile median (alpha=0.5) vs point prediction, feature importance, clustering per volatilitas pos, per-pos model.
- **Fit final & submission:** semua keputusan digabung, fit ke seluruh data train, prediksi test, hasil disimpan ke `submission.csv`.
- **Bonus uncertainty quantification:** interval prediksi 80% (p10-p90) pakai quantile regression LightGBM, di luar submission utama.

## Cara Menjalankan

1. Siapkan file zip kompetisi (`sebelas-maret-statistics-data-science-2026.zip`) di path yang sesuai (`/content/` untuk Colab), atau pastikan data sudah ada di `/kaggle/input/competitions/...`.
2. Jalankan **Bagian 1** untuk eksplorasi (opsional, tidak diperlukan untuk hasil akhir).
3. Jalankan **Bagian 2** dari awal — bagian ini *self-contained* (tidak bergantung variabel dari Bagian 1), termasuk instalasi `optuna`.
4. Output akhir: `submission.csv` di `/content/`.

## Dependencies

`pandas`, `numpy`, `matplotlib`, `seaborn`, `lightgbm`, `scikit-learn`, `optuna`

## Catatan Penting

- EDA di Bagian 1 memakai `train` mentah (belum di-despike) jadi beberapa insight soal outlier/spike sebaiknya dibaca sebagai catatan historis, bukan kondisi data yang dipakai model final (lihat poin #18 & bagian cleaning di Bagian 2).
- Fitur `nino_34` dan MJO sudah tersedia sampai periode test, sehingga bisa dipakai langsung tanpa forecasting tambahan.
