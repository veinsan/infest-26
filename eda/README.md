# EDA + Preprocessing — dataset baru INFEST XII

Run: `cd eda && ../.venv/bin/python 0X_*.py` (in order; later scripts read earlier outputs), then `cd prepro && ../.venv/bin/python 0X_*.py`.
Every script prints its numbers and saves figures in `outputs/<script>/`.

## Temuan utama

| # | Temuan (angka) | Script | Implikasi → tindakan |
|---|---|---|---|
| 1 | Holdout v1 0.976 vs LB 0.863. Train vs test **terpisah dengan mudah**: adversarial AUC 0.91 (meta+pixel), 0.90 (DINOv3) | 01, 03, 06 | Holdout random tidak memprediksi LB. Masalahnya distribution shift, bukan model. |
| 2 | Test = crop train-like **+ korupsi sintetis**: rotasi dengan edge-replicate (streak), segitiga abu semi-transparan, dash hitam, tint parchment + tinta warna, blur/pixelate, speckle, bar hitam, partial crop. Test: colourful 51% (train 3.6%), non-binary 72% (9%), blurry 69% (12%), bpp median 5.9 (0.98) | 02, 03 | Augmentasi harus **meniru korupsi test**, dan diterapkan SEBELUM preprocessing. |
| 3 | **Shortcut**: tanpa membaca aksara (shape + bpp + jpeg + pixel stats) → macro-F1 **0.873** (group CV). Shape saja 0.62. jawi/pegon/sunda 66–76% terbinarisasi, jawa 9% | 06 | Hilangkan sinyal non-aksara: crop ke tinta, tinggi tetap, normalisasi level, degradasi acak. |
| 4 | Web image (poster/tabel/foto buku): test 21% JPEG vs train 4.6%; non-line setelah preprocess 41% vs 21% | 01, 02, prepro/03 | Tidak bisa diatasi augmentasi → oversample / bobot gambar non-line di train. |
| 5 | Skew: |angle| ≥3° test 30% vs train 4%; p95 10°, p99 13° | 07 | RandomRotation ±15° (berdasarkan data). |
| 6 | Letterbox 160×640: bali/lontara tinggi tinta ~2 patch ViT (detail hilang); 224×224: 55% line < 1.5 patch | 07 | Tinggi tetap 160 + window/tiling lebar 640 (tinta ~10 patch). Jangan square resize. |
| 7 | Duplikat: exact di train 0; test identik train 2, near-copy 17. Cluster sumber sama (pixcorr ≥0.5, 99% label sama): 8.7% train, cluster terbesar 42. Split random v1: 8.3% val bocor | 04, 05 | StratifiedGroupKFold pada `dup_group`, weight 1/√size (`prepro/outputs/folds.csv`). DINOv3 cosine mentah tidak berguna untuk dedup (semua teks ≈0.97). |

## Preprocessing yang dicoba → rusak → diperbaiki (prepro/01–03)

| Percobaan | Teori | Hasil nyata | Perbaikan |
|---|---|---|---|
| RandomRotation fill hitam | aman | 87% gambar dapat "tinta" palsu (median 2.2×); garis tipis → >50% frame hitam → **polarity terbalik** (putih di atas hitam) | fill removal sebelum cek polaritas, deteksi segitiga (solidity + tanpa hole) → p95 fake ink 2.06× → 1.29× |
| Guard "≥50% tinta tersisa" | cegah rotasi merusak | lolos padahal teks hilang: streak edge-replicate dihitung sebagai tinta | ukur dengan rotasi *mask* tinta; 23% sudut acak ditolak (<60%) |
| Hapus bar hitam di border | aman untuk train | 2% train kehilangan >5% tinta (glyph kecil tebal, background gelap) | cek hole, gate untuk gambar kecil (max side <64) → 1.0% (sisa: foto dengan latar hitam, tidak masalah) |
| crop ke tinta | buang padding | halaman jarang + dash hitam → input **hitam penuh** | tolak crop jika densitas tinta >60% |
| ColorJitter sebelum preprocess | tutup gap warna | perubahan piksel 0.32/255 → **no-op** (tint sendiri juga 0.43) | tidak perlu; normalisasi gray+stretch sudah menutup gap warna |
| ColorJitter setelah preprocess (v1) | robust | bg turun ke 221 di 100% sampel, test hanya 0.2% bg <235 → **gap baru** | dihapus |
| Kalibrasi blur/occluder | — | domain AUC fitur canvas: clean 0.885 → aug 0.764 → 0.736 → **0.728** (line saja 0.706) | blur p=.45 σ 0.3–1.2, occluder p=.08 |

Frozen DINOv3-S + logistic regression, group 5-fold (prepro/04):

| | v1 pipeline | pipeline baru |
|---|---|---|
| F1 clean→clean (yang dilaporkan holdout) | 0.929 | 0.935 |
| F1 clean→corrupt (tanpa aug, dites seperti test ≈ LB drop v1) | 0.835 | 0.882 |
| F1 corrupt→corrupt (dengan aug) | 0.891 | **0.907** |
| adversarial AUC train-aug vs test | 0.827 | 0.840 |

Catatan: evaluasi "corrupt" memakai `corrupt()` sendiri (sirkular) — angka ini ranking, bukan prediksi LB. Clean→corrupt v1 (0.835) mereproduksi penurunan LB (0.863), jadi proxy ini masuk akal.

## Pipeline final — `prepro/pipeline.py`
- train: `train_view(path, rng)` = `corrupt` → `preprocess` → random window 160×640
- eval/test: `eval_views(path)` = `preprocess` → tile 160×640 (rata-rata logits antar tile)
- CV: `prepro/outputs/folds.csv` (fold, weight). Near-copy test: `prepro/outputs/test_train_twins.csv`.
