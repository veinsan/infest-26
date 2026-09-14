## Overview

Indonesia adalah salah satu negara dengan kekayaan aksara tradisional terbesar di dunia. Aksara Jawa, Lontara, Jawi, Sunda, dan beberapa aksara daerah lainnya telah menjadi medium komunikasi dan dokumentasi selama berabad-abad, merekam sejarah, sastra, hingga pengetahuan lokal yang tidak tercatat dalam bahasa lain. Namun hari ini, sebagian besar aksara tersebut terancam punah karena jumlah penutur yang mampu membaca dan menulisnya terus menyusut setiap generasi, sementara ribuan manuskrip kuno masih tersimpan tanpa pernah berhasil didigitalisasi.

Kemampuan komputer untuk mengenali dan mengklasifikasikan citra aksara tradisional secara otomatis membuka peluang besar dalam pengembangan sistem digitalisasi manuskrip, transliterasi otomatis, hingga pembangunan arsip budaya digital berskala nasional. Penelitian di bidang klasifikasi aksara daerah masih sangat terbatas dibandingkan aksara Latin atau Han, sehingga kontribusi dari kompetisi ini memiliki nilai ilmiah yang nyata, bukan sekadar latihan teknis semata.

Dalam kompetisi Data Science INFEST XII 2026, peserta ditantang untuk membangun model klasifikasi gambar yang mampu mengidentifikasi asal daerah dari citra aksara tradisional Nusantara. Performa model dievaluasi menggunakan F1-Score Macro yang memastikan setiap aksara daerah berkontribusi setara terhadap skor akhir, tanpa ada kelas yang mendominasi penilaian hanya karena jumlah datanya lebih banyak.

### Submission

Peserta wajib mengumpulkan file CSV dengan dua kolom: `image_id` dan `label`. Kolom `image_id` berisi nama file gambar pada test set, dan kolom `label` berisi prediksi kelas aksara untuk setiap gambar. Contoh format submission yang benar:

```stylus
image_id,label
a3f8c1.png,Sunda
b92d4e.png,Jawa
7k1m9p.png,Batak
c45e2f.png,Lontara
d81b3a.png,Bali

```

content\_copy

File `sample_submission.csv` yang disediakan panitia sudah mengikuti format ini dan dapat digunakan sebagai template. Pastikan jumlah baris pada file submission sesuai dengan jumlah gambar pada test set.

### Evaluation

Performa model dievaluasi menggunakan **F1-Score Macro**, yaitu rata-rata F1-Score yang dihitung secara terpisah untuk setiap kelas kemudian dirata-ratakan tanpa pembobotan berdasarkan jumlah sampel per kelas. F1-Score sendiri merupakan harmonic mean dari precision dan recall yang diformulasikan sebagai berikut:

|   |
| - |

```
F1 = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}
```

|   |
| - |

```
\text{F1-Score Macro} = \frac{1}{N} \sum_{i=1}^{N} F1_i
```

Dengan metrik ini, setiap kelas aksara daerah berkontribusi setara terhadap skor akhir tanpa memandang jumlah sampelnya. Model yang hanya pintar memprediksi kelas mayoritas tidak akan mendapat keuntungan karena performa buruk di kelas minoritas akan langsung menarik skor keseluruhan ke bawah. Peserta diharapkan membangun model yang mampu mengenali seluruh kelas aksara secara merata, bukan hanya mengoptimalkan akurasi global.

## Dataset Description

Dataset yang digunakan dalam kompetisi ini berisi gambar aksara tradisional dari beberapa daerah di Indonesia. Setiap gambar merepresentasikan tulisan dalam satu jenis aksara daerah tertentu dengan label asal daerahnya.

Dataset dibagi menjadi dua bagian. Train set berisi gambar beserta label asal daerah aksara yang dapat digunakan peserta untuk melatih model. Test set berisi gambar tanpa label yang harus diprediksi oleh model peserta dan dikumpulkan melalui platform Kaggle.

## File yang tersedia

- **images/train/** - berisi seluruh gambar untuk pelatihan model
- **images/test/** - berisi seluruh gambar untuk prediksi
- **train.csv** - berisi dua kolom yaitu **image\_id** (nama file gambar pada folder train/) dan **label** (kelas aksara daerah asal gambar tersebut)
- **test.csv** - berisi kolom **image\_id** (nama file gambar pada folder test/)
- **sample\_submission.csv** - berisi format pengumpulan yang harus diikuti peserta dengan kolom **image\_id** dan **label**

# Rules

1. Kompetisi ini terbuka untuk tim yang telah terdaftar resmi melalui mekanisme pendaftaran INFEST XII 2026.
2. Setiap tim hanya diperbolehkan memiliki satu akun Kaggle yang aktif selama kompetisi berlangsung. Akun yang digunakan harus sesuai dengan email yang didaftarkan saat pendaftaran. Apabila terdapat perbedaan, peserta wajib melakukan konfirmasi dan revisi kepada Contact Person yang tertera, paling lambat H+2 setelah perilisan dataset.
3. Peserta wajib menggunakan bahasa pemrograman Python dalam pengerjaan solusi.
4. Peserta hanya diperbolehkan menggunakan dataset yang disediakan oleh panitia. Penggunaan data eksternal dari sumber manapun dalam bentuk apapun tidak diperbolehkan.
5. Peserta diperbolehkan menggunakan pretrained model sebagai base model dengan melakukan fine-tuning pada dataset yang disediakan.
6. Peserta tidak diperbolehkan menggunakan Large Language Model (LLM), model Generative AI, serta platform AutoML dalam bentuk apapun untuk menghasilkan prediksi. Penggunaan model tersebut sebagai referensi atau studi literatur diperbolehkan.
7. Peserta dilarang menggunakan teknik apa pun yang memanfaatkan prediksi dari model terhadap data tanpa label sebagai label tambahan untuk melatih ulang model, mencakup namun tidak terbatas pada pseudo labeling, self-training, label propagation, atau teknik semi-supervised learning sejenis lainnya. Model wajib dilatih hanya menggunakan data berlabel yang telah disediakan panitia pada split train, dan panitia akan melakukan pemeriksaan menyeluruh terhadap notebook untuk memastikan tidak ada penggunaan label dari sumber manapun selain train set yang disediakan dalam proses training. Pelanggaran terhadap aturan ini dapat berakibat pada diskualifikasi tim.
8. Peserta dapat melakukan submission maksimal 3 kali per hari. Sistem akan mengambil 3 submission terbaik sebagai dasar penilaian akhir.
9. Leaderboard selama kompetisi berlangsung menggunakan Public Score. Penentuan peringkat akhir menggunakan Private Score.
10. Hasil prediksi yang dikumpulkan harus merupakan output langsung dari model, bukan hasil pengisian manual. Pelanggaran terhadap ketentuan ini akan berakibat diskualifikasi.
11. Selain submission Kaggle, peserta wajib mengumpulkan notebook dalam format `.ipynb` melalui platform yang disediakan panitia paling lambat 26 September 2026. Peserta yang tidak mengumpulkan notebook akan dianggap gugur.
12. Notebook yang dikumpulkan harus sudah dijalankan seluruhnya tanpa error, menghasilkan output yang konsisten, dan disertai penjelasan menggunakan komponen markdown pada setiap tahapannya. Penjelasan dalam markdown wajib menggunakan bahasa Indonesia yang baik dan semi-formal.
13. Format penamaan Notebook adalah **NamaTim\_AsalUniversitas.ipynb**
14. Dilarang melakukan kerja sama, berbagi file notebook, maupun bertukar hasil prediksi dengan tim lain selama kompetisi berlangsung.
15. Apabila ditemukan indikasi plagiarisme, kerja sama antar tim, atau ketidaksesuaian signifikan antara submission Kaggle dan notebook yang dikumpulkan, panitia berhak melakukan investigasi dan mendiskualifikasi tim terkait.
16. Dataset yang disediakan hanya boleh digunakan untuk keperluan kompetisi, tidak diperkenankan untuk disebarluaskan atau digunakan untuk tujuan lain di luar kompetisi.