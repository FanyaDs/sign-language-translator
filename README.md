# Sign Language Translator

Project Python Computer Vision untuk mendeteksi gesture tangan dari webcam secara real-time, lalu menerjemahkannya menjadi teks di layar. Project ini dibuat agar cocok untuk portfolio fresh graduate karena memiliki alur end-to-end: pengumpulan dataset, training model, inference real-time, dan logging hasil prediksi.

## Fitur

- Membuka kamera laptop/webcam secara real-time.
- Mendeteksi tangan menggunakan MediaPipe Hands.
- Mengubah landmark tangan menjadi fitur numerik yang ringan.
- Melatih model klasifikasi gesture dengan Scikit-learn.
- Menampilkan hasil prediksi langsung di layar kamera.
- Mendukung tangan kanan dan kiri dengan mirror augmentation.
- Menyimpan riwayat prediksi stabil ke file CSV.
- Memiliki error handling jika kamera atau model belum tersedia.
- Bisa dijalankan di laptop tanpa GPU.

## Gesture Awal

Label gesture yang disarankan untuk dataset pertama:

- A
- B
- C
- I Love You
- Hello
- Thank You
- Yes
- No

Catatan: project ini menggunakan model yang dilatih dari dataset kamu sendiri. Jadi gesture di atas akan dikenali setelah kamu mengumpulkan data dan menjalankan training.

## Struktur Project

```text
sign-language-translator/
├── app.py
├── collect_dataset.py
├── train_model.py
├── predict.py
├── requirements.txt
├── README.md
├── dataset/
├── model/
└── logs/
```

## Instalasi

Masuk ke folder project:

```bash
cd sign-language-translator
```

Buat virtual environment:

```bash
python -m venv .venv
```

Aktifkan virtual environment di Windows:

```bash
.venv\Scripts\activate
```

Aktifkan virtual environment di macOS/Linux:

```bash
source .venv/bin/activate
```

Install dependency:

```bash
pip install -r requirements.txt
```

## Cara Mengumpulkan Dataset

Kumpulkan data untuk setiap gesture. Contoh:

```bash
python collect_dataset.py --label A --samples 120
python collect_dataset.py --label B --samples 120
python collect_dataset.py --label C --samples 120
python collect_dataset.py --label "I Love You" --samples 120
python collect_dataset.py --label Hello --samples 120
python collect_dataset.py --label "Thank You" --samples 120
python collect_dataset.py --label Yes --samples 120
python collect_dataset.py --label No --samples 120
```

Saat window kamera terbuka:

- Arahkan satu tangan ke kamera.
- Tekan `SPACE` untuk menyimpan satu sample.
- Tekan `Q` untuk keluar.

Mode otomatis juga tersedia:

```bash
python collect_dataset.py --label A --samples 120 --auto --interval 0.2
```

Dataset akan tersimpan di:

```text
dataset/gesture_dataset.csv
```

## Tips Dataset Agar Akurat

- Ambil minimal 80-150 sample per gesture.
- Gunakan variasi jarak tangan dari kamera.
- Gunakan variasi posisi tangan sedikit ke kiri, kanan, atas, dan bawah.
- Ambil data di beberapa kondisi cahaya.
- Jaga label tetap konsisten, misalnya gunakan `I Love You`, bukan kadang `ILY`.
- Kamu boleh mengumpulkan dataset dari satu tangan saja karena training otomatis membuat versi mirrored, tetapi dataset dari kedua tangan tetap bisa menambah variasi.
- Untuk gesture seperti `Hello`, `Thank You`, `Yes`, dan `No`, versi awal ini mempelajari pose tangan per frame. Pengembangan berikutnya bisa memakai sequence/time-series agar gerakan dinamis lebih natural.

## Training Model

Setelah dataset terkumpul, jalankan:

```bash
python train_model.py
```

Secara default, training menggandakan data menjadi versi mirrored agar gesture yang sama bisa dikenali dengan tangan kanan maupun kiri. Jika ingin mematikan fitur ini:

```bash
python train_model.py --no-mirror-augment
```

Output training:

```text
model/sign_language_model.pkl
model/training_report.txt
```

Jika dataset masih kecil, report akan memberi catatan bahwa evaluasi belum ideal. Tambahkan sample per label untuk hasil yang lebih stabil.

## Menjalankan Aplikasi Real-Time

```bash
python app.py
```

Jika window kamera terlalu besar atau teks terlihat kepotong, batasi ukuran tampilan:

```bash
python app.py --display-width 960 --display-height 540
```

Kontrol aplikasi:

- `SPACE`: menambahkan prediksi stabil ke teks.
- `BACKSPACE`: menghapus kata terakhir.
- `C`: membersihkan teks.
- `Q` atau `ESC`: keluar.

Aplikasi menampilkan `Hand: Kanan/Kiri` di panel kamera. Untuk hasil paling stabil, gunakan satu tangan saja di dalam frame pada satu waktu.

Riwayat prediksi stabil tersimpan di:

```text
logs/gesture_history.csv
```

## Jika Kamera Tidak Terdeteksi

Coba langkah berikut:

```bash
python app.py --camera 1
python collect_dataset.py --label A --camera 1
```

Jika masih gagal:

- Tutup aplikasi lain yang memakai kamera, seperti Zoom, Google Meet, atau OBS.
- Pastikan browser/aplikasi lain tidak sedang mengunci webcam.
- Periksa permission kamera di sistem operasi.
- Coba webcam eksternal.

## Cara Project Ini Ditulis di CV

Contoh bullet point CV:

```text
Developed a real-time Sign Language Translator using Python, OpenCV, MediaPipe Hands, and Scikit-learn, including webcam-based data collection, gesture classification training, live inference, and CSV prediction logging.
```

Versi bahasa Indonesia:

```text
Mengembangkan aplikasi real-time Sign Language Translator menggunakan Python, OpenCV, MediaPipe Hands, dan Scikit-learn, mencakup pengumpulan dataset dari webcam, training model klasifikasi gesture, inference langsung, dan logging hasil prediksi ke CSV.
```

## Ide Pengembangan Lanjutan

- Menambahkan lebih banyak label bahasa isyarat.
- Menggunakan model sequence untuk gesture dinamis.
- Menambahkan GUI dengan Streamlit atau PyQt.
- Menyimpan metrik training ke dashboard.
- Menambahkan mode kalibrasi untuk tangan kiri dan kanan.
- Export model ke format yang lebih ringan untuk deployment.
