# Klasifikasi Jenis Sampah (Besi, Kertas, Plastik) dengan Transfer Learning

**Kelompok:** <<nama anggota>> | **Topik:** sampah (robot pemilah)

## 1. Dokumen Desain Awal
- **Masalah & tujuan:** Robot pemilah perlu mengenali jenis sampah dari citra kamera agar dapat memisahkannya ke wadah yang sesuai.
- **Kelas:** `besi` (benda logam seperti baut dan kunci), `kertas` (lembaran kertas atau tisu), `plastik` (botol, kemasan, kantong plastik).
- **Perangkat target:** <<Raspberry Pi / laptop>> → **Model dipilih:** MobileNetV3-Small, karena ringan (±2,5 juta parameter, ±0,06 GFLOPs) dan cocok untuk perangkat tanpa GPU.
- **Pengambilan data:** foto sendiri dengan kamera HP dan webcam laptop di atas meja, dengan variasi sudut, jarak, cahaya, dan latar. Jumlah per kelas: <<isi dari output make_metadata.py>>.
- **Preprocessing:** resize 224×224 → BGR→RGB → normalisasi (mean 0,485/0,456/0,406; std 0,229/0,224/0,225), identik saat training dan deployment (`common.py`). Koreksi distorsi kamera tidak diterapkan.
- **Metrik sukses:** akurasi test ≥ 90% dan latensi ≤ 100 ms per citra.

## 2. Dataset
`dataset_raw/` + `metadata.csv` (kolom: filename, label, sumber, split). Sumber data: foto sendiri. Pembagian data acak per citra dengan rasio 70/15/15 (train/val/test), seed 42.

| Kelas | Total foto | Train | Val | Test |
|---|---|---|---|---|
| besi | <<>> | <<>> | <<>> | <<>> |
| kertas | <<>> | <<>> | <<>> | <<>> |
| plastik | <<>> | <<>> | <<>> | <<>> |

## 3. Cara menjalankan
```bash
pip install -r requirements.txt
python make_metadata.py --source "foto sendiri"
python train.py --model mobilenet_v3_small --epochs 15
python benchmark_latency.py --ckpt results/model_fine_tuning.pt
```
Foto tambahan dapat diambil dengan `python capture.py <kelas>`.

## 4. Hasil 3 Mode
Model: MobileNetV3-Small, 15 epoch, optimizer Adam (learning rate 1e-3 untuk scratch dan feature extraction, 1e-4 untuk fine-tuning). Sumber: `results/hasil_3_mode.csv`.

| Mode | Trainable params | Best val acc | Test acc | Waktu latih (s) |
|---|---|---|---|---|
| Scratch | 1.520.931 | 0,3333 | 0,3333 | 26,3 |
| Feature extraction | 3.075 | 1,0000 | 0,8667 | 13,3 |
| Fine-tuning | 1.520.931 | 1,0000 | 0,9333 | 23,7 |

![Akurasi per epoch](results/akurasi_per_epoch.png)

## 5. Latensi Model Terpilih
Sumber: `results/latensi.csv` (preprocessing + inferensi per citra).
Model: MobileNetV3-Small (fine-tuning) | perangkat: <<CPU laptop / tipe>> | mean: <<>> ms | p95: <<>> ms | FPS: <<>>

## 6. Analisis Singkat
- **Mode terbaik:** Fine-tuning memberi test accuracy tertinggi (93,3%), disusul feature extraction (86,7%). Feature extraction hanya melatih 3.075 parameter dan paling cepat (13,3 detik), sehingga menjadi pilihan efisien bila sumber daya terbatas. Karena test set kecil (<<jumlah>> foto), selisih kedua mode setara <<1>> foto dan belum cukup untuk menyimpulkan mana yang lebih baik.
- **Scratch:** hanya mencapai 33,3%, setara tebakan acak untuk 3 kelas. Dengan data yang sedikit, model yang dilatih dari nol dengan 1,5 juta parameter tidak mampu belajar fitur yang berguna, sedangkan bobot ImageNet sudah membawa fitur umum (tepi, tekstur, bentuk). Ini sesuai teori transfer learning.
- **Data leakage:** akurasi validasi mencapai 100% sejak epoch-epoch awal pada dua mode pretrained. Penyebab yang mungkin adalah set validasi yang kecil dan foto mirip (objek sama, posisi hampir sama) yang masuk ke train dan val karena pembagian acak per citra. Karena itu test accuracy lebih layak dipercaya daripada val accuracy.
- **Kelas yang sering salah:** <<belum dianalisis per kelas karena skrip tidak membuat confusion matrix / isi jika Anda mengamatinya>>.
- **Latensi:** <<isi setelah benchmark: apakah ≤ 100 ms? bandingkan dengan target>>. Trade-off: fine-tuning akurasinya sedikit lebih tinggi, tetapi ukuran dan waktu latihnya lebih besar dibanding feature extraction; latensi inferensi keduanya sama karena arsitekturnya sama.
- **Keterbatasan & rencana perbaikan:** jumlah foto terbatas dan dibuat dalam satu sesi pengambilan. Perbaikan: menambah foto dan variasi cahaya serta latar, membagi data per sesi pengambilan, memperbesar test set, dan menambahkan confusion matrix.
