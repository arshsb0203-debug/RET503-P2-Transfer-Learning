# Pilihan Topik Proyek (pilih satu, kodenya sama)

## 1. Jenis sampah (robot pemilah)
Kelas: `organik`, `plastik`, `kertas`, `logam`
Data: foto sampah asli di atas meja/lantai dari sudut kamera robot; variasikan cahaya & latar.
Model: MobileNetV3-Small (Raspberry Pi) atau ResNet-18 (praktikum).

## 2. Kematangan buah (robot pemanen)
Kelas: `mentah`, `setengah_matang`, `matang`, `busuk`
Data: satu jenis buah (mis. tomat/pisang), foto beda hari agar warna berubah, beda pencahayaan.
Model: MobileNetV3-Large (edge menengah).

## 3. Rintangan di jalur (robot navigasi)
Kelas: `jalur_bebas`, `rintangan_statis`, `orang`, `tangga_atau_lubang`
Data: rekam video dari kamera robot lalu ekstrak frame (`cv2.VideoCapture`, ambil 1 frame/0,5 detik).
Model: EfficientNet-B0 (Jetson) atau MobileNetV3-Small.

## 4. Rambu (robot patuh lalu lintas)
Kelas: `stop`, `kiri`, `kanan`, `lurus`, `tidak_ada`
Data: cetak rambu mini lalu foto dari berbagai jarak/sudut; atau dataset publik (catat sumber di metadata).
Model: MobileNetV3-Large / ResNet-18.

## Tips umum
- Minimal 50 citra per kelas; target 80-100 agar akurasi stabil.
- Ambil foto dengan kamera robot yang sama dengan saat deployment.
- Jangan ambil foto hampir identik berurutan (burst) — split train/test jadi bocor.
