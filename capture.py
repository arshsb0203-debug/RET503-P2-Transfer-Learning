"""Ambil foto dari webcam untuk dataset_raw/<kelas>/.

Pemakaian:  python capture.py <kelas>
Contoh:     python capture.py besi
            python capture.py kertas
            python capture.py plastik

Tombol: SPASI = simpan foto | Q atau ESC = selesai
Ubah posisi/sudut/cahaya objek di antara tiap foto, jangan ambil foto yang hampir sama.
"""
import os
import sys
import time

import cv2

CAM_INDEX = 0  # ganti ke 1 atau 2 jika kamera tidak terbuka
ROOT = "dataset_raw"


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    kelas = sys.argv[1]
    out_dir = os.path.join(ROOT, kelas)
    os.makedirs(out_dir, exist_ok=True)

    cap = cv2.VideoCapture(CAM_INDEX, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(CAM_INDEX)
    if not cap.isOpened():
        print("Kamera tidak bisa dibuka. Coba ubah CAM_INDEX di capture.py.")
        sys.exit(1)

    n = len([f for f in os.listdir(out_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))])
    print(f"Kelas '{kelas}': sudah ada {n} foto. SPASI=simpan, Q=keluar.")
    saved = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            print("Gagal membaca frame.")
            break
        view = frame.copy()
        cv2.putText(view, f"{kelas}: {n + saved} foto  [SPASI=simpan, Q=keluar]",
                    (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
        cv2.imshow("capture", view)
        key = cv2.waitKey(1) & 0xFF
        if key == 32:
            name = f"{kelas}_{time.strftime('%Y%m%d_%H%M%S')}_{saved + 1:03d}.jpg"
            cv2.imwrite(os.path.join(out_dir, name), frame)
            saved += 1
            print(f"simpan {name}  (total {n + saved})")
        elif key in (ord("q"), 27):
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"Selesai. Total foto '{kelas}': {n + saved}")


if __name__ == "__main__":
    main()
