# Sistem Rekomendasi - Python Workspace

## Setup Environment (Windows PowerShell)

```powershell
cd D:\MATERIIII\RIL_INI_PROPOSAL\sistem_rekomendasi
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Dependensi Utama

Dependensi proyek sudah didaftarkan di `requirements.txt`.
Catatan penting kompatibilitas model:
- `rf_wsn_multiclass_best_pipeline.pkl` disimpan dengan `scikit-learn 1.4.2`, jadi environment juga harus memakai versi itu.

## Menjalankan Script

### 1. Buat dataset testing dari file Excel

```powershell
.venv\Scripts\python.exe pythonver\buatfiletesting.py
```

Output sukses:

```text
Berhasil dibuat: D:\MATERIIII\RIL_INI_PROPOSAL\sistem_rekomendasi\dataset_labeled_wsn_testing.csv
```

### 2. Final model testing

```powershell
.venv\Scripts\python.exe pythonver\model_dev-readonly.py
```

Catatan:
- Script ini sekarang mencoba bootstrap otomatis dari file dataset + model walau tanpa variabel dari notebook.
- Script `pythonver\eksperimen_readonlynotrun..py` juga sudah membaca file dari root proyek.

## Troubleshooting

### Error: `ImportError: Missing optional dependency 'openpyxl'`

Penyebab:
- `pandas.read_excel(...)` membutuhkan `openpyxl`.

Solusi:

```powershell
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

### Error: `FileNotFoundError: File wajib tidak ditemukan ...`

Penyebab:
- `model_dev-readonly.py` dijalankan dari direktori yang salah, atau file dataset/model belum ada.

Solusi:
1. Pastikan command dijalankan dari folder proyek `sistem_rekomendasi` (root repo).
2. Pastikan file `dataset_labeled_wsn.xlsx` dan `rf_wsn_multiclass_best_pipeline.pkl` tersedia.

### Error: `AttributeError: 'Pipeline' object has no attribute 'transform_input'`

Penyebab:
- Versi `scikit-learn` saat load model berbeda dengan versi saat model disimpan.

Solusi:
1. Gunakan `requirements.txt` terbaru (versi `scikit-learn==1.4.2` dan `imbalanced-learn==0.12.2`).
2. Jalankan ulang install:

```powershell
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

### Error lama: `SyntaxError` pada `y_test,xxx`

Status:
- Sudah diperbaiki di `pythonver\model_dev-readonly.py`.

Jika masih muncul:
1. Pastikan file lokal sudah versi terbaru.
2. Jalankan ulang dengan environment `.venv` yang sama.
