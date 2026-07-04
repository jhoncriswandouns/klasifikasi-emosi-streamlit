# Deployment Streamlit - Klasifikasi Emosi

## Struktur repository

```text
klasifikasi-emosi-streamlit/
├── app.py
├── requirements.txt
├── .streamlit/
│   └── config.toml
└── models/
    ├── cnn_model.keras
    └── mobilenetv2_model.keras
```

## File model yang dibutuhkan

Ambil dari hasil notebook:

```text
/content/hasil_proyek_emosi/models/cnn_model.keras
/content/hasil_proyek_emosi/models/mobilenetv2_model.keras
```

Salin kedua file ke folder `models/`.

## Uji lokal

Gunakan Python 3.11.

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

Linux/macOS:

```bash
source .venv/bin/activate
```

Lalu:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

## Deploy ke Streamlit Community Cloud

1. Buat repository GitHub.
2. Unggah semua file dan folder sesuai struktur.
3. Pastikan file `.keras` benar-benar berada di folder `models`.
4. Buka Streamlit Community Cloud.
5. Pilih **Create app**.
6. Pilih repository, branch `main`, dan entrypoint `app.py`.
7. Buka **Advanced settings** dan pilih Python 3.11.
8. Klik **Deploy**.

## Catatan

- Jangan melakukan normalisasi ulang di `app.py`; preprocessing telah tersimpan di model.
- Jika hanya satu model tersedia, aplikasi tetap berjalan dan hanya menampilkan model tersebut.
- Jika model dilatih ulang, perbarui nilai `MODEL_METRICS` di `app.py`.
