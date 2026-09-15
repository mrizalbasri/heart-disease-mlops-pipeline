# Proyek Pengembangan Machine Learning Pipeline: Prediksi Risiko Penyakit Jantung (Heart Disease)

- **Nama:** M. Rizal Basri
- **Username Dicoding:** M_Rizal_Basri
- **Dataset:** UCI Heart Disease (Cleveland Dataset)
- **Framework:** TensorFlow Extended (TFX) dengan InteractiveContext
- **Target Penilaian:** ⭐⭐⭐⭐⭐ (Bintang 5 - Memenuhi Seluruh Ketentuan Wajib & Menerapkan Semua Saran)

---

## 1. Domain Proyek & Informasi Terkait Dataset

### Latar Belakang & Persoalan yang Ingin Diselesaikan
Penyakit kardiovaskular merupakan salah satu penyebab kematian tertinggi di dunia. Deteksi dini risiko penyakit jantung sangat penting untuk membantu tenaga medis mengambil tindakan preventif secara tepat waktu. Pendekatan konvensional seringkali memakan waktu dan membutuhkan banyak pengujian diagnostik terpisah. Melalui proyek ini, dikembangkan sebuah sistem machine learning end-to-end berbasis **Machine Learning Operations (MLOps)** menggunakan **TensorFlow Extended (TFX)** yang mampu memprediksi ada tidaknya risiko penyakit jantung pada pasien berdasarkan data profil klinis.

### Informasi Dataset
Dataset yang digunakan berasal dari *Heart Disease Dataset* (UCI Machine Learning Repository / Cleveland) yang memuat **303 data observasi pasien** dengan **13 variabel fitur klinis** dan **1 label target biner**:

| Nama Fitur | Tipe Data | Keterangan / Deskripsi |
| :--- | :--- | :--- |
| `age` | Numerik (Kontinu) | Usia pasien dalam satuan tahun |
| `sex` | Kategorikal (Diskrit) | Jenis kelamin (1 = Laki-laki; 0 = Perempuan) |
| `cp` | Kategorikal (Diskrit) | Tipe nyeri dada (Chest Pain Type: 0, 1, 2, 3) |
| `trestbps` | Numerik (Kontinu) | Tekanan darah saat istirahat (Resting Blood Pressure dalam mm Hg) |
| `chol` | Numerik (Kontinu) | Kadar kolesterol serum dalam mg/dl |
| `fbs` | Kategorikal (Diskrit) | Gula darah puasa > 120 mg/dl (1 = Benar; 0 = Salah) |
| `restecg` | Kategorikal (Diskrit) | Hasil elektrokardiogram saat istirahat (0, 1, 2) |
| `thalach` | Numerik (Kontinu) | Detak jantung maksimum yang dicapai |
| `exang` | Kategorikal (Diskrit) | Angina yang diinduksi oleh aktivitas fisik/olahraga (1 = Ya; 0 = Tidak) |
| `oldpeak` | Numerik (Kontinu) | Depresi ST yang diinduksi oleh latihan relatif terhadap istirahat |
| `slope` | Kategorikal (Diskrit) | Kemiringan puncak segmen ST saat latihan (0, 1, 2) |
| `ca` | Kategorikal (Diskrit) | Jumlah pembuluh darah utama (0-4) yang diwarnai fluoroskopi |
| `thal` | Kategorikal (Diskrit) | Thalassemia (1 = normal; 2 = fixed defect; 3 = reversable defect) |
| **`target`** | **Label (Diskrit)** | **Diagnosis penyakit jantung (1 = Memiliki risiko penyakit jantung; 0 = Normal / Sehat)** |

---

## 2. Business Understanding & Solusi Machine Learning

### Permasalahan (Problem Statements)
1. Bagaimana mengotomatisasi alur ingestion, validasi data, ekstraksi fitur, hingga pelatihan dan evaluasi model agar terhindar dari *data drift* dan *training-serving skew*?
2. Bagaimana mencari kombinasi hyperparameter terbaik secara otomatis untuk menghasilkan model dengan performa optimal?
3. Bagaimana mendistribusikan model terlatih ke dalam infrastruktur *model serving* (TensorFlow Serving & Docker) dan menguji keandalan *prediction request*?

### Solusi Machine Learning & Target yang Ingin Dicapai
- **Solusi Utama:** Membangun Machine Learning Pipeline menggunakan framework industri **TensorFlow Extended (TFX)** yang dijalankan secara interaktif dengan `InteractiveContext`.
- **Penerapan 3 Saran Bintang 5:**
  1. **Saran 1 (Tuner):** Mengintegrasikan komponen `Tuner` dengan pustaka `KerasTuner` untuk optimasi hyperparameter otomatis.
  2. **Saran 2 (Model Deployment):** Menyediakan berkas `Dockerfile` untuk deployment model menggunakan TensorFlow Serving serta menyertakan screenshot endpoint metadata model.
  3. **Saran 3 (Testing Notebook):** Menyediakan berkas `M_Rizal_Basri-testing.ipynb` untuk menguji prediction request ke model serving yang telah dibuat.
- **Target yang Dicapai:**
  - Pipeline mengotomatisasi seluruh alur kerja MLOps (10 komponen terintegrasi).
  - Model menghasilkan nilai akurasi $\ge 80\%$ dan validation AUC $\ge 0.85$.
  - Model lolos evaluasi keadilan (*fairness/slicing evaluation*) pada kelompok jenis kelamin (`sex`) serta lolos threshold validation (*BLESSED*).
  - Artefak SavedModel diekspor ke direktori serving siap produksi.

---

## 3. Metodologi Pipeline TFX & Pemrosesan Data

Pipeline terdiri dari 10 komponen terpadu:

```
[ CsvExampleGen ] -> Ingest CSV & Split Train/Eval (TFRecord)
       │
[ StatisticsGen ] -> Hitung Statistik Deskriptif (TFDV)
       │
  [ SchemaGen ]   -> Inferensi Skema Data Otomatis
       │
[ ExampleValidator ] -> Validasi & Deteksi Anomali Data
       │
  [ Transform ]   -> Rekayasa Fitur & Z-score Scaling (TFT)
       │
    [ Tuner ]     -> (Saran 1) Hyperparameter Tuning Otomatis (KerasTuner)
       │
   [ Trainer ]    -> Pelatihan Model DNN Keras dengan Best Hyperparameters
       │
  [ Resolver ]    -> Deteksi Baseline Blessed Model Sebelumnya
       │
  [ Evaluator ]   -> Evaluasi Metrik & Fairness Slicing Analysis (TFMA)
       │
   [ Pusher ]     -> Deployment Model ke Serving Directory
```

### Metode Pengolahan Data (Transform)
Menggunakan **TensorFlow Transform (TFT)** dengan modul `modules/heart_disease_transform.py`:
- **Fitur Numerik** (`age`, `trestbps`, `chol`, `thalach`, `oldpeak`): Dinormalisasi menggunakan standardisasi z-score (`tft.scale_to_z_score`), menghasilkan mean $\approx 0$ dan standar deviasi $\approx 1$.
- **Fitur Kategorikal** (`sex`, `cp`, `fbs`, `restecg`, `exang`, `slope`, `ca`, `thal`): Dikonversi ke tipe data integer terstandar (`tf.int64`).
- **Label Target** (`target`): Dikonversi menjadi `tf.int64`.
- **Pencegahan Training-Serving Skew:** Graf transformasi TFT diintegrasikan ke dalam serving signature model sehingga logika prapemrosesan inferensi identik dengan fase pelatihan.

---

## 4. Penerapan Saran Kriteria Bintang 5

### Saran 1: Hyperparameter Tuning Otomatis (`Tuner`)
- Diimplementasikan pada modul `modules/heart_disease_tuner.py` menggunakan algoritma `kt.RandomSearch`.
- Hyperparameter ruang pencarian:
  - `units_1`: 32, 64, 96, 128 (unit neuron Dense layer 1)
  - `units_2`: 16, 32, 48, 64 (unit neuron Dense layer 2)
  - `dropout`: 0.1, 0.2, 0.3 (dropout rate)
  - `learning_rate`: 0.01, 0.001, 0.0001 (Adam learning rate)
- Output `best_hyperparameters` diteruskan langsung ke komponen `Trainer` sehingga model akhir dilatih dengan konfigurasi terbaik (Trial terbaik meraih validation AUC $\approx 0.8558$).

### Saran 2: Model Deployment dengan TensorFlow Serving & Dockerfile
- Berkas `Dockerfile` disediakan pada root direktori:
  ```dockerfile
  FROM tensorflow/serving:latest
  COPY ./serving_model_dir /models/heart-disease-model
  ENV MODEL_NAME=heart-disease-model
  EXPOSE 8501 8080
  ```
- Perintah build dan menjalankan container:
  ```bash
  docker build -t heart-disease-serving .
  docker run -p 8080:8501 heart-disease-serving
  ```
- **Screenshot Endpoint Metadata Model Serving:**
  Telah dilampirkan berkas screenshot `serving_model_metadata.png` yang menampilkan respon JSON endpoint `http://localhost:8080/v1/models/heart-disease-model/metadata`:
  
  ![Serving Model Metadata](serving_model_metadata.png)

### Saran 3: Notebook Pengujian Prediction Request (`M_Rizal_Basri-testing.ipynb`)
- Disediakan berkas notebook khusus bernama **`M_Rizal_Basri-testing.ipynb`** sesuai ketentuan penamaan Dicoding (`<username_dicoding>-testing.ipynb`).
- Seluruh sel pada notebook telah dieksekusi tuntas, mencakup:
  1. Serialisasi data observasi pasien menjadi format biner `tf.train.Example`.
  2. Pengujian pemanggilan langsung signature `serving_default` dari SavedModel.
  3. Simulasi dan pengujian pengiriman payload JSON request REST API (`{"signature_name": "serving_default", "instances": [{"b64": "..."}]}`).
  4. Tabel perbandingan akurasi prediksi vs target aktual.

---

## 5. Evaluasi & Performa Model

### Metrik Evaluasi
Evaluasi dilakukan menggunakan **TensorFlow Model Analysis (TFMA)** dengan kriteria:
1. **BinaryAccuracy**: Mengukur proporsi prediksi kelas yang benar secara keseluruhan.
2. **AUC (Area Under the ROC Curve)**: Mengukur kemampuan separasi model dalam membedakan pasien berisiko vs normal.
3. **Confusion Matrix Metrics**: True Positives (TP), False Positives (FP), True Negatives (TN), False Negatives (FN).
4. **Slicing Specification**: Evaluasi performa menyeluruh (*overall*) dan evaluasi per subkelompok jenis kelamin (`sex: 0` / Perempuan, `sex: 1` / Laki-laki) untuk menjamin keadilan model (*fairness*).
5. **Threshold Validation:** Model wajib melampaui baseline minimum BinaryAccuracy $\ge 0.50$ agar mendapatkan status **BLESSED**.

### Hasil Evaluasi
- **Akurasi Pelatihan (Train):** $\approx 88.59\%$
- **Akurasi Evaluasi (Validation):** $\approx 80.00\% - 81.25\%$
- **Validation AUC:** $\approx 0.855 - 0.864$
- **Status Evaluator:** **BLESSED (Lolos Uji)**
- **Hasil Pusher:** Model yang berstatus *BLESSED* berhasil dipush secara otomatis ke direktori `serving_model_dir/` lengkap dengan `saved_model.pb`, assets, dan variables.

---

## 6. Struktur Direktori Proyek

```
M_Rizal_Basri-pipeline/
├── data/
│   └── heart.csv                       # Dataset klinis penyakit jantung
├── modules/
│   ├── heart_disease_transform.py      # Modul prapemrosesan TFT (preprocessing_fn)
│   ├── heart_disease_tuner.py          # Modul hyperparameter tuning (Saran 1)
│   ├── heart_disease_trainer.py        # Modul pelatihan model Keras (run_fn)
│   └── pipeline_module.py              # Modul gabungan pipeline
├── pipeline_root/                      # Direktori artefak dari seluruh komponen TFX
├── serving_model_dir/                  # Direktori tujuan deployment model yang lolos blessing
├── Dockerfile                          # Dockerfile untuk TensorFlow Serving (Saran 2)
├── serving_model_metadata.png          # Screenshot endpoint metadata model serving (Saran 2)
├── M_Rizal_Basri-testing.ipynb         # Notebook pengujian prediction request (Saran 3)
├── notebook.ipynb                      # Jupyter Notebook pipeline utama (seluruh cell telah dijalankan)
├── requirements.txt                    # Daftar dependensi proyek
└── README.md                           # Berkas dokumentasi komprehensif proyek
```

---

## 7. Panduan Menjalankan Proyek

1. **Persiapan Virtual Environment (Python 3.10):**
   ```bash
   python -m venv .venv
   .venv\Scripts\activate
   ```

2. **Instalasi Dependensi:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Menjalankan Pipeline:**
   Buka dan jalankan `notebook.ipynb` pada Jupyter Notebook atau VS Code. Seluruh sel akan menjalankan komponen TFX secara interaktif.

4. **Menjalankan Pengujian Serving:**
   Buka dan jalankan `M_Rizal_Basri-testing.ipynb` untuk menguji inferensi dan prediction request.
