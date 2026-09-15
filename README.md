# Proyek Pengembangan Machine Learning Pipeline: Prediksi Risiko Penyakit Jantung (Heart Disease)

- **Nama:** M. Rizal Basri
- **Username Dicoding:** rizalbasri
- **Pipeline Name:** rizalbasri-pipeline
- **Dataset:** UCI Heart Disease (Cleveland Dataset)
- **Framework:** TensorFlow Extended (TFX) dengan InteractiveContext

---

## 1. Informasi terkait Dataset yang Digunakan

### Sumber dan Karakteristik Dataset
Dataset yang digunakan dalam proyek ini adalah **Heart Disease Dataset** yang bersumber dari *UCI Machine Learning Repository (Cleveland Clinic Foundation)*. Dataset ini merupakan salah satu acuan standar dalam penelitian medis dan pembelajaran mesin untuk mendeteksi tanda-tanda awal risiko penyakit jantung.

- **Jumlah Sampel (Observasi):** 303 data pasien
- **Jumlah Atribut/Fitur:** 13 variabel fitur klinis (kombinasi data kontinu/numerik dan diskrit/kategorikal)
- **Label Target:** 1 kolom diagnosis biner (`target`)
- **Kondisi Kualitas Data:** Tidak terdapat *missing values* (nilai kosong), seluruh rentang nilai berada dalam domain klinis yang valid, dan distribusi label target terdistribusi secara seimbang ($\approx 54.5\%$ pasien berisiko dan $\approx 45.5\%$ pasien normal/sehat).

### Deskripsi Seluruh Fitur dan Target

| No | Nama Fitur | Tipe Data | Domain / Rentang | Keterangan Klinis |
| :-: | :--- | :--- | :--- | :--- |
| 1 | `age` | Numerik (Kontinu) | 29 – 77 tahun | Usia pasien dalam satuan tahun |
| 2 | `sex` | Kategorikal (Diskrit) | 0, 1 | Jenis kelamin (0 = Perempuan; 1 = Laki-laki) |
| 3 | `cp` | Kategorikal (Diskrit) | 0, 1, 2, 3 | Tipe nyeri dada (*Chest Pain Type*: 0 = typical angina, 1 = atypical angina, 2 = non-anginal pain, 3 = asymptomatic) |
| 4 | `trestbps` | Numerik (Kontinu) | 94 – 200 mm Hg | Tekanan darah saat istirahat (*resting blood pressure*) |
| 5 | `chol` | Numerik (Kontinu) | 126 – 564 mg/dl | Kadar kolesterol serum darah |
| 6 | `fbs` | Kategorikal (Diskrit) | 0, 1 | Gula darah puasa > 120 mg/dl (1 = Benar; 0 = Salah) |
| 7 | `restecg` | Kategorikal (Diskrit) | 0, 1, 2 | Hasil elektrokardiogram saat istirahat (0 = normal, 1 = memiliki gelombang ST-T abnormal, 2 = hipertrofi ventrikel kiri) |
| 8 | `thalach` | Numerik (Kontinu) | 71 – 202 bpm | Detak jantung maksimum yang dicapai saat uji beban |
| 9 | `exang` | Kategorikal (Diskrit) | 0, 1 | Angina yang diinduksi oleh aktivitas fisik/olahraga (1 = Ya; 0 = Tidak) |
| 10 | `oldpeak` | Numerik (Kontinu) | 0.0 – 6.2 | Depresi segmen ST yang diinduksi oleh latihan relatif terhadap istirahat |
| 11 | `slope` | Kategorikal (Diskrit) | 0, 1, 2 | Kemiringan puncak segmen ST saat latihan (0 = upsloping, 1 = flat, 2 = downsloping) |
| 12 | `ca` | Kategorikal (Diskrit) | 0 – 4 | Jumlah pembuluh darah utama yang diwarnai oleh fluoroskopi |
| 13 | `thal` | Kategorikal (Diskrit) | 0, 1, 2, 3 | Hasil tes Thalassemia (1 = normal, 2 = *fixed defect*, 3 = *reversible defect*) |
| 14 | **`target`** | **Label Biner (Diskrit)** | **0, 1** | **Diagnosis penyakit jantung (1 = Memiliki risiko penyakit jantung; 0 = Normal / Bebas risiko)** |

---

## 2. Informasi tentang Persoalan yang Ingin Diselesaikan

### Latar Belakang Permasalahan
Penyakit kardiovaskular merupakan penyebab kematian nomor satu di dunia menurut *World Health Organization (WHO)*. Sebagian besar kasus penyakit jantung dapat dicegah jika risiko terdeteksi pada fase awal. Namun, metode diagnostik konvensional seringkali memerlukan rangkaian prosedur medis yang memakan waktu, biaya tinggi, serta interpretasi manual yang rentan terhadap variabilitas subjektif tenaga medis.

Dalam domain teknik *Machine Learning Operations (MLOps)*, terdapat tantangan kritis saat membawa model prediktif dari fase eksperimen laboratorium ke tahap produksi:
1. **Pencegahan Anomali & Data Drift:** Data klinis pasien yang baru masuk sewaktu-waktu dapat mengalami pergeseran distribusi atau anomali tipe data yang menyebabkan penurunan performa model secara drastis (*concept drift*).
2. **Eliminasi Training-Serving Skew:** Prapemrosesan data (seperti normalisasi) yang ditulis terpisah antara fase pelatihan (*training*) dan fase produksi (*serving/inference*) seringkali memicu disparitas komputasi (*training-serving skew*).
3. **Keadilan dan Audit Model (Model Fairness):** Model medis wajib diuji secara objektif pada berbagai irisan demografis (misalnya kelompok gender/jenis kelamin) untuk memastikan model tidak diskriminatif atau bias terhadap subpopulasi tertentu.

### Pernyataan Masalah (*Problem Statements*)
1. Bagaimana mengotomatisasi alur *ingestion*, validasi skema, perhitungan statistik, dan deteksi anomali data klinis pasien agar sistem selalu menerima data yang valid dan bersih?
2. Bagaimana merancang arsitektur model *Deep Learning* yang dioptimasi secara otomatis menggunakan *hyperparameter tuning* guna menghasilkan akurasi prediksi risiko penyakit jantung yang tinggi dan andal?
3. Bagaimana mengaudit keadilan model (*fairness analysis*) pada subkelompok gender dan mendistribusikan model yang lolos evaluasi (*blessed*) ke dalam infrastruktur *serving* siap produksi (*TensorFlow Serving & Docker*)?

---

## 3. Penjelasan terkait Solusi Machine Learning yang Akan Dibuat beserta Target yang Ingin Dicapai

### Solusi Machine Learning yang Diusulkan
Solusi yang dibangun adalah **End-to-End Machine Learning Pipeline** berbasis framework industri **TensorFlow Extended (TFX)** yang dijalankan secara interaktif melalui `InteractiveContext`. Pipeline ini mencakup 10 komponen terintegrasi:

```
[ CsvExampleGen ]       -> Ingestion data CSV & pembagian split Train/Eval (TFRecord)
       │
[ StatisticsGen ]       -> Perhitungan ringkasan statistik deskriptif data (TFDV)
       │
  [ SchemaGen ]         -> Inferensi skema dan tipe data otomatis
       │
[ ExampleValidator ]    -> Validasi data terhadap skema & deteksi anomali/drift
       │
  [ Transform ]         -> Rekayasa fitur terpadu & Z-score scaling (TFT)
       │
    [ Tuner ]           -> (Saran 1) Hyperparameter Tuning Otomatis (KerasTuner)
       │
   [ Trainer ]          -> Pelatihan Deep Neural Network Keras dengan Best Hyperparameters
       │
   [ Resolver ]         -> Evaluasi komparatif terhadap Baseline Blessed Model
       │
  [ Evaluator ]         -> Evaluasi metrik & Fairness Slicing Analysis (TFMA)
       │
   [ Pusher ]           -> Otomasi deployment SavedModel ke direktori serving siap pakai
```

### Target yang Ingin Dicapai
1. **Target Teknis Performa:**
   - Model klasifikasi mencapai nilai akurasi evaluasi $\ge 80\%$ dan validation AUC $\ge 0.85$.
   - Model menghasilkan status **BLESSED** pada komponen `Evaluator` dengan melampaui baseline performa minimum (*threshold validation*).
   - Model terbukti adil (*fair*) tanpa bias signifikan antara kelompok pasien laki-laki (`sex = 1`) dan perempuan (`sex = 0`).
2. **Target Operasional & Serving:**
   - Logika transformasi fitur tertanam langsung di dalam signature inferensi SavedModel (`serving_default`) untuk mengeliminasi *training-serving skew*.
   - Model siap dideploy menggunakan container Docker berbasis TensorFlow Serving dan mampu melayani request inferensi via REST API.
3. **Penerapan Fitur Lanjutan Pipeline:**
   - Mengimplementasikan komponen `Tuner` dengan pustaka `KerasTuner` (`RandomSearch`) untuk hyperparameter tuning otomatis.
   - Menyediakan berkas `Dockerfile` deployment TensorFlow Serving beserta bukti screenshot endpoint metadata model.
   - Menyediakan berkas notebook pengujian prediction request `rizalbasri-testing.ipynb` yang tuntas dieksekusi.

---

## 4. Penjelasan tentang Metode Pengolahan Data, Arsitektur Model yang Digunakan, dan Metrik untuk Mengevaluasi Performa Model

### 4.1 Metode Pengolahan Data (Preprocessing & Feature Engineering)
Prapemrosesan data diimplementasikan pada modul `modules/heart_disease_transform.py` menggunakan **TensorFlow Transform (TFT)** melalui fungsi `preprocessing_fn`:
- **Standardisasi Z-Score pada Fitur Numerik:** Fitur kontinu (`age`, `trestbps`, `chol`, `thalach`, `oldpeak`) diskalakan menggunakan fungsi `tft.scale_to_z_score()`, menghasilkan distribusi dengan rata-rata (*mean*) $\approx 0$ dan standar deviasi $\approx 1$. Hal ini mencegah fitur dengan skala besar (seperti `chol` hingga 564) mendominasi gradien bobot model.
- **Konversi Fitur Kategorikal:** Fitur diskrit (`sex`, `cp`, `fbs`, `restecg`, `exang`, `slope`, `ca`, `thal`) dikonversi secara eksplisit ke dalam tipe data `tf.int64`.
- **Standarisasi Label Target:** Fitur target dikonversi menjadi tipe data `tf.int64` dengan penamaan `target_xf`.
- **Eliminasi Training-Serving Skew:** Transformasi TFT dikompilasi menjadi graf TensorFlow yang disematkan langsung ke dalam serving signature model melalui fungsi `_get_serve_tf_examples_fn`. Saat inferensi, model menerima data masukan mentah ter-serialisasi (`tf.train.Example`) dan secara otomatis menjalankan transformasi yang identik dengan proses pelatihan.

### 4.2 Arsitektur Model yang Digunakan
Model klasifikasi dibangun menggunakan **TensorFlow Keras Functional API** pada modul `modules/heart_disease_trainer.py`:
- **Lapisan Masukan (*Input Layers*):**
  - 5 input layer bertipe `tf.float32` untuk fitur numerik yang telah ditransformasi (`age_xf`, `trestbps_xf`, `chol_xf`, `thalach_xf`, `oldpeak_xf`).
  - 8 input layer bertipe `tf.int64` untuk fitur kategorikal yang kemudian di-cast menjadi `tf.float32`.
- **Lapisan Penggabungan (*Concatenation Layer*):** Seluruh representasi fitur digabungkan menjadi satu tensor vektor input terpadu (`tf.keras.layers.concatenate`).
- **Lapisan Tersembunyi (*Hidden Layers*):**
  - **Dense Layer 1:** Memiliki 64 unit neuron (dapat disesuaikan hingga 128 unit melalui Tuner) dengan fungsi aktivasi non-linear **ReLU** (*Rectified Linear Unit*).
  - **Dropout Layer 1:** Dropout rate sebesar 0.2 (ruang pencarian: 0.1 – 0.3) untuk mencegah *overfitting*.
  - **Dense Layer 2:** Memiliki 32 unit neuron (ruang pencarian: 16 – 64 unit) dengan aktivasi ReLU.
  - **Dropout Layer 2:** Dropout rate sebesar 0.2 untuk regularisasi tambahan.
- **Lapisan Keluaran (*Output Layer*):** Dense layer dengan 1 unit neuron dan fungsi aktivasi **Sigmoid** untuk memprediksi probabilitas keluaran biner $\hat{y} \in [0, 1]$.
- **Kompilasi Model:**
  - **Optimizer:** Adam dengan learning rate adaptif ($0.001$).
  - **Loss Function:** `binary_crossentropy`.
  - **Metrics:** `accuracy` dan `AUC` (*Area Under the ROC Curve*).
- **Hyperparameter Tuning Otomatis (Saran 1):** Komponen `Tuner` menggunakan algoritma `keras_tuner.RandomSearch` pada modul `modules/heart_disease_tuner.py` untuk menguji kombinasi neuron layer 1, neuron layer 2, dropout rate, dan learning rate, lalu mengalirkan konfigurasi hyperparameter terbaik langsung ke komponen `Trainer`.

### 4.3 Metrik untuk Mengevaluasi Performa Model
Evaluasi performa dilakukan secara menyeluruh menggunakan **TensorFlow Model Analysis (TFMA)** pada komponen `Evaluator`:
1. **BinaryAccuracy:** Mengukur persentase prediksi kelas risiko penyakit jantung yang benar secara keseluruhan terhadap diagnosis riil.
2. **AUC (Area Under the ROC Curve):** Mengukur kemampuan separasi model dalam membedakan pasien yang benar-benar berisiko vs pasien yang sehat di berbagai ambang batas klasifikasi.
3. **Confusion Matrix Metrics:** Evaluasi metrik *True Positives* (TP), *False Positives* (FP), *True Negatives* (TN), dan *False Negatives* (FN).
4. **Fairness Slicing Specification:** Evaluasi slicing TFMA dikonfigurasi untuk mengevaluasi performa model secara agregat (*overall*) serta pada subpopulasi jenis kelamin (`sex: 0` untuk perempuan dan `sex: 1` untuk laki-laki) guna memastikan tidak terjadi bias performa antargender.
5. **Threshold Validation & Model Blessing:** Model wajib melampaui nilai akurasi baseline minimum ($> 0.50$) dan tidak mengalami regresi performa dibanding baseline model terdahulu agar memperoleh predikat **BLESSED**.

---

## 5. Informasi terkait Performa Model Machine Learning yang Telah Dibuat

### Ringkasan Hasil Pelatihan dan Evaluasi
Berdasarkan eksekusi pipeline TFX yang tuntas dijalankan pada notebook `notebook.ipynb`:

| Parameter Evaluasi | Nilai yang Dicapai | Keterangan / Status |
| :--- | :---: | :--- |
| **Akurasi Pelatihan (*Train Accuracy*)** | **$pprox 88.59\%$** | Model mempelajari pola fitur klinis dengan sangat baik |
| **Akurasi Validasi (*Validation Accuracy*)** | **$pprox 81.25\%$** | Melampaui target minimum submission ($\ge 80\%$) |
| **Validation AUC** | **$pprox 0.8558 - 0.864$** | Kemampuan diskriminasi kelas klasifikasi sangat baik ($\ge 0.85$) |
| **Training Loss** | **$0.3341$** | Konvergensi loss stabil dengan regularisasi dropout |
| **Validation Loss** | **$0.4357$** | Generalisasi model baik tanpa indikasi *overfitting* berat |

### Hasil Analisis Slicing (Fairness Analysis pada Atribut `sex`)
Evaluasi TFMA menunjukkan performa yang adil dan konsisten pada setiap irisan kelompok data:
- **Slice Keseluruhan (*Overall*):** Akurasi mencapai $pprox 81.25\%$ dengan AUC $pprox 0.86$.
- **Slice Pasien Perempuan (`sex: 0`):** Model mencapai akurasi tinggi ($> 80\%$), membuktikan model mampu mendiagnosis pasien perempuan secara akurat.
- **Slice Pasien Laki-laki (`sex: 1`):** Model mempertahankan akurasi stabil ($> 80\%$) dan AUC yang kuat.

### Status Komponen Evaluator & Pusher
- **Status Evaluator:** **BLESSED (Lolos Uji)**. Model berhasil melampaui seluruh ambang batas validasi metrik dan disetujui untuk tahap produksi.
- **Hasil Pusher:** Komponen `Pusher` secara otomatis menyalin artefak SavedModel yang telah di-*bless* ke dalam direktori serving `serving_model_dir/` lengkap dengan `saved_model.pb`, direktori `variables/`, dan `assets/`.

---

## 6. Fitur Lanjutan dan Penerapan Komponen Tambahan

Proyek ini mengimplementasikan fitur-fitur lanjutan end-to-end MLOps sebagai berikut:

### 6.1 Hyperparameter Tuning Otomatis (`Tuner` + `KerasTuner`)
- Komponen `Tuner` diimplementasikan dengan modul `modules/heart_disease_tuner.py` menggunakan algoritma `kt.RandomSearch`.
- Hyperparameter yang dioptimasi secara otomatis:
  - `units_1`: Ukuran unit Dense layer 1 (32, 64, 96, 128)
  - `units_2`: Ukuran unit Dense layer 2 (16, 32, 48, 64)
  - `dropout`: Tingkat dropout rate (0.1, 0.2, 0.3)
  - `learning_rate`: Laju pembelajaran optimizer Adam (0.01, 0.001, 0.0001)
- Hasil konfigurasi hyperparameter terbaik (`best_hyperparameters`) dihubungkan secara otomatis ke komponen `Trainer` melalui `hyperparameters=tuner.outputs['best_hyperparameters']`.

### 6.2 Model Deployment Menggunakan TensorFlow Serving & Dockerfile
- Berkas `Dockerfile` disediakan pada direktori root proyek untuk mengemas model ke dalam container TensorFlow Serving:
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
- **Screenshot Respon Metadata Model Serving:**
  Telah dilampirkan berkas gambar tangkapan layar `serving_model_metadata.png` yang menampilkan respon JSON endpoint `http://localhost:8080/v1/models/heart-disease-model/metadata`:

  ![Serving Model Metadata](serving_model_metadata.png)

### 6.3 Notebook Pengujian Prediction Request (`rizalbasri-testing.ipynb`)
- Disediakan berkas notebook pengujian bernama **`rizalbasri-testing.ipynb`** sesuai ketentuan penamaan Dicoding (`<username_dicoding>-testing.ipynb`).
- Seluruh sel dalam notebook telah dieksekusi secara lengkap tanpa kesalahan, mencakup:
  1. Pengambilan sampel observasi pasien aktual dari `data/heart.csv` (Pasien berisiko vs Pasien normal).
  2. Serialisasi fitur masukan ke format biner standar `tf.train.Example`.
  3. Pengujian langsung pemanggilan signature `serving_default` dari SavedModel.
  4. Simulasi dan pengujian pengiriman payload JSON request inferensi REST API (`{"signature_name": "serving_default", "instances": [{"b64": "..."}]}`).
  5. Validasi probabilitas prediksi model terhadap diagnosis aktual klinis.

---

## 7. Struktur Direktori Proyek

Struktur direktori proyek `rizalbasri-pipeline` tersusun rapi sesuai standar submission Dicoding:

```
rizalbasri-pipeline/
├── data/
│   └── heart.csv                       # Dataset klinis penyakit jantung (UCI Cleveland)
├── modules/
│   ├── heart_disease_transform.py      # Modul prapemrosesan TFT (preprocessing_fn)
│   ├── heart_disease_tuner.py          # Modul hyperparameter tuning KerasTuner (Saran 1)
│   ├── heart_disease_trainer.py        # Modul pelatihan model Keras DNN (run_fn)
│   └── pipeline_module.py              # Modul gabungan fungsi pipeline
├── pipeline_root/                      # Direktori artefak dari seluruh komponen TFX & MLMD sqlite
├── serving_model_dir/                  # Direktori SavedModel produksi hasil ekspor Pusher
├── Dockerfile                          # Berkas Docker untuk TensorFlow Serving (Saran 2)
├── serving_model_metadata.png          # Screenshot endpoint metadata model serving (Saran 2)
├── rizalbasri-testing.ipynb            # Notebook pengujian prediction request (Saran 3)
├── notebook.ipynb                      # Jupyter Notebook pipeline utama TFX (seluruh sel telah dieksekusi)
├── requirements.txt                    # Daftar pustaka dan dependensi proyek
└── README.md                           # Dokumentasi komprehensif proyek
```

---

## 8. Panduan Menjalankan Proyek

### 1. Menyiapkan Lingkungan Virtual (Python 3.10)
```bash
python -m venv .venv
# Mengaktifkan virtual environment pada Windows:
.\.venv\Scripts\activate
# Atau pada Linux / macOS:
# source .venv/bin/activate
```

### 2. Memasang Dependensi
```bash
pip install -r requirements.txt
```

### 3. Menjalankan Pipeline Utama TFX
Buka berkas `notebook.ipynb` pada Jupyter Notebook atau VS Code, lalu jalankan seluruh sel secara berurutan. Pipeline akan mengeksekusi seluruh 10 komponen TFX secara interaktif mulai dari ingest data hingga model berhasil di-push ke `serving_model_dir/`.

### 4. Menjalankan Pengujian Model Serving
Buka berkas `rizalbasri-testing.ipynb` untuk menjalankan pengujian serialisasi `tf.train.Example`, pemanggilan signature SavedModel, dan simulasi REST API request TensorFlow Serving.