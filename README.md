# Submission 1: Machine Learning Pipeline - Heart Disease Prediction
Nama: M. Rizal Basri

Username dicoding: rizalbasri

| | Deskripsi |
| ----------- | ----------- |
| Dataset | [Heart Disease Dataset](https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset) (UCI Cleveland) berisi 303 data pasien dengan 13 atribut klinis dan 1 label target biner diagnosis risiko penyakit jantung. |
| Masalah | Mendeteksi risiko penyakit jantung sejak dini serta membangun pipeline MLOps otomatis untuk mengatasi potensi data drift, mencegah training-serving skew, dan menjaga keadilan prediksi model. |
| Solusi machine learning | Membangun end-to-end Machine Learning Pipeline menggunakan TensorFlow Extended (TFX) dengan 10 komponen: CsvExampleGen, StatisticsGen, SchemaGen, ExampleValidator, Transform, Tuner, Trainer, Resolver, Evaluator, dan Pusher. |
| Metode pengolahan | Menggunakan TensorFlow Transform (TFT) dengan penskalaan Z-score (tft.scale_to_z_score) untuk fitur numerik serta konversi tipe data fitur kategorikal dan label target ke tf.int64. |
| Arsitektur model | Model Deep Neural Network (DNN) berbasis Keras menggunakan hyperparameter terbaik dari Tuner: Dense 96 unit (ReLU), Dropout 0.1, Dense 16 unit (ReLU), Dropout 0.1, Dense Output 1 unit (Sigmoid), optimizer Adam (learning rate 0.01), dan loss binary_crossentropy. |
| Metrik evaluasi | Dievaluasi dengan TensorFlow Model Analysis (TFMA) menggunakan metrik BinaryAccuracy (threshold >= 0.50), AUC, Loss, Confusion Matrix, serta fairness slicing pada fitur jenis kelamin (sex). |
| Performa model | Status Evaluator: BLESSED (Lolos).<br>- Overall: Binary Accuracy = 80.37% (0.8037), AUC = 0.8105, Loss = 1.3181.<br>- Slice Perempuan (sex: 0): Binary Accuracy = 90.91% (0.9091), AUC = 0.8210, Loss = 0.7564.<br>- Slice Laki-laki (sex: 1): Binary Accuracy = 75.68% (0.7568), AUC = 0.7832, Loss = 1.5686. |

---

## Fitur Tambahan (Saran Bintang 5)
1. **Hyperparameter Tuning (Tuner)**: Menggunakan KerasTuner RandomSearch pada `modules/heart_disease_tuner.py`, menghasilkan konfigurasi terbaik: `{"units_1": 96, "units_2": 16, "dropout": 0.1, "learning_rate": 0.01}`.
2. **Model Deployment (Docker & TF Serving)**: Tersedia `Dockerfile` untuk deployment model via TensorFlow Serving dan tangkapan layar endpoint metadata `serving_model_metadata.png`.
3. **Notebook Pengujian (`rizalbasri-testing.ipynb`)**: Notebook pengujian inferensi SavedModel dan request REST API TensorFlow Serving yang tuntas dieksekusi.

---

## Cara Menjalankan
1. Aktifkan virtual environment dan install dependensi:
   ```bash
   python -m venv .venv
   .\.venv\Scripts\activate
   pip install -r requirements.txt
   ```
2. Jalankan seluruh sel pada `notebook.ipynb` untuk mengeksekusi pipeline TFX.
3. Jalankan `rizalbasri-testing.ipynb` untuk menguji inferensi model.
