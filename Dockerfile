FROM tensorflow/serving:latest

COPY ./serving_model_dir /models/heart-disease-model
ENV MODEL_NAME=heart-disease-model

EXPOSE 8501 8080
