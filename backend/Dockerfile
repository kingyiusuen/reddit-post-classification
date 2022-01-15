FROM python:3.8-slim

ARG MODEL_DIR=./artifacts
RUN mkdir $MODEL_DIR
COPY artifacts/model.pickle ${MODEL_DIR}/model.pickle

RUN pip install flask==2.0.2 flask-cors==3.0.10 nltk==3.6.7 scikit-learn==0.24.2
RUN python -m nltk.downloader stopwords

COPY backend/app.py app.py
RUN chmod -R 0755 $MODEL_DIR

EXPOSE 8080
CMD ["python", "app.py"]
