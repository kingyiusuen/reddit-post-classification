FROM public.ecr.aws/lambda/python:3.8

ARG MODEL_DIR=./artifacts
RUN mkdir $MODEL_DIR
COPY artifacts/model.pickle ${MODEL_DIR}/model.pickle

RUN pip install scikit-learn==0.24.2

COPY backend/lambda_function.py lambda_function.py
RUN python lambda_function.py
RUN chmod -R 0755 $MODEL_DIR

CMD ["lambda_function.lambda_handler"]