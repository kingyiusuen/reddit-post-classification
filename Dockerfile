# Base image
FROM tiangolo/uvicorn-gunicorn-fastapi:python3.7

# Install dependencies
COPY setup.py setup.py
COPY requirements.txt requirements.txt
COPY Makefile Makefile
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && make install \
    && apt-get purge -y --auto-remove build-essential

# Copy only the relevant directories
COPY reddit_post_classification reddit_post_classification
COPY backend backend
COPY artifacts artifacts

# Expose is NOT supported by Heroku
# EXPOSE 5000

# Run the app. CMD is required to run on Heroku
# $PORT is set by Heroku
CMD uvicorn backend.api:app --host 0.0.0.0 --port $PORT