FROM python:3.9-slim

WORKDIR /app

# Install Java for PySpark
RUN apt-get update && apt-get upgrade -y && \
    apt-get install -y --no-install-recommends openjdk-17-jre-headless && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install PySpark and other dependencies
RUN pip install --no-cache-dir pyspark==3.2.0 pandas==1.3.3 numpy==1.21.2 redis==3.5.3

# Copy files
COPY . /app/

# Run the training script
CMD ["python", "train_model.py"]