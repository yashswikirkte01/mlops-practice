FROM tensorflow/tensorflow:2.15.0-gpu

WORKDIR /app

# Install metrics and plotting dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy execution script
COPY evaluate.py .

ENTRYPOINT ["python", "evaluate.py"]