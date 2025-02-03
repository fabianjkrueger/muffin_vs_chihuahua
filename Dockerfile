FROM python:3.9-slim

WORKDIR /app

# install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# copy requirements and install dependencies
COPY api/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# copy application code and model
COPY api/app ./app
COPY models ./models

# expose port
EXPOSE 8000

# run with uvicorn
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
