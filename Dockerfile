FROM --platform=$TARGETPLATFORM python:3.9-slim

# Add build arguments for platform-specific decisions
ARG TARGETPLATFORM
ARG BUILDPLATFORM

# Install system dependencies including graphviz
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    default-jre \
    graphviz \
    graphviz-dev \
    python3-dev \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies
RUN python -m pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Download NLTK data
RUN python -c "import nltk; nltk.download('punkt'); nltk.download('averaged_perceptron_tagger'); nltk.download('averaged_perceptron_tagger_eng'); nltk.download('wordnet'); nltk.download('omw-1.4')"

# Download spaCy model
RUN python -m spacy download en_core_web_sm

# Copy application code
COPY . .

# Command to run the FastAPI server
CMD ["uvicorn", "graphSAGE_rag_visualization:app", "--host", "0.0.0.0", "--port", "8000"] 