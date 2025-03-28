# Knowledge Graph API

A FastAPI-based service that creates knowledge graphs from text and allows querying them. The service uses spaCy for NLP processing, FastCoref for coreference resolution, and Neo4j for graph storage.

## Setup

1. Install the required dependencies:
```bash
pip install -r requirements.txt
```

2. If you have a CUDA-capable GPU, install CUDA-enabled PyTorch:
```bash
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```
Note: Replace `cu118` with your CUDA version (e.g., `cu117` for CUDA 11.7)

3. Download the required spaCy model:
```bash
python -m spacy download en_core_web_sm
```

4. Set up environment variables in a `.env` file:
```
NEO4J_URI=your_neo4j_uri
NEO4J_USER=your_neo4j_username
NEO4J_PASSWORD=your_neo4j_password
```

5. Start the FastAPI server:
```bash
python app.py
```

The server will start on `http://localhost:8000`

## API Endpoints

### 1. Create Knowledge Graph
- **Endpoint**: `POST /create-knowledge-graph`
- **Input**: JSON with `file_path` pointing to your text file
- **Description**: Creates a knowledge graph from the input text file and stores it in Neo4j
- **Example Request**:
```json
{
    "file_path": "path/to/your/text/file.txt"
}
```

### 2. Get Subgraph
- **Endpoint**: `POST /get-subgraph`
- **Input**: JSON with `query` string
- **Description**: Retrieves a relevant subgraph from the existing knowledge graph based on the query
- **Example Request**:
```json
{
    "query": "your query here"
}
```

### 3. Create and Query
- **Endpoint**: `POST /create-and-query`
- **Input**: JSON with `file_path` and `query`
- **Description**: Creates a knowledge graph from the input text file and immediately queries it
- **Example Request**:
```json
{
    "file_path": "path/to/your/text/file.txt",
    "query": "your query here"
}
```