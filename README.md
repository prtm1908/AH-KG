# Knowledge Graph API

A FastAPI-based service that creates knowledge graphs from text and allows querying them. The service uses spaCy for NLP processing, FastCoref for coreference resolution, and supports both Neo4j and Nebula Graph for storage.

## Setup

1. Set up environment variables in a `.env` file:
```
# Database Type (required)
DB_TYPE=neo4j  # Options: 'neo4j', 'nebula', or 'both'

# Neo4j Configuration (required if DB_TYPE is 'neo4j' or 'both')
NEO4J_URI=your_neo4j_uri
NEO4J_USER=your_neo4j_username
NEO4J_PASSWORD=your_neo4j_password

# Nebula Graph Configuration (required if DB_TYPE is 'nebula' or 'both')
NEBULA_HOST=your_nebula_host
NEBULA_PORT=9669  # Default port, can be changed
NEBULA_USER=your_nebula_username
NEBULA_PASSWORD=your_nebula_password
NEBULA_SPACE=your_nebula_space

# Nebula Graph Service Configuration (optional)
NEBULA_META_SERVER_ADDRS=metad0:9559,metad1:9559,metad2:9559  # Meta server addresses
NEBULA_LOCAL_IP=graphd  # Local IP for graphd service
NEBULA_WS_IP=graphd  # WebSocket IP for graphd service
NEBULA_PORT=9669  # Graphd service port
NEBULA_WS_HTTP_PORT=19669  # WebSocket HTTP port
NEBULA_LOG_DIR=/logs  # Log directory
```

2. Build and start the services:
```bash
# Build the services first
docker compose build

# Then start them
docker compose up
```

The API will be available at `http://localhost:8000`

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