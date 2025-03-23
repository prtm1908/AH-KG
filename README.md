# Knowledge Graph API

A FastAPI service for creating and visualizing knowledge graphs from text documents using Stanford CoreNLP and Neo4j.

## Quick Setup

1. Clone and navigate to the repository:
```bash
git clone https://gitlab.com/aihello/llm-project.git
cd llm-project/KG-Project
git checkout newBranch
```

2. Create `.env` file with Neo4j credentials:
```env
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your_password
```

3. Run the services:
```bash
# Build and start
docker compose up --build

# Stop services
docker compose down
```

## Prerequisites

- Docker and Docker Compose
- At least 4GB of RAM available for CoreNLP service
- Internet connection for downloading dependencies

## API Endpoints

The API will be available at:
- `http://localhost:8000/create-knowledge-graph` - Create knowledge graph
- `http://localhost:8000/visualize` - Visualize knowledge graph

### 1. Create Knowledge Graph
```bash
curl -X POST "http://localhost:8000/create-knowledge-graph" \
     -H "Content-Type: application/json" \
     -d '{
           "text_url": "https://example.com/text.txt"
         }'
```

### 2. Visualize Knowledge Graph
```bash
curl -X POST "http://localhost:8000/visualize" \
     -H "Content-Type: application/json" \
     -d '{
           "text_url": "https://example.com/text.txt",
           "query": "your search query"
         }'
```