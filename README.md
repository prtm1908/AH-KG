# Knowledge Graph Creation with spaCy, FastCoref, and Stanford CoreNLP

This project creates knowledge graphs from text using a combination of:
- spaCy with FastCoref for coreference resolution
- Stanford CoreNLP for part-of-speech tagging and extracting nouns and verbs
- Neo4j or Nebula Graph for graph storage

## Setup

1. Clone this repository
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Download spaCy model:
   ```
   python -m spacy download en_core_web_sm
   ```
4. Create a `.env` file with your database credentials:
   ```
   # For Neo4j
   NEO4J_URI=bolt://localhost:7687
   NEO4J_USER=neo4j
   NEO4J_PASSWORD=password
   
   # For Nebula Graph
   NEBULA_HOST=localhost
   NEBULA_PORT=9669
   NEBULA_USER=root
   NEBULA_PASSWORD=nebula
   NEBULA_SPACE=knowledge_graph
   
   # Database type (neo4j, nebula, or both)
   DB_TYPE=neo4j
   ```

## Running with Docker

1. Start the services:
   ```
   docker-compose up -d
   ```
2. Wait for all services to start (especially Stanford CoreNLP)
3. Test the Stanford CoreNLP connection:
   ```
   python test_stanford_corenlp.py
   ```
4. Run the knowledge graph creation:
   ```
   python knowledge_graph_creation.py
   ```

## How It Works

The knowledge graph creation process is divided into sequential steps:

1. **Coreference Resolution**: The text is first processed with spaCy and FastCoref to resolve pronouns and other references.
2. **POS Tagging and Entity Extraction**: The resolved text is then sent to Stanford CoreNLP for part-of-speech tagging and extracting nouns and verbs.
3. **Triplet Creation**: Triplets are created from the extracted nouns and verbs.
4. **Lemmatization**: The triplets are processed to lemmatize the relations.
5. **Graph Storage**: The triplets are uploaded to Neo4j, Nebula Graph, or both.

This sequential approach allows for better control and debugging of each step in the process.

## API Endpoints

The API provides the following endpoints:

1. **Create Knowledge Graph** (`POST /create-knowledge-graph`):
   - **Request Body**:
     ```json
     {
       "file_path": "string",
       "is_url": boolean
     }
     ```
   - **Response**:
     ```json
     {
       "status": "success",
       "message": "string"
     }
     ```

2. **Get Subgraph** (`POST /get-subgraph`):
   - **Request Body**:
     ```json
     {
       "query": "string"
     }
     ```
   - **Response**:
     ```json
     [
       {
         "subject": "string",
         "relation": "string",
         "object": "string"
       }
     ]
     ```

3. **Create and Query** (`POST /create-and-query`):
   - **Request Body**:
     ```json
     {
       "file_path": "string",
       "is_url": boolean,
       "query": "string"
     }
     ```
   - **Response**:
     ```json
     {
       "subgraph": [
         {
           "subject": "string",
           "relation": "string",
           "object": "string"
         }
       ]
     }
     ```

## Customization

- To use only Neo4j, set `DB_TYPE=neo4j` in your `.env` file.
- To use only Nebula Graph, set `DB_TYPE=nebula` in your `.env` file.
- To use both, set `DB_TYPE=both` in your `.env` file.

## Troubleshooting

- If Stanford CoreNLP is not responding, make sure it's running and accessible at `http://localhost:9000`.
- If you're having issues with the database connection, check your credentials in the `.env` file.
- For Nebula Graph, make sure to create the space before running the script:
  ```
  docker exec -it console nebula-console -addr graphd -port 9669 -u root -p nebula -e 'CREATE SPACE IF NOT EXISTS knowledge_graph(vid_type=FIXED_STRING(128), partition_num=10, replica_factor=3);'
  ```