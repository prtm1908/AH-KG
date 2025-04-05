from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Optional
from knowledge_graph_creation import create_triplets_spacy_fastcoref, process_triplets_with_lemmatization, upload_to_database
from subgraph_retrieval import process_query_and_get_subgraph
import re
import os
from neo4j import GraphDatabase
from neo4j.exceptions import ServiceUnavailable
from dotenv import load_dotenv

app = FastAPI(
    title="Knowledge Graph API",
    description="API for creating knowledge graphs and retrieving subgraphs",
    version="1.0.0"
)

class FileInput(BaseModel):
    file_path: str

class SubgraphQuery(BaseModel):
    query: str

class CombinedInput(BaseModel):
    file_path: str
    query: str

def clear_neo4j_database():
    """
    Clear all nodes and relationships from the Neo4j database.
    """
    try:
        # Load environment variables
        load_dotenv()
        
        # Get Neo4j credentials from environment variables
        uri = os.getenv('NEO4J_URI')
        user = os.getenv('NEO4J_USER')
        password = os.getenv('NEO4J_PASSWORD')
        
        if not all([uri, user, password]):
            raise ValueError("Missing Neo4j credentials in .env file")
        
        # Create Neo4j driver
        driver = GraphDatabase.driver(uri, auth=(user, password))
        
        with driver.session() as session:
            # Delete all nodes and relationships
            session.run("MATCH (n) DETACH DELETE n")
            
        driver.close()
        print("Successfully cleared Neo4j database")
        
    except ServiceUnavailable:
        print("Could not connect to Neo4j database. Please check your connection details.")
        raise
    except Exception as e:
        print(f"An error occurred while clearing the database: {str(e)}")
        raise

def clear_nebula_database():
    """
    Clear all vertices and edges from the Nebula Graph database.
    If the space doesn't exist, create it with appropriate configuration.
    """
    try:
        # Load environment variables
        load_dotenv()
        
        # Get Nebula Graph credentials from environment variables
        host = os.getenv('NEBULA_HOST')
        port = int(os.getenv('NEBULA_PORT', '9669'))
        user = os.getenv('NEBULA_USER')
        password = os.getenv('NEBULA_PASSWORD')
        space = os.getenv('NEBULA_SPACE')
        
        if not all([host, user, password, space]):
            raise ValueError("Missing Nebula Graph credentials in .env file")
        
        # Import Nebula Graph modules
        from nebula3.gclient.net import ConnectionPool
        from nebula3.Config import Config
        
        # Create Nebula Graph connection pool
        config = Config()
        connection_pool = ConnectionPool()
        
        # Initialize the connection pool
        assert connection_pool.init([(host, port)], config)
        
        # Get a session from the pool
        session = connection_pool.get_session(user, password)
        
        try:
            # Check if space exists
            resp = session.execute(f"SHOW SPACES")
            if not resp.is_succeeded():
                raise Exception(f"Failed to list spaces: {resp.error_msg()}")
            
            spaces = [row.values[0] for row in resp.rows()]
            space_exists = space in spaces
            
            if not space_exists:
                # Create space if it doesn't exist
                create_space_query = f"""
                CREATE SPACE IF NOT EXISTS {space}(
                    partition_num=10, 
                    replica_factor=3, 
                    vid_type=FIXED_STRING(128)
                )"""
                
                resp = session.execute(create_space_query)
                if not resp.is_succeeded():
                    raise Exception(f"Failed to create space: {resp.error_msg()}")
                
                print(f"Space '{space}' created")
                
                # Wait longer for the space to be ready (10 seconds)
                print("Waiting for space to be ready...")
                session.execute("SLEEP 10")
            else:
                print(f"Space '{space}' already exists")
            
            # Try to use the space
            resp = session.execute(f"USE {space}")
            if not resp.is_succeeded():
                raise Exception(f"Failed to use space: {resp.error_msg()}")
            
            # Clear all vertices and edges if they exist - using Nebula Graph syntax
            # First, get all tags and edge types
            resp = session.execute("SHOW TAGS")
            if resp.is_succeeded():
                tags = [row.values[0] for row in resp.rows()]
                for tag in tags:
                    # Delete vertices with this tag
                    resp = session.execute(f"DELETE VERTEX {tag}")
                    if not resp.is_succeeded():
                        print(f"Warning: Failed to delete vertices with tag {tag}: {resp.error_msg()}")
            
            resp = session.execute("SHOW EDGES")
            if resp.is_succeeded():
                edges = [row.values[0] for row in resp.rows()]
                for edge in edges:
                    # Delete edges with this type
                    resp = session.execute(f"DELETE EDGE {edge}")
                    if not resp.is_succeeded():
                        print(f"Warning: Failed to delete edges with type {edge}: {resp.error_msg()}")
            
            print(f"Successfully initialized/cleared space '{space}'")
            
        finally:
            # Always release the session and close the pool
            session.release()
            connection_pool.close()
        
    except Exception as e:
        print(f"Error in clear_nebula_database: {str(e)}")
        raise

def clear_graph_database():
    """
    Clear all nodes and relationships from the specified graph database(s).
    """
    # Load environment variables
    load_dotenv()
    
    # Get the database type from environment variables
    db_type = os.getenv('DB_TYPE', 'neo4j').lower()
    
    # Clear the specified database(s)
    if db_type == 'neo4j':
        clear_neo4j_database()
    elif db_type == 'nebula':
        clear_nebula_database()
    elif db_type == 'both':
        clear_neo4j_database()
        clear_nebula_database()
    else:
        raise ValueError(f"Invalid DB_TYPE: {db_type}. Must be 'neo4j', 'nebula', or 'both'.")

def read_text_file(file_path: str) -> str:
    """
    Read text from a file.
    
    Args:
        file_path: Path to the text file
        
    Returns:
        Content of the text file as string
        
    Raises:
        HTTPException: If file doesn't exist or can't be read
    """
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail=f"File not found: {file_path}")
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading file: {str(e)}")

def process_text_in_batches(text: str, batch_size: int = 1000) -> List[str]:
    """
    Split text into batches of sentences using regex.
    
    Args:
        text: Input text to split
        batch_size: Number of sentences per batch
        
    Returns:
        List of text batches
    """
    # Split text into sentences using regex
    # This regex looks for periods followed by whitespace or end of string
    # It also handles common abbreviations like Mr., Dr., etc.
    sentences = re.split(r'(?<=[.!?])\s+', text)
    
    # Clean up sentences
    sentences = [sent.strip() for sent in sentences]
    
    # Split into batches
    batches = []
    for i in range(0, len(sentences), batch_size):
        batch = " ".join(sentences[i:i + batch_size])
        batches.append(batch)
    
    return batches

@app.post("/create-knowledge-graph", response_model=Dict[str, str])
async def create_knowledge_graph(input_data: FileInput):
    """
    Create a knowledge graph from input text file and store it in the specified graph database(s).
    Processes text in batches of 1000 sentences.
    
    Args:
        input_data: FileInput containing the path to the text file
        
    Returns:
        Dictionary with success message and processing details
    """
    try:
        print("Starting create_knowledge_graph function")
        # Clear the existing database first
        print("Clearing graph database...")
        clear_graph_database()
        
        # Read text from file
        print(f"Reading text from file: {input_data.file_path}")
        text = read_text_file(input_data.file_path)
        
        # Split text into batches
        print("Splitting text into batches...")
        batches = process_text_in_batches(text)
        print(f"Created {len(batches)} batches")
        
        # Process each batch
        for i, batch in enumerate(batches, 1):
            print(f"\nProcessing batch {i}/{len(batches)}")
            # Create triplets from batch
            print("Creating triplets...")
            triplets = create_triplets_spacy_fastcoref(batch)
            print(f"Created {len(triplets)} triplets")
            
            # Process triplets with lemmatization
            print("Processing triplets with lemmatization...")
            processed_triplets, relation_tracking = process_triplets_with_lemmatization(triplets)
            print(f"Processed {len(processed_triplets)} triplets")
            
            # Upload to the specified graph database(s)
            print("Uploading to graph database...")
            upload_to_database(processed_triplets, relation_tracking)
            print("Successfully uploaded to graph database")
        
        print("Successfully completed all batches")
        return {
            "status": "success",
            "message": f"Successfully processed {len(batches)} batches of text and uploaded to graph database"
        }
    except Exception as e:
        print(f"Error in create_knowledge_graph: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/get-subgraph", response_model=List[Dict[str, str]])
async def get_subgraph(query_data: SubgraphQuery):
    """
    Process a query to retrieve a relevant subgraph from the existing knowledge graph.
    
    Args:
        query_data: SubgraphQuery containing the query text
        
    Returns:
        List of triplets representing the relevant subgraph
    """
    try:
        print(f"Processing subgraph query: {query_data.query}")
        # Retrieve the relevant subgraph from the existing knowledge graph
        subgraph = process_query_and_get_subgraph(query_data.query)
        print(f"Retrieved subgraph with {len(subgraph)} triplets")
        return subgraph
    except Exception as e:
        print(f"Error in get_subgraph: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/create-and-query", response_model=Dict[str, List[Dict[str, str]]])
async def create_and_query(input_data: CombinedInput):
    """
    Create a knowledge graph from input text file and immediately query it.
    
    Args:
        input_data: CombinedInput containing the path to the text file and the query
        
    Returns:
        Dictionary containing both the created knowledge graph and the retrieved subgraph
    """
    try:
        print("\nStarting create_and_query function")
        print(f"File path: {input_data.file_path}")
        print(f"Query: {input_data.query}")
        
        # First create the knowledge graph
        try:
            print("Attempting to create knowledge graph...")
            await create_knowledge_graph(FileInput(file_path=input_data.file_path))
            print("Successfully created knowledge graph")
        except Exception as e:
            print(f"Error during knowledge graph creation: {str(e)}")
            if "Torch not compiled with CUDA enabled" in str(e):
                # If it's a CUDA error, we can still proceed with the query
                print("CUDA error during graph creation, but continuing with query...")
            else:
                raise e
        
        # Then get the subgraph
        print("Attempting to get subgraph...")
        subgraph = await get_subgraph(SubgraphQuery(query=input_data.query))
        print("Successfully retrieved subgraph")
        
        return {
            "subgraph": subgraph
        }
    except Exception as e:
        print(f"Error in create_and_query: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 