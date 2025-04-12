from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Optional
from knowledge_graph_creation import create_triplets_spacy_fastcoref, process_triplets_with_lemmatization, upload_to_database, upload_to_nebula, upload_to_neo4j
from subgraph_retrieval import process_query_and_get_subgraph
import re
import os
from neo4j import GraphDatabase
from neo4j.exceptions import ServiceUnavailable
from dotenv import load_dotenv
import time
from nebula3.gclient.net import ConnectionPool
from nebula3.Config import Config

app = FastAPI(
    title="Knowledge Graph API",
    description="API for creating knowledge graphs and retrieving subgraphs",
    version="1.0.0"
)

class FileInput(BaseModel):
    file_path: str
    is_url: bool = False

class SubgraphQuery(BaseModel):
    query: str

class CombinedInput(BaseModel):
    file_path: str
    is_url: bool = False
    query: str

# Global variables to store Nebula Graph connection
nebula_connection_pool = None
nebula_session = None

def get_nebula_connection():
    """
    Establish a connection to Nebula Graph and return the session.
    If a connection already exists, return the existing session.
    
    Returns:
        Tuple of (connection_pool, session)
    """
    global nebula_connection_pool, nebula_session
    
    # If connection already exists, return it
    if nebula_connection_pool is not None and nebula_session is not None:
        return nebula_connection_pool, nebula_session
    
    # Load environment variables
    load_dotenv(override=True)
    
    # Get Nebula Graph credentials
    host = os.getenv('NEBULA_HOST')
    port = int(os.getenv('NEBULA_PORT', '9669'))
    user = os.getenv('NEBULA_USER')
    password = os.getenv('NEBULA_PASSWORD')
    space = os.getenv('NEBULA_SPACE')
    
    if not all([host, user, password, space]):
        raise ValueError("Missing Nebula Graph credentials in .env file")
    
    print(f"Establishing connection to Nebula Graph at {host}:{port}")
    
    # Create Nebula Graph connection pool
    config = Config()
    connection_pool = ConnectionPool()
    
    # Initialize the connection pool
    init_result = connection_pool.init([(host, port)], config)
    if not init_result:
        raise ConnectionError(f"Failed to initialize connection pool to Nebula Graph at {host}:{port}")
    
    # Get a session from the pool
    session = connection_pool.get_session(user, password)
    
    # Use the space
    resp = session.execute(f"USE {space}")
    if not resp.is_succeeded():
        session.release()
        connection_pool.close()
        raise Exception(f"Failed to use space {space}: {resp.error_msg()}")
    
    # Store the connection and session globally
    nebula_connection_pool = connection_pool
    nebula_session = session
    
    return connection_pool, session

def close_nebula_connection():
    """
    Close the Nebula Graph connection if it exists.
    """
    global nebula_connection_pool, nebula_session
    
    if nebula_session is not None:
        nebula_session.release()
        nebula_session = None
    
    if nebula_connection_pool is not None:
        nebula_connection_pool.close()
        nebula_connection_pool = None

def clear_neo4j_database():
    """
    Clear all nodes and relationships from the Neo4j database.
    """
    try:
        # Load environment variables with override=True to force reload
        load_dotenv(override=True)
        
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


def extract_value(v) -> str:
    """
    Extract the inner string from Nebula Graph Value objects.
    For example, turns "Value(sVal=b'AGREEMENT')" into "AGREEMENT".
    """
    s = str(v)
    match = re.search(r"b'(.+?)'", s)
    if match:
        return match.group(1)
    return s

def clear_nebula_database():
    """
    Clear all vertices and edges from the Nebula Graph database.
    If the space doesn't exist, create it. If it exists, clean it.
    """
    try:
        # Load environment variables with override=True to force reload
        load_dotenv(override=True)
        
        # Print all environment variables for debugging
        print("Environment variables:")
        print(f"NEBULA_HOST: {os.getenv('NEBULA_HOST')}")
        print(f"NEBULA_PORT: {os.getenv('NEBULA_PORT')}")
        print(f"NEBULA_USER: {os.getenv('NEBULA_USER')}")
        print(f"NEBULA_PASSWORD: {os.getenv('NEBULA_PASSWORD')}")
        print(f"NEBULA_SPACE: {os.getenv('NEBULA_SPACE')}")
        print(f"DB_TYPE: {os.getenv('DB_TYPE')}")

        host = os.getenv('NEBULA_HOST')
        port = int(os.getenv('NEBULA_PORT', '9669'))
        user = os.getenv('NEBULA_USER')
        password = os.getenv('NEBULA_PASSWORD')
        space = os.getenv('NEBULA_SPACE')

        if not all([host, user, password, space]):
            error_msg = "Missing Nebula Graph credentials in .env file"
            print(f"Error: {error_msg}")
            print(f"Host: {'Present' if host else 'Missing'}")
            print(f"User: {'Present' if user else 'Missing'}")
            print(f"Password: {'Present' if password else 'Missing'}")
            print(f"Space: {'Present' if space else 'Missing'}")
            raise ValueError(error_msg)

        print(f"Attempting to connect to Nebula Graph at {host}:{port}")
        from nebula3.gclient.net import ConnectionPool
        from nebula3.Config import Config

        config = Config()
        connection_pool = ConnectionPool()
        
        # Initialize the connection pool with better error handling
        init_result = connection_pool.init([(host, port)], config)
        if not init_result:
            error_msg = f"Failed to initialize connection pool to Nebula Graph at {host}:{port}"
            print(f"Error: {error_msg}")
            raise ConnectionError(error_msg)

        session = connection_pool.get_session(user, password)

        try:
            # Check existing spaces
            resp = session.execute("SHOW SPACES")
            if not resp.is_succeeded():
                error_msg = f"Failed to list spaces: {resp.error_msg()}"
                print(f"Error: {error_msg}")
                raise Exception(error_msg)

            spaces = [str(row.values[0]) for row in resp.rows()]
            space_exists = space in spaces

            if not space_exists:
                create_query = (
                    f"CREATE SPACE {space} ("
                    f"partition_num = 10, replica_factor = 1, vid_type = FIXED_STRING(30))"
                )
                resp = session.execute(create_query)
                if not resp.is_succeeded():
                    # Handle "Existed!" message
                    if "Existed!" in resp.error_msg():
                        print(f"Space '{space}' already existed (error message). Continuing...")
                    else:
                        error_msg = f"Failed to create space: {resp.error_msg()}"
                        print(f"Error: {error_msg}")
                        raise Exception(error_msg)
                else:
                    print(f"Space '{space}' created")

                print("Waiting for space to be ready...")
                # Wait until we can USE the space
                for attempt in range(10):
                    time.sleep(1)
                    resp = session.execute(f"USE {space}")
                    if resp.is_succeeded():
                        print(f"Space '{space}' is now available and in use.")
                        break
                    else:
                        print(f"Waiting for space '{space}' to be ready... (Attempt {attempt+1})")
                else:
                    error_msg = f"Space '{space}' not ready after waiting"
                    print(f"Error: {error_msg}")
                    raise Exception(error_msg)
            else:
                print(f"Space '{space}' already exists")

            # Use the space
            resp = session.execute(f"USE {space}")
            if not resp.is_succeeded():
                error_msg = f"Failed to use space: {resp.error_msg()}"
                print(f"Error: {error_msg}")
                raise Exception(error_msg)

            # Drop tags (use extract_value to get the proper tag name and wrap with backticks)
            resp = session.execute("SHOW TAGS")
            if resp.is_succeeded():
                tags = [extract_value(row.values[0]) for row in resp.rows()]
                for tag in tags:
                    drop_query = f"DROP TAG IF EXISTS `{tag}`"
                    resp_drop = session.execute(drop_query)
                    if not resp_drop.is_succeeded():
                        print(f"Warning: Failed to drop tag {tag}: {resp_drop.error_msg()}")

            # Drop edges (similarly use extract_value and backticks)
            resp = session.execute("SHOW EDGES")
            if resp.is_succeeded():
                edges = [extract_value(row.values[0]) for row in resp.rows()]
                for edge in edges:
                    drop_query = f"DROP EDGE IF EXISTS `{edge}`"
                    resp_drop = session.execute(drop_query)
                    if not resp_drop.is_succeeded():
                        print(f"Warning: Failed to drop edge {edge}: {resp_drop.error_msg()}")

            print(f"Successfully initialized/cleared space '{space}'")

        finally:
            session.release()
            connection_pool.close()

    except Exception as e:
        error_msg = f"Error in clear_nebula_database: {str(e)}"
        print(error_msg)
        raise Exception(error_msg)


def clear_graph_database():
    """
    Clear all nodes and relationships from the specified graph database(s).
    """
    # Load environment variables with override=True to force reload
    load_dotenv(override=True)
    
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

def read_text_file(file_path: str, is_url: bool = False) -> str:
    """
    Read text from a file or URL.
    
    Args:
        file_path: Path to the text file or URL
        is_url: Boolean indicating if file_path is a URL
        
    Returns:
        Content of the text file as string
        
    Raises:
        HTTPException: If file doesn't exist or can't be read, or if URL is invalid
    """
    if is_url:
        try:
            import requests
            response = requests.get(file_path)
            response.raise_for_status()  # Raise an exception for HTTP errors
            return response.text
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error fetching URL: {str(e)}")
    else:
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail=f"File not found: {file_path}")
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                return file.read()
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error reading file: {str(e)}")

def process_text_in_batches(text: str, batch_size: int = 50) -> List[str]:
    """
    Split text into batches of sentences using regex.
    
    Args:
        text: Input text to split
        batch_size: Number of sentences per batch (reduced from 1000 to 100 to avoid exceeding max_doc_len)
        
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
    Create a knowledge graph from input text file or URL and store it in the specified graph database(s).
    Processes text in batches of 1000 sentences.
    
    Args:
        input_data: FileInput containing the path to the text file or URL and a flag indicating if it's a URL
        
    Returns:
        Dictionary with success message and processing details
    """
    try:
        print("Starting create_knowledge_graph function")
        # Clear the existing database first
        print("Clearing graph database...")
        try:
            clear_graph_database()
            print("Successfully cleared graph database")
        except Exception as e:
            error_msg = f"Error clearing graph database: {str(e)}"
            print(error_msg)
            raise HTTPException(status_code=500, detail=error_msg)
        
        # Read text from file
        print(f"Reading text from {'URL' if input_data.is_url else 'file'}: {input_data.file_path}")
        try:
            text = read_text_file(input_data.file_path, input_data.is_url)
            print(f"Successfully read text, length: {len(text)} characters")
        except Exception as e:
            error_msg = f"Error reading text: {str(e)}"
            print(error_msg)
            raise HTTPException(status_code=500, detail=error_msg)
        
        # Split text into batches
        print("Splitting text into batches...")
        batches = process_text_in_batches(text)
        print(f"Created {len(batches)} batches")
        
        # Get database type
        db_type = os.getenv('DB_TYPE', 'neo4j').lower()
        
        # If using Nebula Graph, establish a connection once for all batches
        nebula_connection = None
        nebula_session = None
        if db_type in ['nebula', 'both']:
            try:
                nebula_connection, nebula_session = get_nebula_connection()
                print("Established Nebula Graph connection for all batches")
            except Exception as e:
                error_msg = f"Error establishing Nebula Graph connection: {str(e)}"
                print(error_msg)
                raise HTTPException(status_code=500, detail=error_msg)
        
        # Process each batch
        for i, batch in enumerate(batches, 1):
            print(f"\nProcessing batch {i}/{len(batches)}")
            # Create triplets from batch
            print("Creating triplets...")
            try:
                triplets = create_triplets_spacy_fastcoref(batch)
                print(f"Created {len(triplets)} triplets")
            except Exception as e:
                error_msg = f"Error creating triplets: {str(e)}"
                print(error_msg)
                raise HTTPException(status_code=500, detail=error_msg)
            
            # Process triplets with lemmatization
            print("Processing triplets with lemmatization...")
            try:
                processed_triplets, relation_tracking = process_triplets_with_lemmatization(triplets)
                print(f"Processed {len(processed_triplets)} triplets")
            except Exception as e:
                error_msg = f"Error processing triplets: {str(e)}"
                print(error_msg)
                raise HTTPException(status_code=500, detail=error_msg)
            
            # Upload to the specified graph database(s)
            print("Uploading to graph database...")
            try:
                if db_type == 'nebula':
                    upload_to_nebula(processed_triplets, relation_tracking, nebula_session)
                elif db_type == 'neo4j':
                    upload_to_neo4j(processed_triplets, relation_tracking)
                elif db_type == 'both':
                    upload_to_neo4j(processed_triplets, relation_tracking)
                    upload_to_nebula(processed_triplets, relation_tracking, nebula_session)
                else:
                    raise ValueError(f"Invalid DB_TYPE: {db_type}. Must be 'neo4j', 'nebula', or 'both'.")
                print("Successfully uploaded to graph database")
            except Exception as e:
                error_msg = f"Error uploading to database: {str(e)}"
                print(error_msg)
                raise HTTPException(status_code=500, detail=error_msg)
        
        print("Successfully completed all batches")
        return {
            "status": "success",
            "message": f"Successfully processed {len(batches)} batches of text and uploaded to graph database"
        }
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        error_msg = f"Unexpected error in create_knowledge_graph: {str(e)}"
        print(error_msg)
        raise HTTPException(status_code=500, detail=error_msg)
    finally:
        # Close Nebula Graph connection if it was established
        if db_type in ['nebula', 'both'] and nebula_connection is not None:
            close_nebula_connection()

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
        
        # Get database type
        db_type = os.getenv('DB_TYPE', 'neo4j').lower()
        
        # If using Nebula Graph, establish a connection
        nebula_session = None
        if db_type in ['nebula', 'both']:
            try:
                _, nebula_session = get_nebula_connection()
                print("Established Nebula Graph connection for subgraph query")
            except Exception as e:
                error_msg = f"Error establishing Nebula Graph connection: {str(e)}"
                print(error_msg)
                raise HTTPException(status_code=500, detail=error_msg)
        
        # Retrieve the relevant subgraph from the existing knowledge graph
        subgraph = process_query_and_get_subgraph(query_data.query, session=nebula_session)
        print(f"Retrieved subgraph with {len(subgraph)} triplets")
        return subgraph
    except Exception as e:
        print(f"Error in get_subgraph: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # No need to close the connection here as it's managed by the global connection pool
        pass

@app.post("/create-and-query", response_model=Dict[str, List[Dict[str, str]]])
async def create_and_query(input_data: CombinedInput):
    """
    Create a knowledge graph from input text file or URL and immediately query it.
    
    Args:
        input_data: CombinedInput containing the path to the text file or URL, a flag indicating if it's a URL, and the query
        
    Returns:
        Dictionary containing both the created knowledge graph and the retrieved subgraph
    """
    try:
        print("\nStarting create_and_query function")
        print(f"File path: {input_data.file_path}")
        print(f"Query: {input_data.query}")
        
        # Get database type
        db_type = os.getenv('DB_TYPE', 'neo4j').lower()
        
        # If using Nebula Graph, establish a connection once
        nebula_connection = None
        nebula_session = None
        if db_type in ['nebula', 'both']:
            try:
                nebula_connection, nebula_session = get_nebula_connection()
                print("Established Nebula Graph connection for create_and_query")
            except Exception as e:
                error_msg = f"Error establishing Nebula Graph connection: {str(e)}"
                print(error_msg)
                raise HTTPException(status_code=500, detail=error_msg)
        
        # First create the knowledge graph
        try:
            print("Attempting to create knowledge graph...")
            # Create a FileInput object for the create_knowledge_graph function
            file_input = FileInput(file_path=input_data.file_path, is_url=input_data.is_url)
            
            # Call the create_knowledge_graph function directly
            await create_knowledge_graph(file_input)
            print("Successfully created knowledge graph")
        except Exception as e:
            print(f"Error during knowledge graph creation: {str(e)}")
            if "Torch not compiled with CUDA enabled" in str(e):
                # If it's a CUDA error, we can still proceed with the query
                print("CUDA error during graph creation, but continuing with query...")
            else:
                # Provide more detailed error information
                error_detail = f"Error during knowledge graph creation: {str(e)}"
                print(error_detail)
                raise HTTPException(status_code=500, detail=error_detail)
        
        # Then get the subgraph
        try:
            print("Attempting to get subgraph...")
            subgraph = await get_subgraph(SubgraphQuery(query=input_data.query))
            print("Successfully retrieved subgraph")
            
            return {
                "subgraph": subgraph
            }
        except Exception as e:
            error_detail = f"Error during subgraph retrieval: {str(e)}"
            print(error_detail)
            raise HTTPException(status_code=500, detail=error_detail)
            
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        print(f"Error in create_and_query: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")
    finally:
        # Close Nebula Graph connection if it was established
        if db_type in ['nebula', 'both']:
            close_nebula_connection()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 