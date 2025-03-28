from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Optional
from knowledge_graph_creation import create_triplets_spacy_fastcoref, process_triplets_with_lemmatization, upload_to_neo4j
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
    Create a knowledge graph from input text file and store it in Neo4j.
    Processes text in batches of 1000 sentences.
    
    Args:
        input_data: FileInput containing the path to the text file
        
    Returns:
        Dictionary with success message and processing details
    """
    try:
        print("Starting create_knowledge_graph function")
        # Clear the existing database first
        print("Clearing Neo4j database...")
        clear_neo4j_database()
        
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
            
            # Upload to Neo4j
            print("Uploading to Neo4j...")
            upload_to_neo4j(processed_triplets, relation_tracking)
            print("Successfully uploaded to Neo4j")
        
        print("Successfully completed all batches")
        return {
            "status": "success",
            "message": f"Successfully processed {len(batches)} batches of text and uploaded to Neo4j"
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