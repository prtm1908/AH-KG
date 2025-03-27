from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Optional
from knowledge_graph_creation import create_triplets_spacy_fastcoref, process_triplets_with_lemmatization, upload_to_neo4j
from subgraph_retrieval import process_query_and_get_subgraph
import re

app = FastAPI(
    title="Knowledge Graph API",
    description="API for creating knowledge graphs and retrieving subgraphs",
    version="1.0.0"
)

class TextInput(BaseModel):
    text: str

class SubgraphQuery(BaseModel):
    query: str

class CombinedInput(BaseModel):
    text: str
    query: str

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
async def create_knowledge_graph(input_data: TextInput):
    """
    Create a knowledge graph from input text and store it in Neo4j.
    Processes text in batches of 1000 sentences.
    
    Args:
        input_data: TextInput containing the text to process
        
    Returns:
        Dictionary with success message and processing details
    """
    try:
        # Split text into batches
        batches = process_text_in_batches(input_data.text)
        
        # Process each batch
        for i, batch in enumerate(batches, 1):
            # Create triplets from batch
            triplets = create_triplets_spacy_fastcoref(batch)
            
            # Process triplets with lemmatization
            processed_triplets, relation_tracking = process_triplets_with_lemmatization(triplets)
            
            # Upload to Neo4j
            upload_to_neo4j(processed_triplets, relation_tracking)
        
        return {
            "status": "success",
            "message": f"Successfully processed {len(batches)} batches of text and uploaded to Neo4j"
        }
    except Exception as e:
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
        # Retrieve the relevant subgraph from the existing knowledge graph
        subgraph = process_query_and_get_subgraph(query_data.query)
        
        return subgraph
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/create-and-query", response_model=Dict[str, List[Dict[str, str]]])
async def create_and_query(input_data: CombinedInput):
    """
    Create a knowledge graph from input text and immediately query it.
    
    Args:
        input_data: CombinedInput containing both the text to create the graph and the query
        
    Returns:
        Dictionary containing both the created knowledge graph and the retrieved subgraph
    """
    try:
        # First create the knowledge graph
        await create_knowledge_graph(TextInput(text=input_data.text))
        
        # Then get the subgraph
        subgraph = await get_subgraph(SubgraphQuery(query=input_data.query))
        
        return {
            "subgraph": subgraph
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 