from RAG import (
    knowledge_graph_rag, 
    OpenIEKnowledgeGraph, 
    process_text_file, 
    extract_query_entities, 
    find_matching_nodes, 
    extract_relevant_triplets_from_entities
)
import networkx as nx
import matplotlib.pyplot as plt
import logging
import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, HttpUrl
from typing import Optional
import requests
import tempfile

# Set up logging with consistent format
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(title="Knowledge Graph API")

# Define request models
class VisualizationRequest(BaseModel):
    text_url: Optional[HttpUrl] = None
    file_path: Optional[str] = None
    query: str

    def __init__(self, **data):
        super().__init__(**data)
        if not self.text_url and not self.file_path:
            raise ValueError("Either text_url or file_path must be provided")

class KnowledgeGraphRequest(BaseModel):
    text_url: Optional[HttpUrl] = None
    file_path: Optional[str] = None

    def __init__(self, **data):
        super().__init__(**data)
        if not self.text_url and not self.file_path:
            raise ValueError("Either text_url or file_path must be provided")

def plot_networkx_graph(triplets, title="Knowledge Graph"):
    """Plot knowledge graph using NetworkX."""
    G = nx.MultiDiGraph()
    
    # Add nodes and edges from triplets
    for triplet in triplets:
        G.add_node(triplet['subject'])
        G.add_node(triplet['object'])
        G.add_edge(triplet['subject'], 
                  triplet['object'], 
                  relation=triplet['relation'],
                  strength=triplet['strength'])
    
    # Set up the plot
    plt.figure(figsize=(15, 10))
    pos = nx.spring_layout(G, k=1, iterations=50)
    
    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_color='lightblue', 
                          node_size=2000, alpha=0.7)
    
    # Draw edges
    nx.draw_networkx_edges(G, pos, edge_color='gray',
                          arrows=True, arrowsize=20)
    
    # Draw labels with smaller font and word wrapping
    labels = {}
    for node in G.nodes():
        # Wrap long node names
        words = node.split()
        wrapped_name = '\n'.join([' '.join(words[i:i+3]) for i in range(0, len(words), 3)])
        labels[node] = wrapped_name
    nx.draw_networkx_labels(G, pos, labels, font_size=8)
    
    # Draw edge labels with smaller font
    edge_labels = {}
    for u, v, data in G.edges(data=True):
        # Wrap long relation names
        relation = data['relation']
        words = relation.split()
        wrapped_relation = '\n'.join([' '.join(words[i:i+2]) for i in range(0, len(words), 2)])
        edge_labels[(u, v)] = f"{wrapped_relation}\n(s={data['strength']:.1f})"
    nx.draw_networkx_edge_labels(G, pos, edge_labels, font_size=6)
    
    plt.title(title)
    plt.axis('off')
    return plt

async def process_text_from_url(url: str) -> str:
    """Fetch text content from URL and save to temporary file for processing."""
    try:
        logger.info(f"Attempting to fetch content from: {url}")
        
        # Handle Google Drive URLs
        if 'drive.google.com' in url:
            # Check if it's already in the correct format
            if 'export=download' in url:
                file_id = url.split('id=')[1].split('&')[0] if 'id=' in url else None
            else:
                # Try to extract file ID from sharing URL
                try:
                    file_id = url.split('/d/')[1].split('/')[0]
                except IndexError:
                    file_id = None
            
            if file_id:
                # Use the direct download URL
                url = f"https://drive.google.com/uc?export=download&id={file_id}"
                logger.info(f"Converted to Google Drive direct URL: {url}")
            else:
                logger.warning("Could not extract file ID from Google Drive URL")
        
        response = requests.get(url, timeout=30)  # Add timeout
        response.raise_for_status()
        
        logger.info(f"Successfully fetched content, size: {len(response.text)} bytes")
        
        # Create a temporary file
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as temp_file:
            temp_file.write(response.text)
            temp_path = temp_file.name
            logger.info(f"Content saved to temporary file: {temp_path}")
            
        return temp_path
    except requests.Timeout:
        raise HTTPException(status_code=504, detail="Request timed out while fetching the URL")
    except requests.RequestException as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch text from URL: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")

async def process_and_visualize(text_url: Optional[str] = None, file_path: Optional[str] = None, query: Optional[str] = None):
    """Process text and create visualizations."""
    temp_file_path = None
    
    try:
        if text_url:
            # Fetch text from URL and save to temporary file
            logger.info("Fetching text from URL...")
            temp_file_path = await process_text_from_url(text_url)
            file_to_process = temp_file_path
        else:
            # Use the provided file path
            logger.info(f"Using local file path: {file_path}")
            if not os.path.exists(file_path):
                raise HTTPException(status_code=404, detail=f"File not found: {file_path}")
            file_to_process = file_path
        
        # Process text file and get triplets count
        logger.info("Processing text content...")
        triplets_count = process_text_file(file_to_process)
        
        if triplets_count == 0:
            raise HTTPException(status_code=400, detail="No triplets were extracted from the text")
        
        # Build knowledge graph with embeddings
        text_kg = OpenIEKnowledgeGraph()
        
        # Get relevant triplets using the existing knowledge graph
        logger.info("Processing query using RAG...")
        query_entities = extract_query_entities(query)
        matching_nodes = find_matching_nodes(text_kg, query_entities)
        relevant_triplets = extract_relevant_triplets_from_entities(text_kg, matching_nodes)
        
        # Plot the full knowledge graph using NetworkX
        logger.info("Plotting full knowledge graph...")
        full_graph = plot_networkx_graph(relevant_triplets, "Full Knowledge Graph")
        full_graph.savefig('full_knowledge_graph.png', bbox_inches='tight', dpi=300)
        logger.info("Full knowledge graph saved as 'full_knowledge_graph.png'")
        plt.close()  # Close the full graph figure
        
        # Plot the relevant subgraph using NetworkX
        if relevant_triplets:
            logger.info("Plotting relevant subgraph...")
            subgraph = plot_networkx_graph(relevant_triplets, "Relevant Knowledge Graph")
            subgraph.savefig('relevant_knowledge_graph.png', bbox_inches='tight', dpi=300)
            logger.info("Relevant subgraph saved as 'relevant_knowledge_graph.png'")
            plt.close()  # Close the subgraph figure
        else:
            logger.warning("No relevant triplets found for the query")
        
        return {
            "status": "success",
            "message": "Visualization complete!",
            "full_graph_path": "full_knowledge_graph.png",
            "relevant_graph_path": "relevant_knowledge_graph.png" if relevant_triplets else None,
            "triplets_count": triplets_count
        }
            
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Clean up temporary file only if it was created from URL
        if temp_file_path and os.path.exists(temp_file_path):
            os.unlink(temp_file_path)

@app.post("/visualize")
async def create_visualization(request: VisualizationRequest):
    """API endpoint to create knowledge graph visualizations."""
    return await process_and_visualize(
        text_url=str(request.text_url) if request.text_url else None,
        file_path=request.file_path,
        query=request.query
    )

@app.post("/create-knowledge-graph")
async def create_knowledge_graph(request: KnowledgeGraphRequest):
    """API endpoint to create a knowledge graph from text URL or local file path."""
    temp_file_path = None
    
    try:
        if request.text_url:
            # Fetch text from URL and save to temporary file
            logger.info("Fetching text from URL...")
            temp_file_path = await process_text_from_url(str(request.text_url))
            file_to_process = temp_file_path
        else:
            # Use the provided file path
            logger.info(f"Using local file path: {request.file_path}")
            if not os.path.exists(request.file_path):
                raise HTTPException(status_code=404, detail=f"File not found: {request.file_path}")
            file_to_process = request.file_path
        
        # Process text file and get triplets count
        logger.info("Processing text content...")
        triplets_count = process_text_file(file_to_process)
        
        if triplets_count == 0:
            raise HTTPException(status_code=400, detail="No triplets were extracted from the text")
        
        # Build knowledge graph with embeddings
        text_kg = OpenIEKnowledgeGraph()
        
        return {
            "status": "success",
            "message": "Knowledge graph created successfully!",
            "triplets_count": triplets_count,
            "nodes_count": len(text_kg.G.nodes()),
            "relationships_count": len(text_kg.G.edges())
        }
            
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Clean up temporary file only if it was created from URL
        if temp_file_path and os.path.exists(temp_file_path):
            os.unlink(temp_file_path)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 