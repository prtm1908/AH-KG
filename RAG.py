import subprocess
import os
import networkx as nx
from knowledge_graph import OpenIEKnowledgeGraph, process_text_file
import tempfile
import logging
import spacy

# Set up logging with consistent format
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

def get_corenlp_server():
    """Get the CoreNLP server URL from environment or use default."""
    return os.environ.get('CORENLP_SERVER', 'http://corenlp:9000')

def get_nodes_at_depth_two(G, start_node, max_depth=2):
    """Get all nodes and edges within 2 hops of the start node."""
    nodes_at_depth = {0: {start_node}}
    all_nodes = {start_node}
    all_edges = []
    
    # BFS to depth 2
    for depth in range(max_depth):
        nodes_at_depth[depth + 1] = set()
        for node in nodes_at_depth[depth]:
            # Get outgoing edges
            out_edges = G.out_edges(node, data=True)
            for u, v, data in out_edges:
                nodes_at_depth[depth + 1].add(v)
                all_nodes.add(v)
                all_edges.append((u, v, data))
            
            # Get incoming edges
            in_edges = G.in_edges(node, data=True)
            for u, v, data in in_edges:
                nodes_at_depth[depth + 1].add(u)
                all_nodes.add(u)
                all_edges.append((u, v, data))
    
    return all_nodes, all_edges

def extract_query_entities(query: str):
    """Extract entities from the query using spaCy."""
    doc = nlp(query)
    entities = set()
    
    # Get named entities
    for ent in doc.ents:
        entities.add(ent.text.lower())
    
    # Get noun chunks as potential entities
    for chunk in doc.noun_chunks:
        entities.add(chunk.text.lower())
    
    # Get individual nouns if they're not part of chunks
    for token in doc:
        if token.pos_ in ['NOUN', 'PROPN'] and token.text.lower() not in entities:
            entities.add(token.text.lower())
    
    logger.info(f"Extracted entities from query: {entities}")
    return entities

def find_matching_nodes(text_kg: OpenIEKnowledgeGraph, query_entities: set):
    """Find nodes in the knowledge graph that match query entities."""
    matching_nodes = set()
    
    # Get all nodes and their names from the graph
    node_names = nx.get_node_attributes(text_kg.G, 'name')
    
    for node, name in node_names.items():
        # Convert name to lowercase for comparison
        name_lower = name.lower()
        
        # Check if any query entity is part of the node name or vice versa
        for entity in query_entities:
            if entity in name_lower or name_lower in entity:
                matching_nodes.add(node)
                logger.info(f"Found matching node: {name} for entity: {entity}")
    
    return matching_nodes

def extract_relevant_triplets_from_entities(text_kg: OpenIEKnowledgeGraph, matching_nodes: set):
    """Extract relevant triplets from text KG based on matching nodes."""
    relevant_triplets = []
    
    # For each matching node, get its neighborhood
    for node in matching_nodes:
        logger.info(f"Getting neighborhood for node: {text_kg.G.nodes[node]['name']}")
        # Get all nodes and edges within 2 hops
        nodes, edges = get_nodes_at_depth_two(text_kg.G, node)
        
        # Convert edges to triplets
        for u, v, data in edges:
            triplet = {
                'subject': text_kg.G.nodes[u]['name'],
                'relation': data['relation'],
                'object': text_kg.G.nodes[v]['name'],
                'strength': data['strength']
            }
            if triplet not in relevant_triplets:
                relevant_triplets.append(triplet)
    
    return relevant_triplets

def process_text_to_kg(text: str, kg: OpenIEKnowledgeGraph):
    """Process text and build knowledge graph."""
    # Create a temporary file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8') as temp_file:
        temp_file.write(text)
        temp_path = temp_file.name
    
    try:
        # Process the temporary file and get triplets
        triplets = process_text_file(temp_path)
        
        if triplets:
            # Build the knowledge graph from triplets
            kg.clear_graph()
            kg.build_knowledge_graph(triplets)
            logger.info(f"Successfully built knowledge graph with {len(triplets)} triplets")
            return True
        
        logger.warning("No triplets found in the text")
        return False
    except Exception as e:
        logger.error(f"Error processing text: {str(e)}")
        return False
    finally:
        # Clean up temporary file
        os.unlink(temp_path)

def knowledge_graph_rag(text_file_path: str, query: str):
    """
    Perform RAG using knowledge graphs.
    
    Args:
        text_file_path (str): Path to the text file to process
        query (str): Query to process
    
    Returns:
        list: List of relevant triplets
    """
    try:
        # Read the text file
        with open(text_file_path, 'r', encoding='utf-8') as f:
            text_content = f.read()
        
        logger.info("Processing main text file...")
        # Process the main text and create its knowledge graph
        text_kg = OpenIEKnowledgeGraph()
        if not process_text_to_kg(text_content, text_kg):
            logger.error("Failed to process main text file")
            return []
        
        logger.info("Extracting entities from query...")
        # Extract entities from query using spaCy
        query_entities = extract_query_entities(query)
        
        # Find matching nodes in the knowledge graph
        logger.info("Finding matching nodes in knowledge graph...")
        matching_nodes = find_matching_nodes(text_kg, query_entities)
        
        if not matching_nodes:
            logger.warning("No matching nodes found in knowledge graph")
            return []
        
        # Extract relevant triplets based on matching nodes
        logger.info("Extracting relevant triplets...")
        relevant_triplets = extract_relevant_triplets_from_entities(text_kg, matching_nodes)
        
        return relevant_triplets
        
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        return []

def main():
    # Example usage
    text_file_path = "textFiles/text1.txt"  # Replace with your text file path
    query = "What is a computer mouse?"  # Changed to a more relevant query
    
    relevant_triplets = knowledge_graph_rag(text_file_path, query)
    logger.info("\nRelevant triplets for the query:")
    if not relevant_triplets:
        logger.info("No relevant triplets found")
    else:
        for triplet in relevant_triplets:
            logger.info(f"Subject: {triplet['subject']}")
            logger.info(f"Relation: {triplet['relation']}")
            logger.info(f"Object: {triplet['object']}")
            logger.info(f"Strength: {triplet['strength']}")
            logger.info("")

if __name__ == "__main__":
    main()
