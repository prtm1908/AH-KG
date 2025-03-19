from RAG import knowledge_graph_rag, OpenIEKnowledgeGraph, process_text_file, start_corenlp_server, extract_query_entities, find_matching_nodes, extract_relevant_triplets_from_entities
from neo4j import GraphDatabase
import networkx as nx
import matplotlib.pyplot as plt
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration variables
TEXT_FILE_PATH = "textFiles/text1.txt"  # Path to your input text file
QUERY = "What is a computer mouse?"  # Your query here

# Neo4j configuration
NEO4J_URI = "neo4j+s://cd0d8c00.databases.neo4j.io"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "c63yXkQm1fVLbpvEHvZsrUbTNiEMMyhU-VPiizHYEes"

class Neo4jConnector:
    def __init__(self, uri=NEO4J_URI, user=NEO4J_USER, password=NEO4J_PASSWORD):
        """Initialize Neo4j connection."""
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        
    def close(self):
        """Close the Neo4j connection."""
        self.driver.close()
        
    def clear_database(self):
        """Clear all nodes and relationships in Neo4j."""
        with self.driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")
            
    def create_knowledge_graph(self, text_kg: OpenIEKnowledgeGraph):
        """Create knowledge graph in Neo4j from the OpenIEKnowledgeGraph object."""
        with self.driver.session() as session:
            # Get all nodes and their attributes
            for node in text_kg.G.nodes():
                node_data = text_kg.G.nodes[node]
                
                # Convert entity to valid label name (remove spaces and special chars)
                label = node_data['name'].replace(' ', '_').replace('-', '_').upper()
                if not label[0].isalpha():
                    label = 'E_' + label
                
                # Prepare node properties
                node_props = {
                    'name': node_data['name'],
                    'text': node_data['name'],
                    'caption': node_data['name']
                }
                
                # Add embedding if available
                if 'embedding' in node_data:
                    node_props['embedding'] = node_data['embedding']
                
                cypher_query = f"""
                MERGE (n:{label} {{name: $name}})
                SET n += $props
                """
                session.run(cypher_query, name=node_data['name'], props=node_props)
            
            # Create relationships
            for u, v, data in text_kg.G.edges(data=True):
                # Convert relation to valid Neo4j identifier
                rel_type = data['relation'].upper().replace(' ', '_').replace('-', '_')
                
                cypher_query = f"""
                MATCH (s {{name: $subject}})
                MATCH (o {{name: $object}})
                CREATE (s)-[r:{rel_type}]->(o)
                SET r.type = $relation
                SET r.name = $relation
                SET r.caption = $relation
                SET r.strength = $strength
                """
                session.run(cypher_query, 
                          subject=text_kg.G.nodes[u]['name'],
                          object=text_kg.G.nodes[v]['name'],
                          relation=data['relation'],
                          strength=data['strength'])
            
            # Set display settings for all nodes
            session.run("""
            MATCH (n)
            SET n.displayName = n.name
            SET n.title = n.name
            """)

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

def process_and_visualize():
    """Main function to process text and create visualizations."""
    # Start CoreNLP server
    server_process = None
    neo4j = None
    
    try:
        # Start the CoreNLP server first
        logger.info("Starting CoreNLP server...")
        server_process = start_corenlp_server()
        
        # Process text file and get triplets
        logger.info("Processing text file...")
        triplets = process_text_file(TEXT_FILE_PATH)
        
        # Build knowledge graph with embeddings
        text_kg = OpenIEKnowledgeGraph()
        text_kg.build_knowledge_graph(triplets)
        
        # Get relevant triplets using the existing knowledge graph
        logger.info("Processing query using RAG...")
        query_entities = extract_query_entities(QUERY)
        matching_nodes = find_matching_nodes(text_kg, query_entities)
        relevant_triplets = extract_relevant_triplets_from_entities(text_kg, matching_nodes)
        
        # Connect to Neo4j and create knowledge graph with embeddings
        logger.info("Creating knowledge graph in Neo4j...")
        neo4j = Neo4jConnector()
        neo4j.clear_database()
        neo4j.create_knowledge_graph(text_kg)
        
        # Plot the full knowledge graph using NetworkX
        logger.info("Plotting full knowledge graph...")
        full_graph = plot_networkx_graph(triplets, "Full Knowledge Graph")
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
        
        logger.info("Visualization complete!")
            
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        raise
    finally:
        # Clean up Neo4j connection
        if neo4j:
            neo4j.close()
        # Stop the CoreNLP server
        if server_process:
            logger.info("Stopping CoreNLP server...")
            server_process.terminate()
            server_process.wait()

if __name__ == "__main__":
    process_and_visualize() 