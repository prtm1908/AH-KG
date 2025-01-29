import networkx as nx
import logging
from typing import List, Dict
import matplotlib.pyplot as plt
from relation_extractor import find_relations

class LORKnowledgeGraph:
    def __init__(self):
        """
        Initialize the NetworkX graph and set up logging.
        """
        self.G = nx.MultiDiGraph()  # Using MultiDiGraph to allow multiple relationships between same nodes
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def clear_graph(self):
        """Remove all nodes and relationships from the graph."""
        self.G.clear()
        self.logger.info("Graph cleared")

    def create_entity_node(self, entity: str):
        """
        Create a node with its name as its label.
        """
        # Sanitize the entity name but keep original as node attribute
        label = entity.replace(' ', '_').replace('-', '_').replace("'", '')
        
        # Ensure label starts with a letter
        if not label[0].isalpha():
            label = 'N_' + label
            
        try:
            self.G.add_node(label, name=entity)
        except Exception as e:
            self.logger.error(f"Error creating node for entity: {entity}")
            self.logger.error(str(e))
            raise

    def create_relationship(self, subject: str, relation: str, object: str):
        """
        Create a relationship between two nodes.
        """
        # Sanitize the labels
        subject_label = subject.replace(' ', '_').replace('-', '_').replace("'", '')
        object_label = object.replace(' ', '_').replace('-', '_').replace("'", '')
        
        # Ensure labels start with letters
        if not subject_label[0].isalpha():
            subject_label = 'N_' + subject_label
        if not object_label[0].isalpha():
            object_label = 'N_' + object_label
            
        # Convert relation to valid relationship type
        relation_type = relation.upper().replace(' ', '_').replace('-', '_')
        
        try:
            self.G.add_edge(subject_label, object_label, relation=relation_type)
        except Exception as e:
            self.logger.error(f"Error creating relationship: {subject} -{relation}-> {object}")
            self.logger.error(str(e))
            raise

    def build_knowledge_graph(self, relation_strings: List[str]):
        """
        Build the knowledge graph from relation strings.
        """
        for rel_str in relation_strings:
            try:
                # Split the relation string
                parts = rel_str.split(' -> ')
                if len(parts) != 3:
                    self.logger.warning(f"Invalid relation string format: {rel_str}")
                    continue
                
                subject, relation, object = parts
                
                # Create nodes for both entities
                self.create_entity_node(subject)
                self.create_entity_node(object)
                
                # Create the relationship
                self.create_relationship(subject, relation, object)
                
            except Exception as e:
                self.logger.error(f"Error processing relation string: {rel_str}")
                self.logger.error(str(e))
                continue
        
        self.logger.info("Knowledge graph built successfully")

    def get_graph_statistics(self) -> Dict:
        """
        Get basic statistics about the knowledge graph.
        """
        try:
            # Count nodes
            node_count = self.G.number_of_nodes()
            
            # Count relationships
            rel_count = self.G.number_of_edges()
            
            # Get all unique node types (labels)
            node_types = list(set(nx.get_node_attributes(self.G, 'name').values()))
            
            # Get relationship types
            rel_types = list(set(nx.get_edge_attributes(self.G, 'relation').values()))
            
            return {
                "node_count": node_count,
                "relationship_count": rel_count,
                "node_types": node_types,
                "relationship_types": rel_types
            }
        except Exception as e:
            self.logger.error("Error getting graph statistics")
            self.logger.error(str(e))
            return {
                "node_count": 0,
                "relationship_count": 0,
                "node_types": [],
                "relationship_types": []
            }

    def visualize_graph(self, figsize=(16, 12)):
        """
        Visualize the knowledge graph using NetworkX and Matplotlib with Graphviz neato layout.
        """
        plt.figure(figsize=figsize)
        
        try:
            # Calculate node degrees for sizing
            degrees = dict(self.G.degree())
            
            # Use Graphviz neato layout
            pos = nx.nx_agraph.graphviz_layout(self.G, prog='neato', args='-Goverlap=false -Gsplines=true')
            
            # Scale node sizes based on degree centrality but with smaller range
            node_sizes = [1500 + (degrees[node] * 100) for node in self.G.nodes()]
            
            # Draw edges with curved arrows and reduced alpha
            nx.draw_networkx_edges(self.G, pos, 
                                 edge_color='gray',
                                 alpha=0.4,
                                 arrows=True,
                                 arrowsize=15,
                                 connectionstyle='arc3,rad=0.2',  # Add curve to edges
                                 min_source_margin=20,
                                 min_target_margin=20)
            
            # Draw nodes with varying sizes
            nx.draw_networkx_nodes(self.G, pos,
                                 node_color='lightblue',
                                 node_size=node_sizes,
                                 alpha=0.7)
            
            # Draw node labels with improved visibility
            node_labels = nx.get_node_attributes(self.G, 'name')
            for node, (x, y) in pos.items():
                plt.text(x, y, node_labels[node],
                        fontsize=8,
                        bbox=dict(facecolor='white',
                                alpha=0.8,
                                edgecolor='lightgray',
                                boxstyle='round,pad=0.5'),
                        horizontalalignment='center',
                        verticalalignment='center')
            
            # Draw edge labels selectively
            edge_labels = nx.get_edge_attributes(self.G, 'relation')
            # Filter to show only shorter labels and add line breaks for longer ones
            filtered_edge_labels = {}
            for k, v in edge_labels.items():
                if len(v) < 15:
                    filtered_edge_labels[k] = v
                else:
                    # Split long labels into multiple lines
                    filtered_edge_labels[k] = '\n'.join([v[i:i+10] for i in range(0, len(v), 10)])
            
            nx.draw_networkx_edge_labels(self.G, pos,
                                       edge_labels=filtered_edge_labels,
                                       font_size=6,
                                       alpha=0.7,
                                       bbox=dict(facecolor='white',
                                               alpha=0.7,
                                               edgecolor='none',
                                               pad=0.2))
            
            plt.title("Knowledge Graph Visualization (Neato Layout)", pad=20)
            plt.axis('off')
            plt.tight_layout()
            
        except ImportError:
            self.logger.error("Graphviz not installed. Please install graphviz and pygraphviz.")
            raise
        except Exception as e:
            self.logger.error(f"Error visualizing graph: {str(e)}")
            raise
        
        return plt

def main():
    # Initialize graph
    graph = LORKnowledgeGraph()
    
    # Clear existing data
    graph.clear_graph()
    
    # Example relation strings from your text processing
    relation_strings = find_relations()
    
    # Build the knowledge graph
    graph.build_knowledge_graph(relation_strings)
    
    # Get and print statistics
    stats = graph.get_graph_statistics()
    print("\nGraph Statistics:")
    print(f"Number of nodes: {stats['node_count']}")
    print(f"Number of relationships: {stats['relationship_count']}")
    print("Relationship types:", ", ".join(stats['relationship_types']))
    
    # Visualize the graph
    plt = graph.visualize_graph()
    plt.show()

if __name__ == "__main__":
    main()