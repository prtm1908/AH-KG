from openie import StanfordOpenIE
import fastcoref
import re
import networkx as nx
import matplotlib.pyplot as plt
import logging
from typing import List, Dict
import spacy
from collections import defaultdict

class OpenIEKnowledgeGraph:
    def __init__(self):
        """Initialize the NetworkX graph and set up logging."""
        self.G = nx.MultiDiGraph()  # Using MultiDiGraph to allow multiple relationships between same nodes
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def clear_graph(self):
        """Remove all nodes and relationships from the graph."""
        self.G.clear()
        self.logger.info("Graph cleared")

    def create_entity_node(self, entity: str):
        """Create a node with its name as its label."""
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

    def create_relationship(self, subject: str, relation: str, object: str, strength: float = 1.0):
        """Create a relationship between two nodes."""
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
            self.G.add_edge(subject_label, object_label, relation=relation_type, strength=strength)
        except Exception as e:
            self.logger.error(f"Error creating relationship: {subject} -{relation}-> {object}")
            self.logger.error(str(e))
            raise

    def build_knowledge_graph(self, triplets: List[Dict]):
        """Build the knowledge graph from OpenIE triplets."""
        for triplet in triplets:
            try:
                subject = triplet['subject']
                relation = triplet['relation']
                obj = triplet['object']
                strength = triplet.get('strength', 1.0)  # Default strength is 1.0
                
                # Create nodes for both entities
                self.create_entity_node(subject)
                self.create_entity_node(obj)
                
                # Create the relationship
                self.create_relationship(subject, relation, obj, strength)
                
            except Exception as e:
                self.logger.error(f"Error processing triplet: {triplet}")
                self.logger.error(str(e))
                continue
        
        self.logger.info("Knowledge graph built successfully")

    def get_graph_statistics(self) -> Dict:
        """Get basic statistics about the knowledge graph."""
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
        """Visualize the knowledge graph using NetworkX and Matplotlib with Graphviz neato layout."""
        plt.figure(figsize=figsize)
        
        try:
            # Calculate node degrees for sizing
            degrees = dict(self.G.degree())
            
            # Use Graphviz neato layout
            pos = nx.nx_agraph.graphviz_layout(self.G, prog='neato', args='-Goverlap=false -Gsplines=true')
            
            # Scale node sizes based on degree centrality but with smaller range
            node_sizes = [1500 + (degrees[node] * 100) for node in self.G.nodes()]
            
            # Get edge strengths for width and alpha
            edge_strengths = nx.get_edge_attributes(self.G, 'strength')
            edge_widths = [strength * 2 for strength in edge_strengths.values()]
            edge_alphas = [min(0.2 + strength * 0.3, 0.9) for strength in edge_strengths.values()]
            
            # Draw edges with curved arrows and strength-based width/alpha
            nx.draw_networkx_edges(self.G, pos, 
                                 edge_color='gray',
                                 width=edge_widths,
                                 alpha=edge_alphas,
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
            
            # Draw edge labels with strength
            edge_labels = {}
            for (u, v, k), relation in nx.get_edge_attributes(self.G, 'relation').items():
                strength = self.G[u][v][k]['strength']
                edge_labels[(u, v)] = f"{relation}\n(s={strength:.1f})"
            
            nx.draw_networkx_edge_labels(self.G, pos,
                                       edge_labels=edge_labels,
                                       font_size=6,
                                       alpha=0.7,
                                       bbox=dict(facecolor='white',
                                               alpha=0.7,
                                               edgecolor='none',
                                               pad=0.2))
            
            plt.title("OpenIE + FastCoref", pad=20)
            plt.axis('off')
            plt.tight_layout()
            
        except ImportError:
            self.logger.error("Graphviz not installed. Please install graphviz and pygraphviz.")
            raise
        except Exception as e:
            self.logger.error(f"Error visualizing graph: {str(e)}")
            raise
        
        plt.savefig('graphCoref.png', bbox_inches='tight', dpi=300)
        plt.close()

def is_pronoun(word):
    pronouns = {'he', 'him', 'his', 'she', 'her', 'hers', 'they', 'them', 'their', 'theirs', 'it', 'its'}
    return word.lower() in pronouns

def get_replacement_text(antecedent, mention):
    if mention.lower() in {'his', 'her', 'their', 'its'}:
        if not antecedent.endswith("'s"):
            return antecedent + "'s"
    return antecedent

def has_special_symbols(text):
    return any(char in text for char in '()[]{},\n')

def contains_pronoun(text):
    pronouns = {'he', 'him', 'his', 'she', 'her', 'hers', 'they', 'them', 'their', 'theirs', 'it', 'its'}
    words = text.lower().split()
    return any(word in pronouns for word in words)

def is_clean_mention(mention):
    return not has_special_symbols(mention) and not contains_pronoun(mention)

def get_pronouns_from_triplets(triplets):
    pronouns_to_resolve = set()
    for triple in triplets:
        # Only check for standalone pronouns
        if is_pronoun(triple['subject']):
            pronouns_to_resolve.add(triple['subject'])
        if is_pronoun(triple['object']):
            pronouns_to_resolve.add(triple['object'])
    
    return pronouns_to_resolve

def resolve_triplet(triple, clusters, pronouns_to_resolve, text):
    resolved_triple = triple.copy()
    
    # For each cluster, try to resolve pronouns in subject and object
    for cluster in clusters:
        # Skip clusters that don't contain any of our target pronouns
        if not any(mention.lower() in pronouns_to_resolve for mention in cluster):
            continue
            
        # Find the best antecedent
        antecedent = None
        antecedent_pos = float('inf')
        
        # First try to find a clean mention
        for mention in cluster:
            if not is_pronoun(mention) and is_clean_mention(mention):
                pos = text.find(mention)
                if pos != -1 and pos < antecedent_pos:
                    antecedent_pos = pos
                    antecedent = mention
        
        # If no clean mention, try one without special symbols
        if antecedent is None:
            for mention in cluster:
                if not is_pronoun(mention) and not has_special_symbols(mention):
                    pos = text.find(mention)
                    if pos != -1 and pos < antecedent_pos:
                        antecedent_pos = pos
                        antecedent = mention
        
        # If still no antecedent, use any non-pronoun
        if antecedent is None:
            for mention in cluster:
                if not is_pronoun(mention):
                    pos = text.find(mention)
                    if pos != -1 and pos < antecedent_pos:
                        antecedent_pos = pos
                        antecedent = mention
        
        # If still nothing found, use first mention
        if antecedent is None:
            antecedent = cluster[0]
        
        # Only replace if the entire subject/object is a pronoun
        if resolved_triple['subject'] in pronouns_to_resolve:
            resolved_triple['subject'] = antecedent
            
        if resolved_triple['object'] in pronouns_to_resolve:
            resolved_triple['object'] = antecedent
    
    return resolved_triple

def normalize_text(text):
    # Remove articles and normalize spaces
    text = re.sub(r'\b(a|an|the)\b', '', text.lower())
    return ' '.join(text.split())

def clean_entity_text(text):
    # Remove articles, possessive pronouns, and normalize spaces
    text = re.sub(r'\b(a|an|the|his|her|their|its)\b', '', text.lower())
    # Remove extra spaces and strip
    return ' '.join(text.split()).strip()

def are_entities_similar(entity1: str, entity2: str) -> bool:
    """
    Check if two entities are similar by comparing their words.
    Returns True if one is a substring of other or if they share most words.
    """
    # First check direct substring relationship
    if entity1.lower() in entity2.lower() or entity2.lower() in entity1.lower():
        return True
        
    # Split into words and remove common stop words
    stop_words = {'a', 'an', 'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
    words1 = set(w.lower() for w in entity1.split() if w.lower() not in stop_words)
    words2 = set(w.lower() for w in entity2.split() if w.lower() not in stop_words)
    
    # If one set is empty after removing stop words, return False
    if not words1 or not words2:
        return False
    
    # Calculate word overlap
    common_words = words1.intersection(words2)
    total_words = words1.union(words2)
    
    # Consider similar if they share at least 50% of their non-stop words
    similarity_ratio = len(common_words) / len(total_words)
    return similarity_ratio >= 0.4

def is_similar_triplet(t1, t2):
    # Two triplets are similar if they have similar subjects, relations, and objects
    subj1 = clean_entity_text(t1['subject'])
    subj2 = clean_entity_text(t2['subject'])
    rel1 = t1['relation']
    rel2 = t2['relation']
    obj1 = clean_entity_text(t1['object'])
    obj2 = clean_entity_text(t2['object'])
    
    # Check if subjects, relations, and objects are similar
    return (are_entities_similar(subj1, subj2) and 
            are_entities_similar(rel1, rel2) and 
            are_entities_similar(obj1, obj2))

def remove_duplicate_triplets(triplets):
    unique_triplets = []
    for t1 in triplets:
        # Clean both subject and object text in the current triplet
        t1['subject'] = clean_entity_text(t1['subject'])
        t1['object'] = clean_entity_text(t1['object'])
        # Check if this triplet is similar to any we've already kept
        if not any(is_similar_triplet(t1, t2) for t2 in unique_triplets):
            unique_triplets.append(t1)
    return unique_triplets

def find_shortest_entity(entities):
    """Find the shortest entity from a list of similar entities."""
    return min(entities, key=len)

def consolidate_all_entities_and_relations(triplets):
    """
    Consolidate all entities (subjects, objects) and relations that are similar.
    Returns mappings for entities and relations to their consolidated forms.
    """
    # First collect all unique entities and relations
    all_entities = set()
    all_relations = set()
    for t in triplets:
        all_entities.add(t['subject'])
        all_entities.add(t['object'])
        all_relations.add(t['relation'])
    
    # Group similar entities
    entity_groups = []
    processed_entities = set()
    
    for entity1 in all_entities:
        if entity1 in processed_entities:
            continue
            
        similar_entities = {entity1}
        for entity2 in all_entities:
            if entity2 != entity1 and entity2 not in processed_entities:
                if are_entities_similar(entity1, entity2):
                    similar_entities.add(entity2)
        
        if len(similar_entities) > 1:
            entity_groups.append(similar_entities)
            processed_entities.update(similar_entities)
    
    # Group similar relations
    relation_groups = []
    processed_relations = set()
    
    for rel1 in all_relations:
        if rel1 in processed_relations:
            continue
            
        similar_relations = {rel1}
        for rel2 in all_relations:
            if rel2 != rel1 and rel2 not in processed_relations:
                if are_entities_similar(rel1, rel2):  # Reuse same similarity function
                    similar_relations.add(rel2)
        
        if len(similar_relations) > 1:
            relation_groups.append(similar_relations)
            processed_relations.update(similar_relations)
    
    # Create mappings using shortest forms
    entity_mapping = {}
    for group in entity_groups:
        shortest = find_shortest_entity(group)
        for entity in group:
            entity_mapping[entity] = shortest
            
    relation_mapping = {}
    for group in relation_groups:
        shortest = find_shortest_entity(group)  # Reuse same shortest finding function
        for relation in group:
            relation_mapping[relation] = shortest
    
    return entity_mapping, relation_mapping

def lemmatize_entity(text):
    """Lemmatize each word in the entity text."""
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(text)
    return ' '.join([token.lemma_ for token in doc])

def consolidate_and_lemmatize_triplets(triplets):
    """Consolidate similar entities and relations, then lemmatize all entities in triplets."""
    # First consolidate all entities and relations together
    entity_mapping, relation_mapping = consolidate_all_entities_and_relations(triplets)
    
    # Create new triplets with consolidated entities and relations, then lemmatize
    consolidated_triplets = []
    for triplet in triplets:
        new_triplet = triplet.copy()
        
        # Apply consolidation for subject
        subject = entity_mapping.get(triplet['subject'], triplet['subject'])
        new_triplet['subject'] = lemmatize_entity(subject)
        
        # Apply consolidation for relation
        relation = relation_mapping.get(triplet['relation'], triplet['relation'])
        new_triplet['relation'] = relation  # Don't lemmatize relations
        
        # Apply consolidation for object
        obj = entity_mapping.get(triplet['object'], triplet['object'])
        new_triplet['object'] = lemmatize_entity(obj)
        
        consolidated_triplets.append(new_triplet)
    
    # Remove any duplicates that might have been created during consolidation
    return remove_duplicate_triplets(consolidated_triplets)

def main():
    # Initialize models
    model = fastcoref.FCoref()
    properties = {
        'openie.affinity_probability_cap': 2/3,
        'openie.resolve_coref': True,
    }

    # Read original text
    with open('../textFiles/text1.txt', encoding='utf8') as f:
        text = f.read()

    # Split text into sentences (simple split by period for now)
    sentences = [s.strip() + '.' for s in text.split('.') if s.strip()]
    
    # Keep track of all triplets and accumulated text for coref resolution
    all_triplets = []
    accumulated_text = ""
    
    # Process each sentence
    with StanfordOpenIE(properties=properties) as client:
        for i, sentence in enumerate(sentences, 1):
            print(f'\nProcessing sentence {i}/{len(sentences)}:', sentence)
            
            # Get triplets for current sentence
            current_triplets = client.annotate(sentence)
            if not current_triplets:
                print('No triplets found in this sentence')
                continue
                
            # Update accumulated text
            accumulated_text += " " + sentence
            
            # Check for pronouns in current triplets
            pronouns_to_resolve = get_pronouns_from_triplets(current_triplets)
            
            if pronouns_to_resolve:
                print('Pronouns found in triplets:', pronouns_to_resolve)
                # Use all accumulated text for coref resolution
                preds = model.predict(texts=[accumulated_text])
                clusters = preds[0].get_clusters()
                
                # Resolve triplets using full context
                resolved_current = [resolve_triplet(triple, clusters, pronouns_to_resolve, accumulated_text) 
                                 for triple in current_triplets]
            else:
                print('No pronouns to resolve in this sentence')
                resolved_current = current_triplets
            
            # Add resolved triplets to our collection
            all_triplets.extend(resolved_current)
        
        # Remove initial duplicates
        deduped_triplets = remove_duplicate_triplets(all_triplets)
        
        # Consolidate similar entities and lemmatize
        print('\nConsolidating similar entities and lemmatizing...')
        final_triplets = consolidate_and_lemmatize_triplets(deduped_triplets)
        
        # Add default strength to each triplet
        for triplet in final_triplets:
            triplet['strength'] = 1.0

        print('\nOriginal Text:', text)
        print('\nAll resolved relations:')
        for triple in final_triplets:
            print('|-', triple)

        # Create and visualize knowledge graph
        kg = OpenIEKnowledgeGraph()
        kg.clear_graph()
        kg.build_knowledge_graph(final_triplets)
        
        # Get and print statistics
        stats = kg.get_graph_statistics()
        print("\nGraph Statistics:")
        print(f"Number of nodes: {stats['node_count']}")
        print(f"Number of relationships: {stats['relationship_count']}")
        print("Relationship types:", ", ".join(stats['relationship_types']))
        
        # Visualize and save the graph
        kg.visualize_graph()
        print('Graph generated: graphCoref.png')

if __name__ == '__main__':
    main()