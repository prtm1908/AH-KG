from openie import StanfordOpenIE
import fastcoref
import re
import networkx as nx
import logging
from typing import List, Dict
from collections import defaultdict
from graphsage_embeddings import generate_graphsage_embeddings
from de_lemma import lemmatize_triplets, de_lemmatize_triplets

class OpenIEKnowledgeGraph:
    def __init__(self):
        """Initialize the NetworkX graph and set up logging."""
        self.G = nx.MultiDiGraph()  # Using MultiDiGraph to allow multiple relationships between same nodes
        logging.basicConfig(level=logging.INFO) 
        self.logger = logging.getLogger(__name__)
        self.node_embeddings = None  # Store node embeddings

    def clear_graph(self):
        """Remove all nodes and relationships from the graph."""
        self.G.clear()
        self.node_embeddings = None
        self.logger.info("Graph cleared")

    def create_entity_node(self, entity: str):
        """Create a node with its name as its label."""
        # Sanitize the entity name but keep original as node attribute
        label = entity.replace(' ', '_').replace('-', '_').replace("'", '')
        
        # Ensure label starts with a letter
        if not label[0].isalpha():
            label = 'N_' + label
            
        try:
            # Store both the lemmatized name and the original name
            self.G.add_node(label, name=entity, original_name=entity)
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
        """Build the knowledge graph from OpenIE triplets and generate GraphSAGE embeddings."""
        # First build the graph structure
        for triplet in triplets:
            try:
                # Get both lemmatized and original forms
                subject = triplet['subject']
                relation = triplet['relation']
                obj = triplet['object']
                strength = triplet.get('strength', 1.0)  # Default strength is 1.0
                
                # Get original forms if available
                original_subject = triplet.get('original_subject', subject)
                original_object = triplet.get('original_object', obj)
                
                # Create nodes for both entities with their original forms
                subject_label = subject.replace(' ', '_').replace('-', '_').replace("'", '')
                object_label = obj.replace(' ', '_').replace('-', '_').replace("'", '')
                
                # Ensure labels start with letters
                if not subject_label[0].isalpha():
                    subject_label = 'N_' + subject_label
                if not object_label[0].isalpha():
                    object_label = 'N_' + object_label
                
                # Add nodes with both lemmatized and original forms
                self.G.add_node(subject_label, name=subject, original_name=original_subject)
                self.G.add_node(object_label, name=obj, original_name=original_object)
                
                # Create the relationship with original forms
                relation_type = relation.upper().replace(' ', '_').replace('-', '_')
                self.G.add_edge(subject_label, object_label, 
                              relation=relation_type, 
                              strength=strength,
                              original_subject=original_subject,
                              original_object=original_object)
                
            except Exception as e:
                self.logger.error(f"Error processing triplet: {triplet}")
                self.logger.error(str(e))
                continue

        # Generate GraphSAGE embeddings for all nodes
        self.logger.info("Generating GraphSAGE embeddings...")
        self.node_embeddings = generate_graphsage_embeddings(triplets)
        
        # Add embeddings as node attributes
        for node in self.G.nodes():
            node_name = self.G.nodes[node]['name']
            if node_name in self.node_embeddings:
                self.G.nodes[node]['embedding'] = self.node_embeddings[node_name]
        
        self.logger.info("Knowledge graph built successfully with embeddings")

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

def consolidate_and_lemmatize_triplets(triplets):
    """Consolidate similar entities and relations, then lemmatize all entities in triplets."""
    # First consolidate all entities and relations together
    entity_mapping, relation_mapping = consolidate_all_entities_and_relations(triplets)
    
    # Convert triplets to raw format for lemmatization
    raw_triplets = []
    for triplet in triplets:
        # Store original forms
        original_subject = triplet['subject']
        original_object = triplet['object']
        
        # Apply consolidation
        subject = entity_mapping.get(triplet['subject'], triplet['subject'])
        relation = relation_mapping.get(triplet['relation'], triplet['relation'])
        obj = entity_mapping.get(triplet['object'], triplet['object'])
        
        raw_triplets.append((subject, relation, obj))
    
    # Lemmatize the triplets
    lemmatized_triplets = lemmatize_triplets(raw_triplets)
    
    # Add original forms and other metadata back
    consolidated_triplets = []
    for i, lemmatized in enumerate(lemmatized_triplets):
        new_triplet = {
            'subject': lemmatized['subject'],
            'relation': lemmatized['relationship'],
            'object': lemmatized['object'],
            'original_subject': triplets[i]['subject'],
            'original_object': triplets[i]['object'],
            'metadata': lemmatized['metadata']
        }
        consolidated_triplets.append(new_triplet)
    
    # Remove any duplicates that might have been created during consolidation
    # But preserve the original forms when removing duplicates
    unique_triplets = []
    seen_combinations = set()
    
    for t in consolidated_triplets:
        # Create a key for the lemmatized form
        key = (t['subject'], t['relation'], t['object'])
        
        if key not in seen_combinations:
            seen_combinations.add(key)
            unique_triplets.append(t)
        else:
            # If we've seen this combination before, merge the original forms
            for existing in unique_triplets:
                if (existing['subject'], existing['relation'], existing['object']) == key:
                    # Add the original unconsolidated forms to the list
                    if 'all_original_forms' not in existing:
                        existing['all_original_forms'] = {
                            'subjects': [existing['original_subject']],
                            'objects': [existing['original_object']]
                        }
                    existing['all_original_forms']['subjects'].append(t['original_subject'])
                    existing['all_original_forms']['objects'].append(t['original_object'])
                    break
    
    return unique_triplets

def process_text_file(file_path: str):
    """
    Process a text file to generate a knowledge graph.
    
    Args:
        file_path (str): Path to the text file to process
        
    Returns:
        list: List of triplets extracted from the text
    """
    # Initialize models
    model = fastcoref.FCoref()
    properties = {
        'openie.affinity_probability_cap': 2/3,
        'openie.resolve_coref': True,
    }

    # Read original text
    with open(file_path, encoding='utf8') as f:
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
        lemmatized_triplets = consolidate_and_lemmatize_triplets(deduped_triplets)
        
        print('\nLemmatized triplets after duplicate removal:')
        for triple in lemmatized_triplets:
            print('|-', {
                'subject': triple['subject'],
                'relation': triple['relation'],
                'object': triple['object'],
                'strength': triple.get('strength', 1.0)
            })

        # Add default strength to each triplet if not present
        for triplet in lemmatized_triplets:
            if 'strength' not in triplet:
                triplet['strength'] = 1.0

        print('\nOriginal Text:', text)
        print('\nFinal lemmatized triplets:')
        for triple in lemmatized_triplets:
            print('|-', triple)

        # Return the lemmatized triplets directly
        return lemmatized_triplets

def main():
    # Example usage with default text file
    process_text_file('textFiles/text1.txt')

if __name__ == '__main__':
    main()