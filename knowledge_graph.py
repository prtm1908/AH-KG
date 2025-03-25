import fastcoref
import re
import networkx as nx
import logging
from typing import List, Dict
from collections import defaultdict
from graphsage_embeddings import generate_graphsage_embeddings
from de_lemma import lemmatize_triplets, de_lemmatize_triplets
import requests
import time
import json
import os
from neo4j import GraphDatabase
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, HttpUrl
import tempfile
from typing import TYPE_CHECKING

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(title="Knowledge Graph Creation API")

# Define request model
class KnowledgeGraphRequest(BaseModel):
    text_url: HttpUrl

# Neo4j configuration from environment variables
NEO4J_URI = os.environ.get('NEO4J_URI', 'bolt://localhost:7687')
NEO4J_USER = os.environ.get('NEO4J_USER', 'neo4j')
NEO4J_PASSWORD = os.environ.get('NEO4J_PASSWORD', 'password')

class OpenIEKnowledgeGraph:
    def __init__(self):
        """Initialize the NetworkX graph and set up logging."""
        self.G = nx.MultiDiGraph()  # Using MultiDiGraph to allow multiple relationships between same nodes
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
        """Build the knowledge graph from OpenIE triplets."""
        # Build the graph structure
        for triplet in triplets:
            try:
                # Get both lemmatized and original forms
                subject = triplet['subject']
                relation = triplet['relation']
                obj = triplet['object']
                
                # Skip triplets with empty strings
                if not subject or not relation or not obj:
                    self.logger.warning(f"Skipping invalid triplet with empty values: {triplet}")
                    continue
                
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
        
        self.logger.info("Knowledge graph built successfully")

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
            
    def create_knowledge_graph(self, text_kg: 'OpenIEKnowledgeGraph'):
        """Create knowledge graph in Neo4j from the OpenIEKnowledgeGraph object."""
        with self.driver.session() as session:
            # Get all nodes and their attributes
            for node in text_kg.G.nodes():
                node_data = text_kg.G.nodes[node]
                
                # Convert entity to valid label name (remove spaces, dots, and special chars)
                label = re.sub(r'[^A-Za-z0-9_]', '_', node_data['name'].upper())
                # Ensure label starts with a letter
                if not label[0].isalpha():
                    label = 'E_' + label
                # Remove consecutive underscores
                label = re.sub(r'_+', '_', label)
                # Remove trailing underscore
                label = label.rstrip('_')
                
                # Prepare node properties
                node_props = {
                    'name': node_data['name'],
                    'text': node_data['name'],
                    'caption': node_data['name']
                }
                
                cypher_query = f"""
                MERGE (n:{label} {{name: $name}})
                SET n += $props
                """
                session.run(cypher_query, name=node_data['name'], props=node_props)
            
            # Create relationships
            for u, v, data in text_kg.G.edges(data=True):
                # Convert relation to valid Neo4j identifier
                rel_type = re.sub(r'[^A-Za-z0-9_]', '_', data['relation'].upper())
                # Ensure relationship type starts with a letter
                if not rel_type[0].isalpha():
                    rel_type = 'REL_' + rel_type
                # Remove consecutive underscores
                rel_type = re.sub(r'_+', '_', rel_type)
                # Remove trailing underscore
                rel_type = rel_type.rstrip('_')
                
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
    # Skip invalid triplets before processing
    valid_triplets = [t for t in triplets if t['subject'] and t['relation'] and t['object']]
    
    if not valid_triplets:
        return []
        
    # First consolidate all entities and relations together
    entity_mapping, relation_mapping = consolidate_all_entities_and_relations(valid_triplets)
    
    # Convert triplets to raw format for lemmatization
    raw_triplets = []
    for triplet in valid_triplets:
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
            'original_subject': valid_triplets[i]['subject'],
            'original_object': valid_triplets[i]['object'],
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

def download_and_save_text(url, output_file):
    """Download text from URL and save to file."""
    logger.info(f"Downloading text from {url}")
    response = requests.get(url)
    if response.status_code == 200:
        text = response.text
        logger.info(f"Downloaded text content (first 100 chars): {text[:100]}...")
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(text)
        logger.info(f"Saved text to {output_file}")
        return True
    else:
        logger.error(f"Failed to download text. Status code: {response.status_code}")
        return False

def extract_triplets_from_corenlp_response(response_json):
    """Extract OpenIE triplets from CoreNLP server response."""
    triplets = []
    for sentence in response_json['sentences']:
        if 'openie' in sentence:
            for triple in sentence['openie']:
                # Skip if any required field is empty
                if not triple['subject'].strip() or not triple['relation'].strip() or not triple['object'].strip():
                    continue
                    
                triplet = {
                    'subject': triple['subject'].strip(),
                    'relation': triple['relation'].strip(),
                    'object': triple['object'].strip(),
                    'strength': 1.0  # Default strength
                }
                triplets.append(triplet)
    return triplets

def process_text_file(file_path: str):
    """
    Process a text file to generate a knowledge graph.
    
    Args:
        file_path (str): Path to the text file to process
        
    Returns:
        int: Total number of triplets processed
    """
    # If file_path is a URL, download it first
    if file_path.startswith('http'):
        temp_file = 'temp_text.txt'
        if not download_and_save_text(file_path, temp_file):
            logger.error("Failed to download text from URL")
            return 0
        file_path = temp_file

    # Initialize models and connections
    model = fastcoref.FCoref()
    text_kg = OpenIEKnowledgeGraph()
    neo4j = Neo4jConnector()
    neo4j.clear_database()  # Clear at start

    # Read original text
    with open(file_path, encoding='utf8') as f:
        text = f.read()
    
    # Clean the text - remove special annotations and normalize
    text = re.sub(r'\[nb \d+\]|\[\d+\]', '', text)  # Remove [nb 1] and [1] style annotations

    # Split text into sentences
    sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if s.strip()]
    logger.info(f"Number of sentences found: {len(sentences)}")
    
    # Process in batches
    batch_size = 100  # Process 100 sentences at a time
    total_triplets_processed = 0
    
    try:
        for batch_start in range(0, len(sentences), batch_size):
            batch_end = min(batch_start + batch_size, len(sentences))
            batch_sentences = sentences[batch_start:batch_end]
            logger.info(f"\nProcessing batch of sentences {batch_start+1} to {batch_end}")
            
            # Initialize batch variables
            batch_triplets = []
            context_sentences = []  # Keep track of context sentences
            
            # Process each sentence in the batch sequentially
            for i, sentence in enumerate(batch_sentences, batch_start + 1):
                try:
                    logger.info(f'Processing sentence {i}/{len(sentences)}: {sentence[:100]}...')
                    
                    # Add current sentence to context
                    context_sentences.append(sentence)
                    current_context = " ".join(context_sentences)  # Use all previous sentences in batch as context
                    
                    # Send request to CoreNLP server
                    response = requests.post(
                        'http://corenlp:9000/?properties=' + 
                        json.dumps({
                            'annotators': 'tokenize,ssplit,pos,lemma,depparse,natlog,coref,openie',
                            'outputFormat': 'json',
                            'openie.affinity_probability_cap': '0.67',
                            'openie.resolve_coref': 'true'
                        }),
                        data=sentence.encode('utf-8'),
                        headers={'Content-Type': 'text/plain; charset=utf-8'}
                    )
                    response.raise_for_status()
                    
                    # Extract triplets
                    current_triplets = extract_triplets_from_corenlp_response(response.json())
                    
                    if not current_triplets:
                        continue
                    
                    # Resolve coreferences if needed using context
                    pronouns_to_resolve = get_pronouns_from_triplets(current_triplets)
                    if pronouns_to_resolve:
                        # Use all previous sentences in batch as context for coreference resolution
                        preds = model.predict(texts=[current_context])
                        clusters = preds[0].get_clusters()
                        current_triplets = [resolve_triplet(triple, clusters, pronouns_to_resolve, current_context) 
                                         for triple in current_triplets]
                    
                    batch_triplets.extend(current_triplets)
                    
                except Exception as e:
                    logger.error(f"Error processing sentence {i}: {str(e)}")
                    continue
            
            if batch_triplets:
                # Process the batch of triplets
                logger.info(f"Processing batch of {len(batch_triplets)} triplets")
                
                # Remove duplicates within this batch
                deduped_batch = remove_duplicate_triplets(batch_triplets)
                
                # Consolidate and lemmatize the batch
                processed_batch = consolidate_and_lemmatize_triplets(deduped_batch)
                
                # Add default strength if not present
                for triplet in processed_batch:
                    if 'strength' not in triplet:
                        triplet['strength'] = 1.0
                
                # Update knowledge graph with this batch
                text_kg.build_knowledge_graph(processed_batch)
                
                # Save this batch to Neo4j
                neo4j.create_knowledge_graph(text_kg)
                
                # Clear the graph for next batch
                text_kg.clear_graph()
                
                # Update total and clear batch variables
                total_triplets_processed += len(processed_batch)
                logger.info(f"Processed and saved {len(processed_batch)} triplets from this batch")
                logger.info(f"Total triplets processed so far: {total_triplets_processed}")
                
                # Clear batch variables to free memory
                batch_triplets = []
                processed_batch = []
                deduped_batch = []
                context_sentences = []  # Clear context for next batch
            
    except Exception as e:
        logger.error(f"Error in batch processing: {str(e)}")
    finally:
        neo4j.close()
    
    return total_triplets_processed

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

async def process_and_create_knowledge_graph(text_url: str):
    """Process text and create knowledge graph in Neo4j."""
    temp_file_path = None
    
    try:
        # Fetch text from URL and save to temporary file
        logger.info("Fetching text from URL...")
        temp_file_path = await process_text_from_url(text_url)
        
        # Process text file and get triplets count
        logger.info("Processing text content...")
        total_triplets = process_text_file(temp_file_path)
        
        return {
            "status": "success",
            "message": "Knowledge graph created successfully!",
            "triplets_processed": total_triplets
        }
            
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Clean up temporary file
        if temp_file_path and os.path.exists(temp_file_path):
            os.unlink(temp_file_path)

@app.post("/create-knowledge-graph")
async def create_knowledge_graph(request: KnowledgeGraphRequest):
    """API endpoint to create a knowledge graph from text URL."""
    return await process_and_create_knowledge_graph(str(request.text_url))

def main():
    # Example usage with default text file
    process_text_file('textFiles/text1.txt')

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)  # Changed from 8001 to 8000 to match docker-compose