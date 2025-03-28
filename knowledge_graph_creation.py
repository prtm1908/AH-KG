import spacy
from fastcoref import spacy_component
from collections import defaultdict
from typing import Dict, List, Tuple
import re
from neo4j import GraphDatabase
from neo4j.exceptions import ServiceUnavailable
from dotenv import load_dotenv
import os
import torch

def create_triplets_spacy_fastcoref(text):
    # Load English language model with minimal components
    nlp = spacy.load("en_core_web_sm", exclude=["parser", "lemmatizer", "ner", "textcat"])
    
    # Add sentencizer to the pipeline
    nlp.add_pipe("sentencizer")
    
    # Detect if CUDA is available
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    
    # Add FastCoref to the pipeline with LingMessCoref model
    nlp.add_pipe(
        "fastcoref", 
        config={
            'model_architecture': 'LingMessCoref',
            'model_path': 'biu-nlp/lingmess-coref',
            'device': device
        }
    )
    
    doc = nlp(text)
    
    triplets = []

    # Process the text with coreference resolution
    doc = nlp(text, component_cfg={"fastcoref": {'resolve_text': True}})
    
    # Get the resolved text
    resolved_text = doc._.resolved_text
    
    # Process the resolved text
    doc = nlp(resolved_text)
    
    triplets = []
    processed_nouns = set()  # To avoid duplicate triplets
    
    # Process each sentence
    for sent in doc.sents:
        # Get all nouns and verbs in the sentence
        nouns = [token for token in sent if token.pos_ == "NOUN"]
        verbs = [token for token in sent if token.pos_ == "VERB"]
        
        # If we have at least 2 nouns and 1 verb, create triplets
        if len(nouns) >= 2 and len(verbs) >= 1:
            # Create triplets for each consecutive pair of nouns
            for i in range(len(nouns) - 1):
                # Use the first verb for the first triplet, or subsequent verbs for later triplets
                verb = verbs[min(i, len(verbs) - 1)]
                
                # Skip if first_node and second_node are the same
                if nouns[i].text == nouns[i + 1].text:
                    continue
                
                # Create a unique key for this triplet to avoid duplicates
                triplet_key = f"{nouns[i].text}_{verb.text}_{nouns[i + 1].text}"
                
                if triplet_key not in processed_nouns:
                    triplet = {
                        'first_node': nouns[i].text,
                        'relation': verb.text,
                        'second_node': nouns[i + 1].text
                    }
                    triplets.append(triplet)
                    processed_nouns.add(triplet_key)
    
    return triplets

def process_triplets_with_lemmatization(triplets: List[Dict[str, str]]) -> Tuple[List[Dict[str, str]], Dict[str, List[Tuple[str, str]]]]:
    """
    Process triplets to lemmatize relations while tracking original forms and POS tags.
    
    Args:
        triplets: List of dictionaries containing 'first_node', 'relation', and 'second_node'
    
    Returns:
        Tuple containing:
        - List of processed triplets with lemmatized relations
        - Dictionary mapping lemmatized forms to lists of (original_form, pos_tag) tuples
    """
    # Load English language model with lemmatizer
    nlp = spacy.load("en_core_web_sm", exclude=["parser", "ner", "textcat"])
    
    # Initialize tracking dictionary
    relation_tracking = defaultdict(list)
    
    # Process each triplet
    processed_triplets = []
    for triplet in triplets:
        # Create a new triplet with the same nodes
        processed_triplet = {
            'first_node': triplet['first_node'],
            'second_node': triplet['second_node']
        }
        
        # Process the relation
        relation_doc = nlp(triplet['relation'])
        if len(relation_doc) > 0:
            # Get the lemmatized form
            lemmatized_relation = relation_doc[0].lemma_
            processed_triplet['relation'] = lemmatized_relation
            
            # Track the original form and its POS tag
            relation_tracking[lemmatized_relation].append(
                (triplet['relation'], relation_doc[0].pos_)
            )
        else:
            processed_triplet['relation'] = triplet['relation']
        
        processed_triplets.append(processed_triplet)
    
    return processed_triplets, dict(relation_tracking)

def upload_to_neo4j(triplets: List[Dict[str, str]], relation_tracking: Dict[str, List[Tuple[str, str]]]) -> None:
    """
    Upload the knowledge graph triplets to Neo4j using credentials from .env file.
    
    Args:
        triplets: List of dictionaries containing 'first_node', 'relation', and 'second_node'
        relation_tracking: Dictionary mapping lemmatized forms to lists of (original_form, pos_tag) tuples
    """
    try:
        # Load environment variables
        load_dotenv()
        
        # Get Neo4j credentials from environment variables
        uri = os.getenv('NEO4J_URI')
        user = os.getenv('NEO4J_USER')
        password = os.getenv('NEO4J_PASSWORD')
        
        if not all([uri, user, password]):
            raise ValueError("Missing Neo4j credentials in .env file. Please ensure NEO4J_URI, NEO4J_USER, and NEO4J_PASSWORD are set.")
        
        # Create Neo4j driver
        driver = GraphDatabase.driver(uri, auth=(user, password))
        
        with driver.session() as session:
            # Create nodes and relationships
            for triplet in triplets:
                # Convert entity names to valid label names
                subject_label = re.sub(r'[^A-Za-z0-9_]', '_', triplet['first_node'].upper())
                object_label = re.sub(r'[^A-Za-z0-9_]', '_', triplet['second_node'].upper())
                
                # Ensure labels start with a letter
                if not subject_label[0].isalpha():
                    subject_label = 'E_' + subject_label
                if not object_label[0].isalpha():
                    object_label = 'E_' + object_label
                
                # Remove consecutive underscores and trailing underscores
                subject_label = re.sub(r'_+', '_', subject_label).rstrip('_')
                object_label = re.sub(r'_+', '_', object_label).rstrip('_')
                
                # Convert relation to valid Neo4j identifier
                rel_type = re.sub(r'[^A-Za-z0-9_]', '_', triplet['relation'].upper())
                if not rel_type[0].isalpha():
                    rel_type = 'REL_' + rel_type
                rel_type = re.sub(r'_+', '_', rel_type).rstrip('_')
                
                # Get the original forms and POS tags for this relation
                lemmatized_relation = triplet['relation']
                original_forms = relation_tracking.get(lemmatized_relation, [])
                
                # Store the first original form and POS tag as simple strings
                original_form = original_forms[0][0] if original_forms else triplet['relation']
                pos_tag = original_forms[0][1] if original_forms else 'VERB'
                
                # Create nodes with labels
                cypher_query = f"""
                MERGE (s:{subject_label} {{name: $subject}})
                SET s.text = $subject
                SET s.caption = $subject
                MERGE (o:{object_label} {{name: $object}})
                SET o.text = $object
                SET o.caption = $object
                CREATE (s)-[r:{rel_type}]->(o)
                SET r.type = $relation
                SET r.name = $relation
                SET r.caption = $relation
                SET r.original_form = $original_form
                SET r.pos_tag = $pos_tag
                SET r.strength = 1.0
                """
                
                session.run(cypher_query,
                          subject=triplet['first_node'],
                          object=triplet['second_node'],
                          relation=triplet['relation'],
                          original_form=original_form,
                          pos_tag=pos_tag)
            
            # Set display settings for all nodes
            session.run("""
            MATCH (n)
            SET n.displayName = n.name
            SET n.title = n.name
            """)
            
        driver.close()
        print("Successfully uploaded knowledge graph to Neo4j")
        
    except ServiceUnavailable:
        print("Could not connect to Neo4j database. Please check your connection details.")
        raise  # Re-raise the exception to be handled by the API endpoint
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        raise  # Re-raise the exception to be handled by the API endpoint

# Example usage
if __name__ == "__main__":
    sample_text = "John loves playing football. He also enjoys basketball. Mary reads books in the library. She often studies there."
    # First get the triplets
    triplets = create_triplets_spacy_fastcoref(sample_text)
    print("Original triplets:", triplets)
    
    # Then process them with lemmatization
    processed_triplets, relation_tracking = process_triplets_with_lemmatization(triplets)
    print("\nProcessed triplets:", processed_triplets)
    
    # Upload to Neo4j with relation tracking metadata
    upload_to_neo4j(processed_triplets, relation_tracking) 