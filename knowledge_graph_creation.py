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
from nebula3.gclient.net import ConnectionPool
from nebula3.Config import Config
import hashlib
import time

def create_triplets_spacy_fastcoref(text):
    print("\nStarting create_triplets_spacy_fastcoref")
    # Load English language model with minimal components
    print("Loading spaCy model...")
    nlp = spacy.load("en_core_web_sm", exclude=["parser", "lemmatizer", "ner", "textcat"])
    
    # Add sentencizer to the pipeline
    print("Adding sentencizer to pipeline...")
    nlp.add_pipe("sentencizer")
    
    # Try with CUDA first
    try:
        print("Checking CUDA availability...")
        if torch.cuda.is_available() and torch.cuda.is_initialized():
            device = 'cuda:0'
            print("CUDA is available, using GPU")
        else:
            device = 'cpu'
            print("CUDA is not available, using CPU")
            
        # Add FastCoref to the pipeline with LingMessCoref model
        print("Adding FastCoref to pipeline...")
        nlp.add_pipe(
            "fastcoref", 
            config={
                'model_architecture': 'LingMessCoref',
                'model_path': 'biu-nlp/lingmess-coref',
                'device': device
            }
        )
        
        # Process the text with coreference resolution
        print("Processing text with coreference resolution...")
        doc = nlp(text, component_cfg={"fastcoref": {'resolve_text': True}})
        print("Successfully processed text with coreference resolution")
        
    except Exception as e:
        print(f"Error during CUDA processing: {str(e)}")
        # If any error occurs (including CUDA errors), retry with CPU
        print("Retrying with CPU...")
        # Remove the existing FastCoref component
        if "fastcoref" in nlp.pipe_names:
            nlp.remove_pipe("fastcoref")
        # Add FastCoref to the pipeline with CPU configuration
        nlp.add_pipe(
            "fastcoref", 
            config={
                'model_architecture': 'LingMessCoref',
                'model_path': 'biu-nlp/lingmess-coref',
                'device': 'cpu'
            }
        )
        print("Processing text with CPU...")
        doc = nlp(text, component_cfg={"fastcoref": {'resolve_text': True}})
        print("Successfully processed text with CPU")
    
    # Get the resolved text
    print("Getting resolved text...")
    resolved_text = doc._.resolved_text
    
    # Process the resolved text
    print("Processing resolved text...")
    doc = nlp(resolved_text)
    
    triplets = []
    processed_nouns = set()  # To avoid duplicate triplets
    
    # Process each sentence
    print("Processing sentences to create triplets...")
    for sent in doc.sents:
        # Get all nouns and verbs in the sentence
        nouns = [token for token in sent if token.pos_ == "NOUN"]
        verbs = [token for token in sent if token.pos_ == "VERB"]
        
        # If we have at least 2 nouns and 1 verb, create triplets
        if len(nouns) >= 2 and len(verbs) >= 1:
            # Create triplets for each consecutive pair of nouns
            for i in range(len(nouns) - 1):
                verb = verbs[min(i, len(verbs) - 1)]
                if nouns[i].text == nouns[i + 1].text:
                    continue
                triplet_key = f"{nouns[i].text}_{verb.text}_{nouns[i + 1].text}"
                if triplet_key not in processed_nouns:
                    triplet = {
                        'first_node': nouns[i].text,
                        'relation': verb.text,
                        'second_node': nouns[i + 1].text
                    }
                    triplets.append(triplet)
                    processed_nouns.add(triplet_key)
    
    print(f"Created {len(triplets)} triplets")
    return triplets

def process_triplets_with_lemmatization(triplets: List[Dict[str, str]]) -> Tuple[List[Dict[str, str]], Dict[str, List[Tuple[str, str]]]]:
    print("\nStarting process_triplets_with_lemmatization")
    print("Loading spaCy model with lemmatizer...")
    nlp = spacy.load("en_core_web_sm", exclude=["parser", "ner", "textcat"])
    relation_tracking = defaultdict(list)
    print("Processing triplets with lemmatization...")
    processed_triplets = []
    for triplet in triplets:
        processed_triplet = {
            'first_node': triplet['first_node'],
            'second_node': triplet['second_node']
        }
        relation_doc = nlp(triplet['relation'])
        if len(relation_doc) > 0:
            lemmatized_relation = relation_doc[0].lemma_
            processed_triplet['relation'] = lemmatized_relation
            relation_tracking[lemmatized_relation].append(
                (triplet['relation'], relation_doc[0].pos_)
            )
        else:
            processed_triplet['relation'] = triplet['relation']
        processed_triplets.append(processed_triplet)
    
    print(f"Processed {len(processed_triplets)} triplets")
    return processed_triplets, dict(relation_tracking)

def upload_to_neo4j(triplets: List[Dict[str, str]], relation_tracking: Dict[str, List[Tuple[str, str]]]) -> None:
    print("\nStarting upload_to_neo4j")
    try:
        print("Loading environment variables...")
        load_dotenv()
        uri = os.getenv('NEO4J_URI')
        user = os.getenv('NEO4J_USER')
        password = os.getenv('NEO4J_PASSWORD')
        if not all([uri, user, password]):
            raise ValueError("Missing Neo4j credentials in .env file. Please ensure NEO4J_URI, NEO4J_USER, and NEO4J_PASSWORD are set.")
        print("Creating Neo4j driver...")
        driver = GraphDatabase.driver(uri, auth=(user, password))
        with driver.session() as session:
            print("Starting to upload triplets to Neo4j...")
            for triplet in triplets:
                subject_label = re.sub(r'[^A-Za-z0-9_]', '_', triplet['first_node'].upper())
                object_label = re.sub(r'[^A-Za-z0-9_]', '_', triplet['second_node'].upper())
                if not subject_label[0].isalpha():
                    subject_label = 'E_' + subject_label
                if not object_label[0].isalpha():
                    object_label = 'E_' + object_label
                subject_label = re.sub(r'_+', '_', subject_label).rstrip('_')
                object_label = re.sub(r'_+', '_', object_label).rstrip('_')
                rel_type = re.sub(r'[^A-Za-z0-9_]', '_', triplet['relation'].upper())
                if not rel_type[0].isalpha():
                    rel_type = 'REL_' + rel_type
                rel_type = re.sub(r'_+', '_', rel_type).rstrip('_')
                lemmatized_relation = triplet['relation']
                original_forms = relation_tracking.get(lemmatized_relation, [])
                original_form = original_forms[0][0] if original_forms else triplet['relation']
                pos_tag = original_forms[0][1] if original_forms else 'VERB'
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
            
            print("Setting display settings for nodes...")
            session.run("""
            MATCH (n)
            SET n.displayName = n.name
            SET n.title = n.name
            """)
            
        driver.close()
        print("Successfully uploaded knowledge graph to Neo4j")
        
    except ServiceUnavailable:
        print("Could not connect to Neo4j database. Please check your connection details.")
        raise
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        raise

def extract_value(v) -> str:
    """
    Extract the inner string from Nebula Graph Value objects.
    For example, turns "Value(sVal=b'AGREEMENT')" into "AGREEMENT".
    """
    s = str(v)
    match = re.search(r"b'(.+?)'", s)
    if match:
        return match.group(1)
    return s

def sanitize_vertex(name: str) -> str:
    """
    Sanitize vertex label names.
    """
    name = re.sub(r'[^A-Za-z0-9_]', '_', name.upper())
    name = re.sub(r'_+', '_', name).strip('_')
    if not name or not name[0].isalpha():
        name = 'E_' + name
    return name

def sanitize_edge(name: str) -> str:
    """
    Sanitize edge type names.
    """
    name = re.sub(r'[^A-Za-z0-9_]', '_', name.upper())
    name = re.sub(r'_+', '_', name).strip('_')
    return "REL_" + name if name else "REL_UNKNOWN"

def upload_to_nebula(triplets: list, relation_tracking: dict) -> None:
    print("\nStarting upload_to_nebula")
    try:
        load_dotenv()
        host = os.getenv('NEBULA_HOST')
        port = int(os.getenv('NEBULA_PORT', '9669'))
        user = os.getenv('NEBULA_USER')
        password = os.getenv('NEBULA_PASSWORD')
        space = os.getenv('NEBULA_SPACE')
    
        if not all([host, user, password, space]):
            raise ValueError("Missing Nebula Graph credentials")
    
        config = Config()
        connection_pool = ConnectionPool()
        assert connection_pool.init([(host, port)], config)
        session = connection_pool.get_session(user, password)
    
        # Use the specified space
        resp = session.execute(f"USE `{space}`")
        if not resp.is_succeeded():
            raise RuntimeError(f"Cannot access space `{space}`: {resp.error_msg()}")
    
        # --- Drop existing tags and edges ---
        resp = session.execute("SHOW TAGS")
        if resp.is_succeeded():
            tags = [extract_value(row.values[0]) for row in resp.rows()]
            for tag in tags:
                drop_query = f"DROP TAG IF EXISTS `{tag}`"
                resp_drop = session.execute(drop_query)
                if not resp_drop.is_succeeded():
                    print(f"Warning: Failed to drop tag {tag}: {resp_drop.error_msg()}")
    
        resp = session.execute("SHOW EDGES")
        if resp.is_succeeded():
            edges = [extract_value(row.values[0]) for row in resp.rows()]
            for edge in edges:
                drop_query = f"DROP EDGE IF EXISTS `{edge}`"
                resp_drop = session.execute(drop_query)
                if not resp_drop.is_succeeded():
                    print(f"Warning: Failed to drop edge {edge}: {resp_drop.error_msg()}")
    
        print(f"Successfully initialized/cleared space '{space}'")
    
        # --- Build expected schema ---
        vertex_labels = set()
        edge_types = set()
        for triplet in triplets:
            vertex_labels.add(sanitize_vertex(triplet['first_node']))
            vertex_labels.add(sanitize_vertex(triplet['second_node']))
            edge_types.add(sanitize_edge(triplet['relation']))
    
        # --- Create vertex tags (schema for vertices) ---
        for label in vertex_labels:
            query = f"CREATE TAG IF NOT EXISTS `{label}` (name string, text string, caption string, displayName string, title string)"
            result = session.execute(query)
            if not result.is_succeeded():
                print(f"Failed to create tag {label}: {result.error_msg()}")
    
        # --- Create edge types (schema for edges) ---
        for edge in edge_types:
            query = f"CREATE EDGE IF NOT EXISTS `{edge}` (type string, name string, caption string, original_form string, pos_tag string, strength double)"
            result = session.execute(query)
            if not result.is_succeeded():
                print(f"Failed to create edge {edge}: {result.error_msg()}")
    
        # --- Wait for schema propagation ---
        def wait_for_schema_propagation(timeout: int = 30) -> bool:
            start_time = time.time()
            while time.time() - start_time < timeout:
                session.execute(f"USE `{space}`")
                tag_resp = session.execute("SHOW TAGS")
                current_tags = set()
                if tag_resp.is_succeeded():
                    current_tags = set(extract_value(row.values[0]) for row in tag_resp.rows())
                edge_resp = session.execute("SHOW EDGES")
                current_edges = set()
                if edge_resp.is_succeeded():
                    current_edges = set(extract_value(row.values[0]) for row in edge_resp.rows())
                if vertex_labels.issubset(current_tags) and edge_types.issubset(current_edges):
                    return True
                time.sleep(2)
            return False
    
        if not wait_for_schema_propagation():
            print("Warning: Schema propagation timed out. Some insertions might fail.")
        else:
            print("Schema propagation successful.")
    
        # --- Re-establish session so that the new schema is recognized ---
        session.release()
        session = connection_pool.get_session(user, password)
        session.execute(f"USE `{space}`")
        # Added extra delay to ensure the schema is fully visible
        time.sleep(5)
    
        import hashlib
        for triplet in triplets:
            sub_name = triplet['first_node'].replace('"', '\\"')
            obj_name = triplet['second_node'].replace('"', '\\"')
            rel = triplet['relation'].replace('"', '\\"')
    
            sub_label = sanitize_vertex(sub_name)
            obj_label = sanitize_vertex(obj_name)
            edge_type = sanitize_edge(rel)
    
            original_forms = relation_tracking.get(rel, [])
            original_form = original_forms[0][0] if original_forms else rel
            pos_tag = original_forms[0][1] if original_forms else 'VERB'
    
            sub_id = f"v_{int(hashlib.sha256(sub_name.encode()).hexdigest()[:8], 16) % 1000000}"
            obj_id = f"v_{int(hashlib.sha256(obj_name.encode()).hexdigest()[:8], 16) % 1000000}"
    
            insert_subject = (
                f'INSERT VERTEX `{sub_label}`(name, text, caption, displayName, title) '
                f'VALUES "{sub_id}":("{sub_name}", "{sub_name}", "{sub_name}", "{sub_name}", "{sub_name}")'
            )
            insert_object = (
                f'INSERT VERTEX `{obj_label}`(name, text, caption, displayName, title) '
                f'VALUES "{obj_id}":("{obj_name}", "{obj_name}", "{obj_name}", "{obj_name}", "{obj_name}")'
            )
            insert_edge = (
                f'INSERT EDGE `{edge_type}`(type, name, caption, original_form, pos_tag, strength) '
                f'VALUES "{sub_id}" -> "{obj_id}": ("{rel}", "{rel}", "{rel}", "{original_form}", "{pos_tag}", 1.0)'
            )
    
            for query in [insert_subject, insert_object, insert_edge]:
                resp = session.execute(query)
                if not resp.is_succeeded():
                    print(f"Error running query:\n{query}\nReason: {resp.error_msg()}")
    
        print("Successfully uploaded triplets to Nebula Graph.")
        session.release()
        connection_pool.close()
    
    except Exception as e:
        print(f"upload_to_nebula error: {str(e)}")
        raise

def upload_to_database(triplets: List[Dict[str, str]], relation_tracking: Dict[str, List[Tuple[str, str]]]) -> None:
    """
    Upload triplets to the database(s) specified in the DB_TYPE environment variable.
    """
    print("\nStarting upload_to_database")
    load_dotenv()
    db_type = os.getenv('DB_TYPE', 'neo4j').lower()
    if db_type == 'neo4j':
        upload_to_neo4j(triplets, relation_tracking)
    elif db_type == 'nebula':
        upload_to_nebula(triplets, relation_tracking)
    elif db_type == 'both':
        upload_to_neo4j(triplets, relation_tracking)
        upload_to_nebula(triplets, relation_tracking)
    else:
        raise ValueError(f"Invalid DB_TYPE: {db_type}. Must be 'neo4j', 'nebula', or 'both'.")

# Example usage when running this module directly:
if __name__ == "__main__":
    sample_text = ("John loves playing football. He also enjoys basketball. "
                   "Mary reads books in the library. She often studies there.")
    triplets = create_triplets_spacy_fastcoref(sample_text)
    print("Original triplets:", triplets)
    processed_triplets, relation_tracking = process_triplets_with_lemmatization(triplets)
    print("\nProcessed triplets:", processed_triplets)
    upload_to_database(processed_triplets, relation_tracking)
