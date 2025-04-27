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
import json
import requests

def create_resolved_text_spacy_fastcoref(text):
    """
    Process text with spaCy and FastCoref to resolve coreferences.
    
    Args:
        text: Input text to process
        
    Returns:
        Resolved text with coreferences resolved
    """
    print("\nStarting create_resolved_text_spacy_fastcoref")
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
    
    print("Successfully created resolved text")
    return resolved_text

def create_triplets_stanford_corenlp(text):
    """
    Process text with Stanford CoreNLP to extract triplets.
    
    Args:
        text: Input text to process
        
    Returns:
        List of triplets extracted from the text
    """
    print("\nStarting create_triplets_stanford_corenlp")
    print("Sending text to Stanford CoreNLP...")
    
    # Stanford CoreNLP server URL
    corenlp_url = "http://localhost:9000"
    
    # Prepare the request with dependency parsing
    properties = {
        "annotators": "tokenize,ssplit,pos,lemma,depparse",
        "outputFormat": "json"
    }
    
    # Send the request to Stanford CoreNLP
    try:
        response = requests.post(
            f"{corenlp_url}/?properties={json.dumps(properties)}",
            data=text.encode('utf-8'),
            headers={'Content-Type': 'application/x-www-form-urlencoded; charset=UTF-8'}
        )
        
        if response.status_code != 200:
            print(f"Error from Stanford CoreNLP: {response.status_code}")
            print(f"Response: {response.text}")
            return []
            
        # Parse the response
        result = response.json()
        
        triplets = []
        processed_nouns = set()  # To avoid duplicate triplets
        
        # Process each sentence
        for sentence in result.get('sentences', []):
            tokens = sentence.get('tokens', [])
            dependencies = sentence.get('enhancedPlusPlusDependencies', [])
            
            # Create a dictionary to store token dependencies
            token_deps = {}
            for dep in dependencies:
                # Skip ROOT and punctuation dependencies
                if dep.get('dep') in ['ROOT', 'punct']:
                    continue
                    
                # Get the indices from the correct fields
                dependent_idx = dep.get('dependent', 0)
                governor_idx = dep.get('governor', 0)
                
                # Only process if we have valid indices
                if dependent_idx > 0 and governor_idx > 0:
                    # Convert to 0-based index for our token list
                    dependent_idx = dependent_idx - 1
                    governor_idx = governor_idx - 1
                    
                    if dependent_idx not in token_deps:
                        token_deps[dependent_idx] = []
                    token_deps[dependent_idx].append({
                        'governor_idx': governor_idx,
                        'relation': dep.get('dep'),
                        'governor': dep.get('governor')
                    })
            
            # Extract nouns, verbs, adjectives, and adverbs
            nouns = []
            verbs = []
            adjectives = []
            adverbs = []
            
            for idx, token in enumerate(tokens):
                pos = token.get('pos', '')
                word = token.get('word', '')
                
                # Stanford CoreNLP POS tags:
                # NN, NNS, NNP, NNPS for nouns
                # VB, VBD, VBG, VBN, VBP, VBZ for verbs
                # JJ, JJR, JJS for adjectives
                # RB, RBR, RBS for adverbs
                if pos.startswith('NN'):
                    nouns.append((word, idx))
                elif pos.startswith('VB'):
                    verbs.append((word, idx))
                elif pos.startswith('JJ'):
                    adjectives.append((word, idx))
                elif pos.startswith('RB'):
                    adverbs.append((word, idx))
            
            # Create a dictionary to store metadata for nouns and verbs
            metadata = {}
            
            # Process adjectives and adverbs to find those that depend on nouns or verbs
            for adj in adjectives:
                adj_word, adj_idx = adj
                if adj_idx in token_deps:
                    for dep_info in token_deps[adj_idx]:
                        governor_idx = dep_info['governor_idx']
                        governor_word = tokens[governor_idx].get('word', '')
                        governor_pos = tokens[governor_idx].get('pos', '')
                        
                        # Check if the adjective depends on a noun or verb
                        if governor_pos.startswith('NN') or governor_pos.startswith('VB'):
                            # Add the adjective as metadata to the noun or verb
                            if governor_word not in metadata:
                                metadata[governor_word] = {'adjective': "", 'adverb': ""}
                            # Only set if not already set (keep first one)
                            if not metadata[governor_word]['adjective']:
                                metadata[governor_word]['adjective'] = adj_word
            
            for adv in adverbs:
                adv_word, adv_idx = adv
                if adv_idx in token_deps:
                    for dep_info in token_deps[adv_idx]:
                        governor_idx = dep_info['governor_idx']
                        governor_word = tokens[governor_idx].get('word', '')
                        governor_pos = tokens[governor_idx].get('pos', '')
                        
                        # Check if the adverb depends on a noun or verb
                        if governor_pos.startswith('NN') or governor_pos.startswith('VB'):
                            # Add the adverb as metadata to the noun or verb
                            if governor_word not in metadata:
                                metadata[governor_word] = {'adjective': "", 'adverb': ""}
                            # Only set if not already set (keep first one)
                            if not metadata[governor_word]['adverb']:
                                metadata[governor_word]['adverb'] = adv_word
            
            for noun in nouns:
                noun_word, noun_idx = noun
                if noun_idx in token_deps:
                    for dep_info in token_deps[noun_idx]:
                        governor_idx = dep_info['governor_idx']
                        governor_word = tokens[governor_idx].get('word', '')
                        governor_pos = tokens[governor_idx].get('pos', '')
                        
                        # Check if the noun depends on an adjective or adverb
                        if governor_pos.startswith('JJ') or governor_pos.startswith('RB'):
                            # Add the adjective or adverb as metadata to the noun
                            if noun_word not in metadata:
                                metadata[noun_word] = {'adjective': "", 'adverb': ""}
                            
                            # Only set if not already set (keep first one)
                            if governor_pos.startswith('JJ'):
                                if not metadata[noun_word]['adjective']:
                                    metadata[noun_word]['adjective'] = governor_word
                            else:  # RB
                                if not metadata[noun_word]['adverb']:
                                    metadata[noun_word]['adverb'] = governor_word
            
            for verb in verbs:
                verb_word, verb_idx = verb
                if verb_idx in token_deps:
                    for dep_info in token_deps[verb_idx]:
                        governor_idx = dep_info['governor_idx']
                        governor_word = tokens[governor_idx].get('word', '')
                        governor_pos = tokens[governor_idx].get('pos', '')
                        
                        # Check if the verb depends on an adjective or adverb
                        if governor_pos.startswith('JJ') or governor_pos.startswith('RB'):
                            # Add the adjective or adverb as metadata to the verb
                            if verb_word not in metadata:
                                metadata[verb_word] = {'adjective': "", 'adverb': ""}
                            
                            # Only set if not already set (keep first one)
                            if governor_pos.startswith('JJ'):
                                if not metadata[verb_word]['adjective']:
                                    metadata[verb_word]['adjective'] = governor_word
                            else:  # RB
                                if not metadata[verb_word]['adverb']:
                                    metadata[verb_word]['adverb'] = governor_word
            
            # If we have at least 2 nouns and 1 verb, create triplets
            if len(nouns) >= 2 and len(verbs) >= 1:
                # Create triplets for each consecutive pair of nouns
                for i in range(len(nouns) - 1):
                    # Find the closest verb to the first noun
                    closest_verb = min(verbs, key=lambda v: abs(v[1] - nouns[i][1]))
                    
                    if nouns[i][0] == nouns[i + 1][0]:
                        continue
                        
                    triplet_key = f"{nouns[i][0]}_{closest_verb[0]}_{nouns[i + 1][0]}"
                    if triplet_key not in processed_nouns:
                        triplet = {
                            'first_node': nouns[i][0],
                            'relation': closest_verb[0],
                            'second_node': nouns[i + 1][0]
                        }
                        
                        # Add metadata for the first node (subject)
                        if nouns[i][0] in metadata:
                            triplet['first_node_metadata'] = metadata[nouns[i][0]]
                        
                        # Add metadata for the relation (verb)
                        if closest_verb[0] in metadata:
                            triplet['relation_metadata'] = metadata[closest_verb[0]]
                        
                        # Add metadata for the second node (object)
                        if nouns[i + 1][0] in metadata:
                            triplet['second_node_metadata'] = metadata[nouns[i + 1][0]]
                        
                        triplets.append(triplet)
                        processed_nouns.add(triplet_key)
        
        print(f"Created {len(triplets)} triplets")
        return triplets
        
    except Exception as e:
        print(f"Error processing with Stanford CoreNLP: {str(e)}")
        return []

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
        
        # Process the relation with lemmatization
        relation_doc = nlp(triplet['relation'])
        if len(relation_doc) > 0:
            lemmatized_relation = relation_doc[0].lemma_
            processed_triplet['relation'] = lemmatized_relation
            relation_tracking[lemmatized_relation].append(
                (triplet['relation'], relation_doc[0].pos_)
            )
        else:
            processed_triplet['relation'] = triplet['relation']
        
        # Copy metadata fields if they exist
        if 'first_node_metadata' in triplet:
            processed_triplet['first_node_metadata'] = triplet['first_node_metadata']
        
        if 'relation_metadata' in triplet:
            processed_triplet['relation_metadata'] = triplet['relation_metadata']
        
        if 'second_node_metadata' in triplet:
            processed_triplet['second_node_metadata'] = triplet['second_node_metadata']
        
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
                
                # Prepare metadata for subject (first node)
                subject_adjectives = []
                subject_adverbs = []
                if 'first_node_metadata' in triplet:
                    metadata = triplet['first_node_metadata']
                    subject_adjectives = metadata.get('adjective', "") or metadata.get('adjectives', [""])[0]
                    subject_adverbs = metadata.get('adverb', "") or metadata.get('adverbs', [""])[0]
                
                # Prepare metadata for object (second node)
                object_adjectives = []
                object_adverbs = []
                if 'second_node_metadata' in triplet:
                    metadata = triplet['second_node_metadata']
                    object_adjectives = metadata.get('adjective', "") or metadata.get('adjectives', [""])[0]
                    object_adverbs = metadata.get('adverb', "") or metadata.get('adverbs', [""])[0]
                
                # Prepare metadata for relation (verb)
                relation_adjectives = []
                relation_adverbs = []
                if 'relation_metadata' in triplet:
                    metadata = triplet['relation_metadata']
                    relation_adjectives = metadata.get('adjective', "") or metadata.get('adjectives', [""])[0]
                    relation_adverbs = metadata.get('adverb', "") or metadata.get('adverbs', [""])[0]
                
                cypher_query = f"""
                MERGE (s:{subject_label} {{name: $subject}})
                SET s.text = $subject
                SET s.caption = $subject
                SET s.adjectives = $subject_adjectives
                SET s.adverbs = $subject_adverbs
                MERGE (o:{object_label} {{name: $object}})
                SET o.text = $object
                SET o.caption = $object
                SET o.adjectives = $object_adjectives
                SET o.adverbs = $object_adverbs
                CREATE (s)-[r:{rel_type}]->(o)
                SET r.type = $relation
                SET r.name = $relation
                SET r.caption = $relation
                SET r.original_form = $original_form
                SET r.pos_tag = $pos_tag
                SET r.strength = 1.0
                SET r.adjectives = $relation_adjectives
                SET r.adverbs = $relation_adverbs
                """
                session.run(cypher_query,
                          subject=triplet['first_node'],
                          object=triplet['second_node'],
                          relation=triplet['relation'],
                          original_form=original_form,
                          pos_tag=pos_tag,
                          subject_adjectives=subject_adjectives,
                          subject_adverbs=subject_adverbs,
                          object_adjectives=object_adjectives,
                          object_adverbs=object_adverbs,
                          relation_adjectives=relation_adjectives,
                          relation_adverbs=relation_adverbs)
            
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
    return name if name else "REL_UNKNOWN"

def create_nebula_schema(session=None, connection_pool=None):
    """
    Create the NOUN tag and VERB/ADJECTIVE edges in Nebula Graph.
    This function should be called once before processing any batches.
    
    Args:
        session: Optional Nebula session to reuse (if None, a new session will be created)
        connection_pool: Optional connection pool to reuse (if None, a new pool will be created)
        
    Returns:
        Tuple of (noun_tag, verb_edge, adjective_edge) that were created
    """
    # Load environment variables with override=True to force reload
    load_dotenv(override=True)
    
    # Get Nebula Graph credentials
    host = os.getenv('NEBULA_HOST')
    port = int(os.getenv('NEBULA_PORT', '9669'))
    user = os.getenv('NEBULA_USER')
    password = os.getenv('NEBULA_PASSWORD')
    space = os.getenv('NEBULA_SPACE')
    
    if not all([host, user, password, space]):
        print("Error: Missing Nebula Graph credentials in .env file")
        print(f"Host: {'Present' if host else 'Missing'}")
        print(f"User: {'Present' if user else 'Missing'}")
        print(f"Password: {'Present' if password else 'Missing'}")
        print(f"Space: {'Present' if space else 'Missing'}")
        raise ValueError("Missing Nebula Graph credentials in .env file")
    
    # Create connection pool and session if not provided
    should_close = False
    if connection_pool is None:
        print(f"Attempting to connect to Nebula Graph at {host}:{port}")
        config = Config()
        connection_pool = ConnectionPool()
        
        # Initialize the connection pool
        init_result = connection_pool.init([(host, port)], config)
        if not init_result:
            raise ConnectionError(f"Failed to initialize connection pool to Nebula Graph at {host}:{port}")
        
        # Get a session from the pool
        session = connection_pool.get_session(user, password)
        should_close = True
    
    try:
        # Use the space
        resp = session.execute(f"USE {space}")
        if not resp.is_succeeded():
            raise Exception(f"Failed to use space {space}: {resp.error_msg()}")
        
        # Helper function to retry an operation until it succeeds or timeout is reached
        def retry_operation(operation_func, operation_name, max_timeout=100):
            """
            Retry an operation until it succeeds or timeout is reached.
            
            Args:
                operation_func: Function that performs the operation and returns (success, result)
                operation_name: Name of the operation for logging
                max_timeout: Maximum time in seconds to retry
                
            Returns:
                Tuple of (success, result)
            """
            start_time = time.time()
            attempt = 1
            
            while time.time() - start_time < max_timeout:
                print(f"Attempt {attempt} for {operation_name}...")
                success, result = operation_func()
                
                if success:
                    print(f"Successfully completed {operation_name} on attempt {attempt}")
                    return True, result
                
                # If we get here, the operation failed
                elapsed = time.time() - start_time
                remaining = max_timeout - elapsed
                
                if remaining > 0:
                    # Sleep briefly before retrying, but not too long
                    sleep_time = min(1, remaining)
                    print(f"Operation {operation_name} failed: {result}. Retrying in {sleep_time:.1f} seconds...")
                    time.sleep(sleep_time)
                else:
                    print(f"Operation {operation_name} failed after {elapsed:.1f} seconds: {result}")
                    return False, result
                
                attempt += 1
            
            print(f"Operation {operation_name} timed out after {max_timeout} seconds")
            return False, f"Operation timed out after {max_timeout} seconds"
        
        # Create NOUN tag
        def create_noun_tag():
            query = "CREATE TAG IF NOT EXISTS NOUN (name string, text string, caption string, displayName string, title string, adjective string, adverb string)"
            result = session.execute(query)
            
            if not result.is_succeeded():
                # Add prefix for any error, not just syntax errors
                print(f"Trying with VER_ prefix for tag NOUN...")
                query_with_prefix = "CREATE TAG IF NOT EXISTS VER_NOUN (name string, text string, caption string, displayName string, title string, adjective string, adverb string)"
                result_with_prefix = session.execute(query_with_prefix)
                if result_with_prefix.is_succeeded():
                    return True, result_with_prefix
                return False, result_with_prefix.error_msg()
            
            return True, result
        
        success, result = retry_operation(create_noun_tag, "creating NOUN tag")
        if success:
            noun_tag = "NOUN"
            print("Waiting 30 seconds for NOUN tag to be fully propagated...")
            time.sleep(30)
        else:
            print(f"Failed to create NOUN tag after multiple attempts: {result}")
            # Try with prefix
            success, result = retry_operation(lambda: session.execute("CREATE TAG IF NOT EXISTS VER_NOUN (name string, text string, caption string, displayName string, title string, adjective string, adverb string)"), "creating VER_NOUN tag")
            if success:
                noun_tag = "VER_NOUN"
                print("Waiting 30 seconds for VER_NOUN tag to be fully propagated...")
                time.sleep(30)
            else:
                print(f"Failed to create VER_NOUN tag after multiple attempts: {result}")
                raise Exception("Failed to create NOUN tag")
        
        # Create VERB edge
        def create_verb_edge():
            query = "CREATE EDGE IF NOT EXISTS VERB (type string, name string, caption string, original_form string, pos_tag string, strength double, adjective string, adverb string)"
            result = session.execute(query)
            
            if not result.is_succeeded():
                # Add prefix for any error, not just syntax errors
                print(f"Trying with REL_ prefix for edge VERB...")
                query_with_prefix = "CREATE EDGE IF NOT EXISTS REL_VERB (type string, name string, caption string, original_form string, pos_tag string, strength double, adjective string, adverb string)"
                result_with_prefix = session.execute(query_with_prefix)
                if result_with_prefix.is_succeeded():
                    return True, result_with_prefix
                return False, result_with_prefix.error_msg()
            
            return True, result
        
        success, result = retry_operation(create_verb_edge, "creating VERB edge")
        if success:
            verb_edge = "VERB"
            print("Waiting 30 seconds for VERB edge to be fully propagated...")
            time.sleep(30)
        else:
            print(f"Failed to create VERB edge after multiple attempts: {result}")
            # Try with prefix
            success, result = retry_operation(lambda: session.execute("CREATE EDGE IF NOT EXISTS REL_VERB (type string, name string, caption string, original_form string, pos_tag string, strength double, adjective string, adverb string)"), "creating REL_VERB edge")
            if success:
                verb_edge = "REL_VERB"
                print("Waiting 30 seconds for REL_VERB edge to be fully propagated...")
                time.sleep(30)
            else:
                print(f"Failed to create REL_VERB edge after multiple attempts: {result}")
                raise Exception("Failed to create VERB edge")
        
        # Create ADJECTIVE edge
        def create_adjective_edge():
            query = "CREATE EDGE IF NOT EXISTS ADJECTIVE (type string, name string, caption string, original_form string, pos_tag string, strength double, adjective string, adverb string)"
            result = session.execute(query)
            
            if not result.is_succeeded():
                # Add prefix for any error, not just syntax errors
                print(f"Trying with REL_ prefix for edge ADJECTIVE...")
                query_with_prefix = "CREATE EDGE IF NOT EXISTS REL_ADJECTIVE (type string, name string, caption string, original_form string, pos_tag string, strength double, adjective string, adverb string)"
                result_with_prefix = session.execute(query_with_prefix)
                if result_with_prefix.is_succeeded():
                    return True, result_with_prefix
                return False, result_with_prefix.error_msg()
            
            return True, result
        
        success, result = retry_operation(create_adjective_edge, "creating ADJECTIVE edge")
        if success:
            adjective_edge = "ADJECTIVE"
            print("Waiting 30 seconds for ADJECTIVE edge to be fully propagated...")
            time.sleep(30)
        else:
            print(f"Failed to create ADJECTIVE edge after multiple attempts: {result}")
            # Try with prefix
            success, result = retry_operation(lambda: session.execute("CREATE EDGE IF NOT EXISTS REL_ADJECTIVE (type string, name string, caption string, original_form string, pos_tag string, strength double, adjective string, adverb string)"), "creating REL_ADJECTIVE edge")
            if success:
                adjective_edge = "REL_ADJECTIVE"
                print("Waiting 30 seconds for REL_ADJECTIVE edge to be fully propagated...")
                time.sleep(30)
            else:
                print(f"Failed to create REL_ADJECTIVE edge after multiple attempts: {result}")
                raise Exception("Failed to create ADJECTIVE edge")
        
        print("Successfully created Nebula Graph schema")
        
        # Only release and close if we created the session and connection pool
        if should_close:
            session.release()
            connection_pool.close()
        
        return noun_tag, verb_edge, adjective_edge
    
    except Exception as e:
        print(f"create_nebula_schema error: {str(e)}")
        # Only release and close if we created the session and connection pool
        if should_close and session is not None:
            session.release()
        if should_close and connection_pool is not None:
            connection_pool.close()
        raise

def upload_to_nebula(triplets: list, relation_tracking: dict, session=None, connection_pool=None) -> None:
    """
    Upload triplets to Nebula Graph.
    
    Args:
        triplets: List of triplets to upload
        relation_tracking: Dictionary tracking relations and their original forms
        session: Optional Nebula session to reuse (if None, a new session will be created)
        connection_pool: Optional connection pool to reuse (if None, a new pool will be created)
    """
    # Load environment variables with override=True to force reload
    load_dotenv(override=True)
    
    # Get Nebula Graph credentials
    host = os.getenv('NEBULA_HOST')
    port = int(os.getenv('NEBULA_PORT', '9669'))
    user = os.getenv('NEBULA_USER')
    password = os.getenv('NEBULA_PASSWORD')
    space = os.getenv('NEBULA_SPACE')
    
    if not all([host, user, password, space]):
        print("Error: Missing Nebula Graph credentials in .env file")
        print(f"Host: {'Present' if host else 'Missing'}")
        print(f"User: {'Present' if user else 'Missing'}")
        print(f"Password: {'Present' if password else 'Missing'}")
        print(f"Space: {'Present' if space else 'Missing'}")
        raise ValueError("Missing Nebula Graph credentials in .env file")
    
    # Create connection pool and session if not provided
    should_close = False
    if connection_pool is None:
        print(f"Attempting to connect to Nebula Graph at {host}:{port}")
        config = Config()
        connection_pool = ConnectionPool()
        
        # Initialize the connection pool
        init_result = connection_pool.init([(host, port)], config)
        if not init_result:
            raise ConnectionError(f"Failed to initialize connection pool to Nebula Graph at {host}:{port}")
        
        # Get a session from the pool
        session = connection_pool.get_session(user, password)
        should_close = True
    
    try:
        # Use the space
        resp = session.execute(f"USE {space}")
        if not resp.is_succeeded():
            raise Exception(f"Failed to use space {space}: {resp.error_msg()}")
    
        # Use default tag and edge names
        # These should have been created by create_nebula_schema before this function is called
        noun_tag = "NOUN"
        verb_edge = "VERB"
        adjective_edge = "ADJECTIVE"
        
        # Helper function to retry an operation until it succeeds or timeout is reached
        def retry_operation(operation_func, operation_name, max_timeout=100):
            """
            Retry an operation until it succeeds or timeout is reached.
            
            Args:
                operation_func: Function that performs the operation and returns (success, result)
                operation_name: Name of the operation for logging
                max_timeout: Maximum time in seconds to retry
                
            Returns:
                Tuple of (success, result)
            """
            start_time = time.time()
            attempt = 1
            
            while time.time() - start_time < max_timeout:
                print(f"Attempt {attempt} for {operation_name}...")
                success, result = operation_func()
                
                if success:
                    print(f"Successfully completed {operation_name} on attempt {attempt}")
                    return True, result
                
                # If we get here, the operation failed
                elapsed = time.time() - start_time
                remaining = max_timeout - elapsed
                
                if remaining > 0:
                    # Sleep briefly before retrying, but not too long
                    sleep_time = min(1, remaining)
                    print(f"Operation {operation_name} failed: {result}. Retrying in {sleep_time:.1f} seconds...")
                    time.sleep(sleep_time)
                else:
                    print(f"Operation {operation_name} failed after {elapsed:.1f} seconds: {result}")
                    return False, result
                
                attempt += 1
            
            print(f"Operation {operation_name} timed out after {max_timeout} seconds")
            return False, f"Operation timed out after {max_timeout} seconds"
        
        # Process each triplet
        for triplet in triplets:
            sub_name = triplet['first_node']
            obj_name = triplet['second_node']
            rel = triplet['relation']
            rel_type = triplet.get('relation_type', 'VERB')  # Default to VERB if not specified
            
            # Sanitize names
            sub_label = noun_tag  # Use NOUN tag for all vertices
            obj_label = noun_tag  # Use NOUN tag for all vertices
            
            # Determine which edge type to use based on the relation_type
            if rel_type == 'ADJECTIVE':
                edge_type = adjective_edge
            else:
                edge_type = verb_edge
            
            # Insert vertices and edge
            original_forms = relation_tracking.get(rel, [])
            original_form = original_forms[0][0] if original_forms else rel
            pos_tag = original_forms[0][1] if original_forms else rel_type
            
            # Get metadata for subject (first node)
            subject_adjective = ""
            subject_adverb = ""
            if 'first_node_metadata' in triplet:
                metadata = triplet['first_node_metadata']
                subject_adjective = metadata.get('adjective', "") or metadata.get('adjectives', [""])[0]
                subject_adverb = metadata.get('adverb', "") or metadata.get('adverbs', [""])[0]
            
            # Get metadata for object (second node)
            object_adjective = ""
            object_adverb = ""
            if 'second_node_metadata' in triplet:
                metadata = triplet['second_node_metadata']
                object_adjective = metadata.get('adjective', "") or metadata.get('adjectives', [""])[0]
                object_adverb = metadata.get('adverb', "") or metadata.get('adverbs', [""])[0]
            
            # Get metadata for relation (verb)
            relation_adjective = ""
            relation_adverb = ""
            if 'relation_metadata' in triplet:
                metadata = triplet['relation_metadata']
                relation_adjective = metadata.get('adjective', "") or metadata.get('adjectives', [""])[0]
                relation_adverb = metadata.get('adverb', "") or metadata.get('adverbs', [""])[0]
            
            sub_id = f"v_{int(hashlib.sha256(sub_name.encode()).hexdigest()[:8], 16) % 1000000}"
            obj_id = f"v_{int(hashlib.sha256(obj_name.encode()).hexdigest()[:8], 16) % 1000000}"
            
            # Insert subject vertex
            def insert_subject_vertex():
                insert_subject = (
                    f'INSERT VERTEX {sub_label} (name, text, caption, displayName, title, adjective, adverb) VALUES "{sub_id}":("{sub_name}", "{sub_name}", "{sub_name}", "{sub_name}", "{sub_name}", "{subject_adjective}", "{subject_adverb}")'
                )
                resp = session.execute(insert_subject)
                return resp.is_succeeded(), resp.error_msg() if not resp.is_succeeded() else None
            
            success, result = retry_operation(insert_subject_vertex, f"inserting subject vertex {sub_id}")
            if not success:
                print(f"Failed to insert subject vertex {sub_id} after multiple attempts: {result}")
                # Try with VER_ prefix for the tag
                print(f"Trying with VER_ prefix for subject vertex {sub_id}...")
                def insert_subject_vertex_with_prefix():
                    insert_subject = (
                        f'INSERT VERTEX VER_{sub_label} (name, text, caption, displayName, title, adjective, adverb) VALUES "{sub_id}":("{sub_name}", "{sub_name}", "{sub_name}", "{sub_name}", "{sub_name}", "{subject_adjective}", "{subject_adverb}")'
                    )
                    resp = session.execute(insert_subject)
                    return resp.is_succeeded(), resp.error_msg() if not resp.is_succeeded() else None
                
                success, result = retry_operation(insert_subject_vertex_with_prefix, f"inserting subject vertex {sub_id} with VER_ prefix")
                if not success:
                    print(f"Failed to insert subject vertex {sub_id} with VER_ prefix after multiple attempts: {result}")
            
            # Insert object vertex
            def insert_object_vertex():
                insert_object = (
                    f'INSERT VERTEX {obj_label} (name, text, caption, displayName, title, adjective, adverb) VALUES "{obj_id}":("{obj_name}", "{obj_name}", "{obj_name}", "{obj_name}", "{obj_name}", "{object_adjective}", "{object_adverb}")'
                )
                resp = session.execute(insert_object)
                return resp.is_succeeded(), resp.error_msg() if not resp.is_succeeded() else None
            
            success, result = retry_operation(insert_object_vertex, f"inserting object vertex {obj_id}")
            if not success:
                print(f"Failed to insert object vertex {obj_id} after multiple attempts: {result}")
                # Try with VER_ prefix for the tag
                print(f"Trying with VER_ prefix for object vertex {obj_id}...")
                def insert_object_vertex_with_prefix():
                    insert_object = (
                        f'INSERT VERTEX VER_{obj_label} (name, text, caption, displayName, title, adjective, adverb) VALUES "{obj_id}":("{obj_name}", "{obj_name}", "{obj_name}", "{obj_name}", "{obj_name}", "{object_adjective}", "{object_adverb}")'
                    )
                    resp = session.execute(insert_object)
                    return resp.is_succeeded(), resp.error_msg() if not resp.is_succeeded() else None
                
                success, result = retry_operation(insert_object_vertex_with_prefix, f"inserting object vertex {obj_id} with VER_ prefix")
                if not success:
                    print(f"Failed to insert object vertex {obj_id} with VER_ prefix after multiple attempts: {result}")
            
            # Insert edge
            def insert_edge():
                insert_edge = (
                    f'INSERT EDGE {edge_type} (type, name, caption, original_form, pos_tag, strength, adjective, adverb) VALUES "{sub_id}" -> "{obj_id}":("{rel}", "{rel}", "{rel}", "{original_form}", "{pos_tag}", 1.0, "{relation_adjective}", "{relation_adverb}")'
                )
                resp = session.execute(insert_edge)
                return resp.is_succeeded(), resp.error_msg() if not resp.is_succeeded() else None
            
            success, result = retry_operation(insert_edge, f"inserting edge from {sub_id} to {obj_id}")
            if not success:
                print(f"Failed to insert edge from {sub_id} to {obj_id} after multiple attempts: {result}")
                # Try with REL_ prefix for the edge
                print(f"Trying with REL_ prefix for edge from {sub_id} to {obj_id}...")
                def insert_edge_with_prefix():
                    insert_edge = (
                        f'INSERT EDGE REL_{edge_type} (type, name, caption, original_form, pos_tag, strength, adjective, adverb) VALUES "{sub_id}" -> "{obj_id}":("{rel}", "{rel}", "{rel}", "{original_form}", "{pos_tag}", 1.0, "{relation_adjective}", "{relation_adverb}")'
                    )
                    resp = session.execute(insert_edge)
                    return resp.is_succeeded(), resp.error_msg() if not resp.is_succeeded() else None
                
                success, result = retry_operation(insert_edge_with_prefix, f"inserting edge from {sub_id} to {obj_id} with REL_ prefix")
                if not success:
                    print(f"Failed to insert edge from {sub_id} to {obj_id} with REL_ prefix after multiple attempts: {result}")
        
        print("Successfully uploaded triplets to Nebula Graph.")
        
        # Only release and close if we created the session and connection pool
        if should_close:
            session.release()
            connection_pool.close()
    
    except Exception as e:
        print(f"upload_to_nebula error: {str(e)}")
        # Only release and close if we created the session and connection pool
        if should_close and session is not None:
            session.release()
        if should_close and connection_pool is not None:
            connection_pool.close()
        raise

def upload_to_database(triplets: List[Dict[str, str]], relation_tracking: Dict[str, List[Tuple[str, str]]], session=None, connection_pool=None) -> None:
    """
    Upload triplets to the database(s) specified in the DB_TYPE environment variable.
    
    Args:
        triplets: List of triplets to upload
        relation_tracking: Dictionary tracking relations and their original forms
        session: Optional Nebula session to reuse (if None, a new session will be created)
        connection_pool: Optional connection pool to reuse (if None, a new pool will be created)
    """
    # Load environment variables with override=True to force reload
    load_dotenv(override=True)
    
    # Get the database type from environment variables
    db_type = os.getenv('DB_TYPE', 'neo4j').lower()
    
    # Upload to the specified database(s)
    if db_type == 'neo4j':
        upload_to_neo4j(triplets, relation_tracking)
    elif db_type == 'nebula':
        # Create schema first if needed
        if session is None or connection_pool is None:
            create_nebula_schema()
        upload_to_nebula(triplets, relation_tracking, session, connection_pool)
    elif db_type == 'both':
        upload_to_neo4j(triplets, relation_tracking)
        # Create schema first if needed
        if session is None or connection_pool is None:
            create_nebula_schema()
        upload_to_nebula(triplets, relation_tracking, session, connection_pool)
    else:
        raise ValueError(f"Invalid DB_TYPE: {db_type}. Must be 'neo4j', 'nebula', or 'both'.")

# Example usage when running this module directly:
if __name__ == "__main__":
    sample_text = ("John loves playing football. He also enjoys basketball. "
                   "Mary reads books in the library. She often studies there.")
    
    # Step 1: Resolve coreferences with spaCy and FastCoref
    resolved_text = create_resolved_text_spacy_fastcoref(sample_text)
    print("Resolved text:", resolved_text)
    
    # Step 2: Create triplets with Stanford CoreNLP
    triplets = create_triplets_stanford_corenlp(resolved_text)
    print("Original triplets:", triplets)
    
    # Step 3: Process triplets with lemmatization
    processed_triplets, relation_tracking = process_triplets_with_lemmatization(triplets)
    print("\nProcessed triplets:", processed_triplets)
    
    # Step 4: Upload to database
    upload_to_database(processed_triplets, relation_tracking)
