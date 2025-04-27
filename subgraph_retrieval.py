import spacy
from fastcoref import spacy_component
from collections import defaultdict
from typing import Dict, List, Tuple, Set, Any
from neo4j import GraphDatabase
from neo4j.exceptions import ServiceUnavailable
from dotenv import load_dotenv
import os
import torch
from nebula3.gclient.net import ConnectionPool
from nebula3.Config import Config
import requests
import json
import re

def process_query(query: str) -> Tuple[List[str], List[str]]:
    """
    Process a query to identify nouns and verbs, including coreference resolution.
    First resolves coreferences using spaCy and FastCoref, then extracts nouns and verbs using Stanford CoreNLP.
    
    Args:
        query: Input query string
        
    Returns:
        Tuple containing lists of nouns and verbs found in the query
    """
    print("\nStarting process_query")
    
    # First resolve coreferences using spaCy and FastCoref
    print("Loading spaCy model for coreference resolution...")
    nlp = spacy.load("en_core_web_sm", exclude=["parser", "lemmatizer", "ner", "textcat"])
    
    # Try with CUDA first for FastCoref
    try:
        print("Checking CUDA availability...")
        if torch.cuda.is_available() and torch.cuda.is_initialized():
            device = 'cuda:0'
            print("CUDA is available, using GPU")
        else:
            device = 'cpu'
            print("CUDA is not available, using CPU")
            
        # Add FastCoref to the pipeline
        print("Adding FastCoref to pipeline...")
        nlp.add_pipe(
            "fastcoref", 
            config={
                'model_architecture': 'LingMessCoref',
                'model_path': 'biu-nlp/lingmess-coref',
                'device': device
            }
        )
        
        # Process the query with coreference resolution
        print("Processing query with coreference resolution...")
        doc = nlp(query, component_cfg={"fastcoref": {'resolve_text': True}})
        print("Successfully processed query with coreference resolution")
        
    except Exception as e:
        print(f"Error during CUDA processing: {str(e)}")
        # If any error occurs (including CUDA errors), retry with CPU
        print("Retrying with CPU...")
        
        # Update FastCoref config to use CPU
        if "fastcoref" in nlp.pipe_names:
            nlp.remove_pipe("fastcoref")
        
        nlp.add_pipe(
            "fastcoref", 
            config={
                'model_architecture': 'LingMessCoref',
                'model_path': 'biu-nlp/lingmess-coref',
                'device': 'cpu'
            }
        )
        
        # Process the query with coreference resolution using CPU
        print("Processing query with CPU...")
        doc = nlp(query, component_cfg={"fastcoref": {'resolve_text': True}})
        print("Successfully processed query with CPU")
    
    # Get the resolved text
    print("Getting resolved text...")
    resolved_text = doc._.resolved_text
    print(f"Resolved text: {resolved_text}")
    
    # Now process with Stanford CoreNLP to extract nouns and verbs
    print("Sending resolved text to Stanford CoreNLP...")
    
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
            data=resolved_text.encode('utf-8'),
            headers={'Content-Type': 'application/x-www-form-urlencoded; charset=UTF-8'}
        )
        
        if response.status_code != 200:
            print(f"Error from Stanford CoreNLP: {response.status_code}")
            print(f"Response: {response.text}")
            return [], []
            
        # Parse the response
        result = response.json()
        
        nouns = set()
        verbs = set()
        
        # Process each sentence
        for sentence in result.get('sentences', []):
            tokens = sentence.get('tokens', [])
            
            # Extract nouns and verbs based on POS tags
            for token in tokens:
                pos = token.get('pos', '')
                word = token.get('word', '')
                
                # Stanford CoreNLP POS tags:
                # NN, NNS, NNP, NNPS for nouns
                # VB, VBD, VBG, VBN, VBP, VBZ for verbs
                if pos.startswith('NN'):
                    nouns.add(word.lower())
                elif pos.startswith('VB'):
                    verbs.add(word.lower())
        
        nouns_list = list(nouns)
        verbs_list = list(verbs)
        
        print(f"Found {len(nouns_list)} nouns and {len(verbs_list)} verbs")
        print(f"Nouns: {nouns_list}")
        print(f"Verbs: {verbs_list}")
        return nouns_list, verbs_list
        
    except Exception as e:
        print(f"Error processing with Stanford CoreNLP: {str(e)}")
        return [], []

def lemmatize_relations(verbs: List[str]) -> Dict[str, List[str]]:
    """
    Lemmatize verbs and track original forms.
    
    Args:
        verbs: List of verbs to lemmatize
        
    Returns:
        Dictionary mapping lemmatized forms to lists of original forms
    """
    # Load English language model with lemmatizer
    nlp = spacy.load("en_core_web_sm", exclude=["parser", "ner", "textcat"])
    
    relation_tracking = defaultdict(list)
    
    for verb in verbs:
        doc = nlp(verb.lower())
        if len(doc) > 0:
            lemmatized = doc[0].lemma_
            relation_tracking[lemmatized].append(verb.lower())
    
    return dict(relation_tracking)

def get_subgraph_from_neo4j(nodes: List[str], relations: List[str], depth: int = 2) -> List[Dict[str, str]]:
    """
    Extract a subgraph from Neo4j based on given nodes and relations.
    
    Args:
        nodes: List of nodes to start from
        relations: List of relations to consider
        depth: Depth of traversal (default: 2)
        
    Returns:
        List of triplets representing the subgraph, including original forms metadata
    """
    try:
        # Load environment variables
        load_dotenv()
        
        # Get Neo4j credentials
        uri = os.getenv('NEO4J_URI')
        user = os.getenv('NEO4J_USER')
        password = os.getenv('NEO4J_PASSWORD')
        
        if not all([uri, user, password]):
            print("Error: Missing Neo4j credentials in .env file")
            print(f"URI: {'Present' if uri else 'Missing'}")
            print(f"User: {'Present' if user else 'Missing'}")
            print(f"Password: {'Present' if password else 'Missing'}")
            raise ValueError("Missing Neo4j credentials in .env file")
        
        print(f"Attempting to connect to Neo4j at {uri}")
        # Create Neo4j driver
        driver = GraphDatabase.driver(uri, auth=(user, password))
        
        # Verify connection
        try:
            driver.verify_connectivity()
            print("Successfully connected to Neo4j database")
        except Exception as e:
            print(f"Failed to verify Neo4j connectivity: {str(e)}")
            raise
        
        subgraph_triplets = []
        
        with driver.session() as session:
            # Create Cypher query to get subgraph with original forms metadata
            if relations:  # If we have verbs/relations to match
                # First find all nodes connected by our verbs
                verb_nodes_query = f"""
                MATCH path = (start)-[r*1..{depth}]->(connected)
                WHERE ALL(rel IN r WHERE rel.type IN $relations)
                RETURN DISTINCT start.name as node_name
                UNION
                MATCH path = (start)-[r*1..{depth}]->(connected)
                WHERE ALL(rel IN r WHERE rel.type IN $relations)
                RETURN DISTINCT connected.name as node_name
                """
                
                # Get all nodes connected by our verbs
                verb_nodes_result = session.run(verb_nodes_query, relations=relations)
                verb_connected_nodes = [record["node_name"] for record in verb_nodes_result]
                print(f"Found {len(verb_connected_nodes)} nodes connected by verbs: {verb_connected_nodes}")
                
                # Get all triplets containing our verbs
                verb_triplets_query = f"""
                MATCH path = (start)-[r*1..{depth}]->(connected)
                WHERE ALL(rel IN r WHERE rel.type IN $relations)
                UNWIND path AS p
                WITH nodes(p) AS nodes, relationships(p) AS rels
                UNWIND range(0, size(rels)-1) AS i
                RETURN {
                    first_node: nodes[i].name,
                    relation: rels[i].type,
                    second_node: nodes[i+1].name,
                    relation_original_form: rels[i].original_form
                } AS triplet
                """
                
                verb_results = session.run(verb_triplets_query, relations=relations)
                verb_triplets = [record["triplet"] for record in verb_results]
                print(f"Found {len(verb_triplets)} triplets with verbs")
                
                # If we also have nouns, find triplets containing those nouns
                if nodes:
                    # Find paths containing our nouns
                    noun_query = f"""
                    MATCH path = (start)-[r*1..{depth}]->(connected)
                    WHERE (start.name IN $nodes OR connected.name IN $nodes)
                    UNWIND path AS p
                    WITH nodes(p) AS nodes, relationships(p) AS rels
                    UNWIND range(0, size(rels)-1) AS i
                    RETURN {
                        first_node: nodes[i].name,
                        relation: rels[i].type,
                        second_node: nodes[i+1].name,
                        relation_original_form: rels[i].original_form
                    } AS triplet
                    """
                    
                    noun_results = session.run(noun_query, nodes=nodes)
                    noun_triplets = [record["triplet"] for record in noun_results]
                    print(f"Found {len(noun_triplets)} triplets with nouns")
                    
                    # Combine results and remove duplicates
                    all_triplets = {str(triplet) for triplet in verb_triplets + noun_triplets}
                    subgraph_triplets = [eval(triplet) for triplet in all_triplets]
                else:
                    subgraph_triplets = verb_triplets
            else:  # If no verbs found, get all relationships containing our nodes
                cypher_query = f"""
                MATCH path = (start)-[r*1..{depth}]->(connected)
                WHERE (start.name IN $nodes OR connected.name IN $nodes)
                UNWIND path AS p
                WITH nodes(p) AS nodes, relationships(p) AS rels
                UNWIND range(0, size(rels)-1) AS i
                RETURN {
                    first_node: nodes[i].name,
                    relation: rels[i].type,
                    second_node: nodes[i+1].name,
                    relation_original_form: rels[i].original_form
                } AS triplet
                """
                result = session.run(cypher_query, nodes=nodes)
                subgraph_triplets = [record["triplet"] for record in result]
            
        driver.close()
        return subgraph_triplets
        
    except ServiceUnavailable:
        print("Could not connect to Neo4j database")
        return []
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        return []

def reinflect_relations(subgraph: List[Dict[str, str]], target_verbs: List[str]) -> List[Dict[str, str]]:
    """
    Re-inflect relations in the subgraph using the original forms from metadata.
    
    Args:
        subgraph: List of triplets with original forms metadata
        target_verbs: List of target verbs to match against (not used anymore)
        
    Returns:
        List of triplets with re-inflected relations
    """
    reinflected_triplets = []
    
    for triplet in subgraph:
        # Create a new triplet with the same nodes
        new_triplet = {
            'first_node': triplet['first_node'].lower(),
            'second_node': triplet['second_node'].lower()
        }
        
        # Use the relation_original_form property if available
        if 'relation_original_form' in triplet:
            new_triplet['relation'] = triplet['relation_original_form'].lower()
        else:
            new_triplet['relation'] = triplet['relation'].lower()
        
        reinflected_triplets.append(new_triplet)
    
    return reinflected_triplets

def process_query_and_get_subgraph(query: str, session=None) -> List[Dict[str, str]]:
    """
    Process a query to identify nouns and verbs, and retrieve a relevant subgraph.
    
    Args:
        query: Input query string
        session: Optional Nebula Graph session to use. If None, a new connection will be established.
        
    Returns:
        List of triplets representing the relevant subgraph
    """
    print("\nStarting process_query_and_get_subgraph")
    
    # Process the query to identify nouns and verbs
    nouns, verbs = process_query(query)
    print(f"Identified nouns: {nouns}")
    print(f"Identified verbs: {verbs}")
    
    # Get the database type from environment variables
    load_dotenv(override=True)
    db_type = os.getenv('DB_TYPE', 'neo4j').lower()
    
    # Retrieve the subgraph based on the database type
    if db_type == 'neo4j':
        return get_subgraph_from_neo4j(nouns, verbs)
    elif db_type == 'nebula':
        return get_subgraph_from_nebula(nouns, verbs, session=session)
    elif db_type == 'both':
        # For 'both', we'll return the Neo4j results as they're more detailed
        return get_subgraph_from_neo4j(nouns, verbs)
    else:
        raise ValueError(f"Invalid DB_TYPE: {db_type}. Must be 'neo4j', 'nebula', or 'both'.")

def get_subgraph_from_nebula(nodes: List[str], relations: List[str], depth: int = 2, session=None) -> List[Dict[str, str]]:
    """
    Extract a subgraph from Nebula Graph up to `depth` hops from the input `nodes`,
    correctly determining edge direction at each step.
    Drop-in replacement for existing function.

    Args:
        nodes: list of starting node names
        relations: (unused) list of relation types
        depth: number of hops to traverse (default 2)
        session: optional Nebula session

    Returns:
        List of triplets: {first_node, relation, second_node, relation_original_form}
    """
    # Load and verify credentials
    load_dotenv(override=True)
    host = os.getenv('NEBULA_HOST')
    port = int(os.getenv('NEBULA_PORT', '9669'))
    user = os.getenv('NEBULA_USER')
    password = os.getenv('NEBULA_PASSWORD')
    space = os.getenv('NEBULA_SPACE')
    if not all([host, user, password, space]):
        raise ValueError("Missing Nebula Graph credentials in .env file")

    should_close = False
    connection_pool = None
    subgraph_triplets: List[Dict[str, str]] = []

    try:
        # Establish connection if not provided
        if session is None:
            should_close = True
            config = Config()
            connection_pool = ConnectionPool()
            if not connection_pool.init([(host, port)], config):
                raise ConnectionError(f"Failed to init Nebula Graph at {host}:{port}")
            session = connection_pool.get_session(user, password)
            resp = session.execute(f"USE {space}")
            if not resp.is_succeeded():
                raise Exception(f"Failed to switch to space {space}: {resp.error_msg()}")

        # Initialize BFS
        visited: Set[str] = set()
        frontier: List[Tuple[str, str]] = []  # (vertex_id, vertex_name)

        # Lookup IDs for each starting noun
        for noun in nodes:
            lookup_q = f'LOOKUP ON NOUN WHERE NOUN.name == "{noun}" YIELD id(vertex)'
            resp = session.execute(lookup_q)
            if not resp.is_succeeded():
                continue
            for row in resp.rows():
                vid = _extract_id(row.values[0])
                if vid not in visited:
                    visited.add(vid)
                    frontier.append((vid, noun))

        # Perform BFS up to `depth` levels
        for _ in range(depth):
            next_frontier: List[Tuple[str, str]] = []
            for vid, vname in frontier:
                go_q = (
                    f'GO FROM "{vid}" OVER * BIDIRECT '
                    'YIELD src(edge) AS source, dst(edge) AS target, '
                    'properties(edge) AS edge_props, properties($$) AS neighbor_props'
                )
                resp = session.execute(go_q)
                if not resp.is_succeeded():
                    continue

                for row in resp.rows():
                    src_id = _extract_id(row.values[0])
                    dst_id = _extract_id(row.values[1])
                    edge_props = _convert_nebula_props(row.values[2])
                    nb_props   = _convert_nebula_props(row.values[3])

                    relation_type = edge_props.get('type')
                    orig_form     = edge_props.get('original_form', '')
                    nb_name       = nb_props.get('name')

                    if not (relation_type and nb_name):
                        continue

                    # Determine neighbor ID and extract its name
                    if src_id == vid:
                        first_node  = vname
                        second_node = nb_name
                        neighbor_id = dst_id
                    else:
                        first_node  = nb_name
                        second_node = vname
                        neighbor_id = src_id

                    subgraph_triplets.append({
                        'first_node': first_node.lower(),
                        'relation': relation_type.lower(),
                        'second_node': second_node.lower(),
                        'relation_original_form': orig_form.lower()
                    })

                    # Queue neighbor for next level if unseen
                    if neighbor_id not in visited:
                        visited.add(neighbor_id)
                        next_frontier.append((neighbor_id, nb_name))

            frontier = next_frontier

        # Cleanup
        if should_close and connection_pool:
            session.release()
            connection_pool.close()

        # Deduplicate
        unique, seen = [], set()
        for t in subgraph_triplets:
            key = (t['first_node'], t['relation'], t['second_node'])
            if key not in seen:
                seen.add(key)
                unique.append(t)
        return unique

    except Exception:
        if should_close and connection_pool:
            session.release()
            connection_pool.close()
        raise

def get_subgraph_from_database(nodes: List[str], relations: List[str], depth: int = 2) -> List[Dict[str, str]]:
    """
    Extract a subgraph from the specified graph database(s) based on given nodes and relations.
    
    Args:
        nodes: List of nodes to start from
        relations: List of relations to consider
        depth: Depth of traversal (default: 2)
        
    Returns:
        List of triplets representing the subgraph, including original forms metadata
    """
    # Load environment variables with override=True to force reload
    load_dotenv(override=True)
    
    # Get the database type from environment variables
    db_type = os.getenv('DB_TYPE', 'neo4j').lower()
    
    # Extract subgraph from the specified database(s)
    if db_type == 'neo4j':
        return get_subgraph_from_neo4j(nodes, relations, depth)
    elif db_type == 'nebula':
        return get_subgraph_from_nebula(nodes, relations, depth)
    elif db_type == 'both':
        # For 'both', we'll use Neo4j as the primary source
        return get_subgraph_from_neo4j(nodes, relations, depth)
    else:
        raise ValueError(f"Invalid DB_TYPE: {db_type}. Must be 'neo4j', 'nebula', or 'both'.")

def _parse_nebula_map(value) -> Dict[str, str]:
    """
    Parse a Nebula Graph Value object containing an NMap (kvs) into a Python dict.
    It flattens the string representation and uses regex to extract keys and values.
    Returns a dict mapping property names to their values (as strings or numbers).
    """
    text = re.sub(r"\s+", " ", str(value))
    result = {}
    # Regex to match string values or float values in the NMap representation
    pattern = r"b'(?P<key>[^']+)': Value\(.*?(?:sVal=b'(?P<sval>[^']*)'|fVal=(?P<fval>[-0-9.]+)).*?\)"
    for m in re.finditer(pattern, text):
        key = m.group('key')
        if m.group('sval') is not None:
            val = m.group('sval')
        else:
            val = m.group('fval')
        result[key] = val
    return result

def _extract_id(val: Any) -> str:
    text = str(val)
    if "sVal=b'" in text:
        return text.split("sVal=b'")[1].split("'")[0]
    return text

def _convert_nebula_props(val: Any) -> Dict[str, Any]:
    if hasattr(val, 'as_map'):
        m = val.as_map()
        out = {}
        for k, v in m.items():
            key = k.decode('utf-8') if isinstance(k, (bytes, bytearray)) else k
            raw = getattr(v, 'sVal', None) or getattr(v, 'iVal', None) or getattr(v, 'fVal', None) or getattr(v, 'bVal', None)
            if isinstance(raw, (bytes, bytearray)):
                raw = raw.decode('utf-8')
            out[key] = raw
        return out
    return _parse_nebula_map(val)

# Example usage
if __name__ == "__main__":
    sample_query = "John loves playing football. He also enjoys basketball."
    subgraph = process_query_and_get_subgraph(sample_query)
    print("Relevant subgraph with re-inflected relations:", subgraph)
