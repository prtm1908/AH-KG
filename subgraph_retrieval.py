import spacy
from fastcoref import spacy_component
from collections import defaultdict
from typing import Dict, List, Tuple, Set
from neo4j import GraphDatabase
from neo4j.exceptions import ServiceUnavailable
from dotenv import load_dotenv
import os
import torch
from nebula3.gclient.net import ConnectionPool
from nebula3.Config import Config

def process_query(query: str) -> Tuple[List[str], List[str]]:
    """
    Process a query to identify nouns and verbs, including coreference resolution.
    
    Args:
        query: Input query string
        
    Returns:
        Tuple containing lists of nouns and verbs found in the query
    """
    print("\nStarting process_query")
    # Load English language model with minimal components
    print("Loading spaCy model...")
    nlp = spacy.load("en_core_web_sm", exclude=["parser", "lemmatizer", "ner", "textcat"])
    
    # Try with CUDA first
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
        nlp.get_pipe("fastcoref").config['device'] = 'cpu'
        
        # Process the query with coreference resolution using CPU
        print("Processing query with CPU...")
        doc = nlp(query, component_cfg={"fastcoref": {'resolve_text': True}})
        print("Successfully processed query with CPU")
    
    # Get the resolved text
    print("Getting resolved text...")
    resolved_text = doc._.resolved_text
    
    # Process the resolved text
    print("Processing resolved text...")
    doc = nlp(resolved_text)
    
    # Extract nouns and verbs
    nouns = [token.text for token in doc if token.pos_ == "NOUN"]
    verbs = [token.text for token in doc if token.pos_ == "VERB"]
    
    print(f"Found {len(nouns)} nouns and {len(verbs)} verbs")
    print(f"Nouns: {nouns}")
    return nouns, verbs

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
        doc = nlp(verb)
        if len(doc) > 0:
            lemmatized = doc[0].lemma_
            relation_tracking[lemmatized].append(verb)
    
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
                RETURN {{
                    first_node: nodes[i].name,
                    relation: rels[i].type,
                    second_node: nodes[i+1].name,
                    original_form: rels[i].original_form,
                    pos_tag: rels[i].pos_tag
                }} AS triplet
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
                    RETURN {{
                        first_node: nodes[i].name,
                        relation: rels[i].type,
                        second_node: nodes[i+1].name,
                        original_form: rels[i].original_form,
                        pos_tag: rels[i].pos_tag
                    }} AS triplet
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
                RETURN {{
                    first_node: nodes[i].name,
                    relation: rels[i].type,
                    second_node: nodes[i+1].name,
                    original_form: rels[i].original_form,
                    pos_tag: rels[i].pos_tag
                }} AS triplet
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
            'first_node': triplet['first_node'],
            'second_node': triplet['second_node']
        }
        
        # Use the original_form property if available
        if 'original_form' in triplet:
            new_triplet['relation'] = triplet['original_form']
        else:
            new_triplet['relation'] = triplet['relation']
        
        reinflected_triplets.append(new_triplet)
    
    return reinflected_triplets

def process_query_and_get_subgraph(query: str) -> List[Dict[str, str]]:
    """
    Main function to process a query and get relevant subgraph.
    
    Args:
        query: Input query string
        
    Returns:
        List of triplets representing the relevant subgraph with re-inflected relations
    """
    # 1. Process query to identify nouns and verbs (including coreference resolution)
    nouns, verbs = process_query(query)
    
    print(f"\nQuery analysis:")
    print(f"Found nouns: {nouns}")
    print(f"Found verbs: {verbs}")
    
    # 2. Lemmatize relations
    relation_tracking = lemmatize_relations(verbs)
    
    # 3. Get all possible relation forms (original and lemmatized)
    all_relations = []
    for lemmatized, originals in relation_tracking.items():
        all_relations.extend([lemmatized] + originals)
    
    print(f"All relations to search for: {all_relations}")
    
    # 4. Get subgraph from the specified database(s)
    if not nouns and all_relations:
        print("No nouns found but verbs found - will search for all nodes connected by these verbs")
    elif nouns and not all_relations:
        print("Nouns found but no verbs - will search for all relationships containing these nouns")
    elif nouns and all_relations:
        print("Both nouns and verbs found - will search for specific relationships between these nouns")
    else:
        print("No nouns or verbs found - will return empty result")
        return []
    
    subgraph = get_subgraph_from_database(nouns, all_relations)
    print(f"Found {len(subgraph)} triplets in subgraph")
    
    # 5. Re-inflect relations to match original verbs from query
    reinflected_subgraph = reinflect_relations(subgraph, verbs)
    
    return reinflected_subgraph

def get_subgraph_from_nebula(nodes: List[str], relations: List[str], depth: int = 2) -> List[Dict[str, str]]:
    """
    Extract a subgraph from Nebula Graph based on given nodes and relations.
    
    Args:
        nodes: List of nodes to start from
        relations: List of relations to consider
        depth: Depth of traversal (default: 2)
        
    Returns:
        List of triplets representing the subgraph
    """
    # Load environment variables with override=True to force reload
    load_dotenv(override=True)
    
    # Get Nebula Graph credentials from environment variables
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
    
    print(f"Attempting to connect to Nebula Graph at {host}:{port}")
    
    try:
        # Create Nebula Graph connection pool
        config = Config()
        connection_pool = ConnectionPool()
        
        # Initialize the connection pool
        init_result = connection_pool.init([(host, port)], config)
        if not init_result:
            raise ConnectionError(f"Failed to initialize connection pool to Nebula Graph at {host}:{port}")
        
        # Get a session from the pool
        session = connection_pool.get_session(user, password)
        
        # Use the space
        resp = session.execute(f"USE {space}")
        if not resp.is_succeeded():
            raise Exception(f"Failed to use space {space}: {resp.error_msg()}")
        
        subgraph_triplets = []
        
        # Create nGQL query to get subgraph with original forms metadata
        if relations:  # If we have verbs/relations to match
            # First find all nodes connected by our verbs
            relations_str = ", ".join([f"'{rel}'" for rel in relations])
            verb_nodes_query = f"""
            MATCH p=(v:entity)-[e:relation*1..{depth}]->(v2:entity)
            WHERE ALL(rel IN e WHERE rel.type IN [{relations_str}])
            RETURN DISTINCT v.name as node_name
            UNION
            MATCH p=(v:entity)-[e:relation*1..{depth}]->(v2:entity)
            WHERE ALL(rel IN e WHERE rel.type IN [{relations_str}])
            RETURN DISTINCT v2.name as node_name
            """
            
            # Get all nodes connected by our verbs
            resp = session.execute(verb_nodes_query)
            if not resp.is_succeeded():
                print(f"Error executing verb nodes query: {resp.error_msg()}")
                session.release()
                connection_pool.close()
                return []
            
            verb_connected_nodes = []
            for row in resp.rows():
                verb_connected_nodes.append(row[0])
            
            print(f"Found {len(verb_connected_nodes)} nodes connected by verbs: {verb_connected_nodes}")
            
            # Get all triplets containing our verbs
            verb_triplets_query = f"""
            MATCH p=(v:entity)-[e:relation*1..{depth}]->(v2:entity)
            WHERE ALL(rel IN e WHERE rel.type IN [{relations_str}])
            UNWIND p AS path
            WITH nodes(path) AS nodes, relationships(path) AS rels
            UNWIND range(0, size(rels)-1) AS i
            RETURN {{
                first_node: nodes[i].name,
                relation: rels[i].type,
                second_node: nodes[i+1].name,
                original_form: rels[i].original_form,
                pos_tag: rels[i].pos_tag
            }} AS triplet
            """
            
            resp = session.execute(verb_triplets_query)
            if not resp.is_succeeded():
                print(f"Error executing verb triplets query: {resp.error_msg()}")
                session.release()
                connection_pool.close()
                return []
            
            verb_triplets = []
            for row in resp.rows():
                verb_triplets.append(eval(row[0]))
            
            print(f"Found {len(verb_triplets)} triplets with verbs")
            
            # If we also have nouns, find triplets containing those nouns
            if nodes:
                # Find paths containing our nouns
                nodes_str = ", ".join([f"'{node}'" for node in nodes])
                noun_query = f"""
                MATCH p=(v:entity)-[e:relation*1..{depth}]->(v2:entity)
                WHERE v.name IN [{nodes_str}] OR v2.name IN [{nodes_str}]
                UNWIND p AS path
                WITH nodes(path) AS nodes, relationships(path) AS rels
                UNWIND range(0, size(rels)-1) AS i
                RETURN {{
                    first_node: nodes[i].name,
                    relation: rels[i].type,
                    second_node: nodes[i+1].name,
                    original_form: rels[i].original_form,
                    pos_tag: rels[i].pos_tag
                }} AS triplet
                """
                
                resp = session.execute(noun_query)
                if not resp.is_succeeded():
                    print(f"Error executing noun query: {resp.error_msg()}")
                    session.release()
                    connection_pool.close()
                    return []
                
                noun_triplets = []
                for row in resp.rows():
                    noun_triplets.append(eval(row[0]))
                
                print(f"Found {len(noun_triplets)} triplets with nouns")
                
                # Combine results and remove duplicates
                all_triplets = {str(triplet) for triplet in verb_triplets + noun_triplets}
                subgraph_triplets = [eval(triplet) for triplet in all_triplets]
            else:
                subgraph_triplets = verb_triplets
        else:  # If no verbs found, get all relationships containing our nodes
            nodes_str = ", ".join([f"'{node}'" for node in nodes])
            nGQL_query = f"""
            MATCH p=(v:entity)-[e:relation*1..{depth}]->(v2:entity)
            WHERE v.name IN [{nodes_str}] OR v2.name IN [{nodes_str}]
            UNWIND p AS path
            WITH nodes(path) AS nodes, relationships(path) AS rels
            UNWIND range(0, size(rels)-1) AS i
            RETURN {{
                first_node: nodes[i].name,
                relation: rels[i].type,
                second_node: nodes[i+1].name,
                original_form: rels[i].original_form,
                pos_tag: rels[i].pos_tag
            }} AS triplet
            """
            
            resp = session.execute(nGQL_query)
            if not resp.is_succeeded():
                print(f"Error executing query: {resp.error_msg()}")
                session.release()
                connection_pool.close()
                return []
            
            subgraph_triplets = []
            for row in resp.rows():
                subgraph_triplets.append(eval(row[0]))
        
        # Release the session back to the pool
        session.release()
        
        # Close the connection pool
        connection_pool.close()
        
        return subgraph_triplets
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        return []

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

# Example usage
if __name__ == "__main__":
    sample_query = "John loves playing football. He also enjoys basketball."
    subgraph = process_query_and_get_subgraph(sample_query)
    print("Relevant subgraph with re-inflected relations:", subgraph)
