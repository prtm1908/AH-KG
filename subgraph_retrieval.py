import spacy
from fastcoref import spacy_component
from collections import defaultdict
from typing import Dict, List, Tuple, Set
from neo4j import GraphDatabase
from neo4j.exceptions import ServiceUnavailable
from dotenv import load_dotenv
import os
import torch

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
    
    # 4. Get subgraph from Neo4j
    if not nouns and all_relations:
        print("No nouns found but verbs found - will search for all nodes connected by these verbs")
    elif nouns and not all_relations:
        print("Nouns found but no verbs - will search for all relationships containing these nouns")
    elif nouns and all_relations:
        print("Both nouns and verbs found - will search for specific relationships between these nouns")
    else:
        print("No nouns or verbs found - will return empty result")
        return []
    
    subgraph = get_subgraph_from_neo4j(nouns, all_relations)
    print(f"Found {len(subgraph)} triplets in subgraph")
    
    # 5. Re-inflect relations to match original verbs from query
    reinflected_subgraph = reinflect_relations(subgraph, verbs)
    
    return reinflected_subgraph

# Example usage
if __name__ == "__main__":
    sample_query = "John loves playing football. He also enjoys basketball."
    subgraph = process_query_and_get_subgraph(sample_query)
    print("Relevant subgraph with re-inflected relations:", subgraph)
