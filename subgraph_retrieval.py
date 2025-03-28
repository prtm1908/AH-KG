import spacy
from fastcoref import spacy_component
from collections import defaultdict
from typing import Dict, List, Tuple, Set
from neo4j import GraphDatabase
from neo4j.exceptions import ServiceUnavailable
from dotenv import load_dotenv
import os

def process_query(query: str) -> Tuple[List[str], List[str]]:
    """
    Process a query to identify nouns and verbs, including coreference resolution.
    
    Args:
        query: Input query string
        
    Returns:
        Tuple containing lists of nouns and verbs found in the query
    """
    # Load English language model with minimal components
    nlp = spacy.load("en_core_web_sm", exclude=["parser", "lemmatizer", "ner", "textcat"])
    
    # Add FastCoref to the pipeline
    nlp.add_pipe(
        "fastcoref", 
        config={
            'model_architecture': 'LingMessCoref',
            'model_path': 'biu-nlp/lingmess-coref',
            'device': 'cuda:0'
        }
    )
    
    # Process the query with coreference resolution
    doc = nlp(query, component_cfg={"fastcoref": {'resolve_text': True}})
    
    # Get the resolved text
    resolved_text = doc._.resolved_text
    
    # Process the resolved text
    doc = nlp(resolved_text)
    
    # Extract nouns and verbs
    nouns = [token.text for token in doc if token.pos_ == "NOUN"]
    verbs = [token.text for token in doc if token.pos_ == "VERB"]
    
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
            raise ValueError("Missing Neo4j credentials in .env file")
        
        # Create Neo4j driver
        driver = GraphDatabase.driver(uri, auth=(user, password))
        
        subgraph_triplets = []
        
        with driver.session() as session:
            # Create Cypher query to get subgraph with original forms metadata
            cypher_query = """
            MATCH path = (start)-[r*1..$depth]-(connected)
            WHERE start.name IN $nodes AND r.type IN $relations
            UNWIND path AS p
            WITH nodes(p) AS nodes, relationships(p) AS rels
            UNWIND range(0, size(rels)-1) AS i
            RETURN {
                first_node: nodes[i].name,
                relation: rels[i].type,
                second_node: nodes[i+1].name,
                original_forms: rels[i].original_forms
            } AS triplet
            """
            
            result = session.run(cypher_query, nodes=nodes, relations=relations, depth=depth)
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
        
        # Get original forms from metadata and use the first original form
        original_forms = triplet.get('original_forms', [])
        if original_forms:
            new_triplet['relation'] = original_forms[0]['original_form']
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
    
    # 2. Lemmatize relations
    relation_tracking = lemmatize_relations(verbs)
    
    # 3. Get all possible relation forms (original and lemmatized)
    all_relations = []
    for lemmatized, originals in relation_tracking.items():
        all_relations.extend([lemmatized] + originals)
    
    # 4. Get subgraph from Neo4j
    subgraph = get_subgraph_from_neo4j(nouns, all_relations)
    
    # 5. Re-inflect relations to match original verbs from query
    reinflected_subgraph = reinflect_relations(subgraph, verbs)
    
    return reinflected_subgraph

# Example usage
if __name__ == "__main__":
    sample_query = "John loves playing football. He also enjoys basketball."
    subgraph = process_query_and_get_subgraph(sample_query)
    print("Relevant subgraph with re-inflected relations:", subgraph)
