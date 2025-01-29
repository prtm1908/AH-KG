import spacy
from typing import List, Dict, Tuple, Set
from collections import defaultdict
from rapidfuzz import fuzz

def fuzzy_match_entity(text: str, entities: set, threshold: float = 0.9) -> str:
    """
    Find the best matching entity for a given text using fuzzy matching.
    Returns the original text if no match is found above the threshold.
    
    Args:
        text: The text to match
        entities: Set of entity texts to match against
        threshold: Minimum similarity ratio required for a match (0-1)
    
    Returns:
        The best matching entity text or the original text if no match found
    """
    best_match = None
    best_ratio = 0
    text_lower = text.lower()
    
    for entity in entities:
        entity_lower = entity.lower()
        
        # Try exact word matching within multi-word entities first
        if text_lower in entity_lower.split() or entity_lower in text_lower.split():
            if len(text) > 3:  # Avoid matching very short words
                return entity
        
        # Try partial string matching
        if text_lower in entity_lower or entity_lower in text_lower:
            if len(text) > 3:  # Avoid matching very short words
                return entity
        
        # Calculate token-sort ratio to handle word order differences
        ratio = fuzz.token_sort_ratio(text_lower, entity_lower) / 100.0
        
        # If that's not high enough, try partial ratio for substring matches
        if ratio < threshold:
            ratio = fuzz.partial_ratio(text_lower, entity_lower) / 100.0
            
        if ratio > threshold and ratio > best_ratio:
            best_match = entity
            best_ratio = ratio
    
    return best_match if best_match else text

def extract_relations_from_texts(texts: List[str]) -> Dict[str, List[Dict]]:
    """
    Process multiple texts and extract relations from each one.
    Maintains a global list of entities across all texts.
    
    Args:
        texts: List of text strings to process
        
    Returns:
        Dictionary containing all entities found and all relations extracted
    """
    nlp = spacy.load("en_core_web_lg")
    
    # Initialize global results
    all_entities = []
    all_relations = []
    all_unique_entities = set()
    relation_strings = []  # New list to store relation strings
    
    # Process each text
    for text_idx, text in enumerate(texts):
        doc = nlp(text)
        entity_positions = []
        
        # First pass: collect entities
        for ent in doc.ents:
            if ent.text not in all_unique_entities:
                entity_dict = {
                    'text': ent.text,
                    'type': ent.label_,
                    'text_index': text_idx
                }
                all_entities.append(entity_dict)
                all_unique_entities.add(ent.text)
            
            if ent.label_ == "PERSON":
                entity_positions.append((ent.text, ent.start_char, ent.end_char))
        
        # Sort entity positions by their occurrence in this text
        entity_positions.sort(key=lambda x: x[1])
        
        # Process relations in this text
        text_relations = _process_text_relations(
            doc, 
            entity_positions, 
            all_unique_entities
        )
        
        # Add text index to relations and collect them
        for relation in text_relations:
            relation['text_index'] = text_idx
            all_relations.append(relation)
            
            # Add the relation string to our list
            relation_str = f"{relation['subject']} -> {relation['relation']} -> {relation['object']}"
            relation_strings.append(relation_str)
    
    return {
        'entities': all_entities,
        'relations': all_relations,
        'relation_strings': relation_strings  # New field for relation strings
    }

def _process_text_relations(doc, entity_positions: List[Tuple], unique_entities: Set[str]) -> List[Dict]:
    """
    Process relations for a single text document.
    
    Args:
        doc: SpaCy document
        entity_positions: List of (entity, start, end) tuples for PERSON entities
        unique_entities: Set of all known entity texts
        
    Returns:
        List of relations found in this text
    """
    relations = []
    pronouns = {"them", "it", "they", "who", "he", "she", "him", "his", "her", "their"}
    
    def find_last_person(pronoun_pos):
        previous_entities = [ent for ent in entity_positions if ent[1] < pronoun_pos]
        return previous_entities[-1][0] if previous_entities else None
    
    verbs = [token for token in doc if token.pos_ == "VERB"]
    
    for verb in verbs:
        # Process subjects
        subjects = _find_subjects(verb)
        resolved_subjects = []
        
        for subj in subjects:
            subj_text = subj.text.lower()
            if subj_text in pronouns:
                last_person = find_last_person(subj.idx)
                resolved_subjects.append(last_person if last_person else subj.text)
            else:
                if subj.text not in unique_entities:
                    matched_text = fuzzy_match_entity(subj.text, unique_entities)
                    resolved_subjects.append(matched_text)
                else:
                    resolved_subjects.append(subj.text)
        
        if not resolved_subjects:
            continue
            
        # Process objects
        objects = _find_objects(verb)
        resolved_objects = []
        
        for obj in objects:
            obj_text = obj.text.lower()
            if obj_text in pronouns:
                last_person = find_last_person(obj.idx)
                resolved_objects.append(last_person if last_person else obj.text)
            else:
                if obj.text not in unique_entities:
                    matched_text = fuzzy_match_entity(obj.text, unique_entities)
                    resolved_objects.append(matched_text)
                else:
                    resolved_objects.append(obj.text)
        
        if not resolved_objects:
            continue
            
        # Create relations
        for subj in resolved_subjects:
            for obj in resolved_objects:
                if subj == obj:
                    continue
                    
                relation = verb.lemma_
                if any(child.dep_ == "neg" for child in verb.children):
                    relation = f"not_{relation}"
                    
                sent = next(sent for sent in doc.sents if verb in sent)
                
                relations.append({
                    'subject': subj,
                    'object': obj,
                    'relation': relation,
                    'sentence': sent.text,
                    'global_position': verb.i
                })
    
    return sorted(relations, key=lambda x: x['global_position'])

def _find_subjects(verb) -> List:
    """Find all subjects related to a verb."""
    subjects = []
    
    for token in verb.lefts:
        if token.dep_ in ("nsubj", "nsubjpass"):
            subjects.append(token)
            subjects.extend([child for child in token.children 
                           if child.dep_ == "conj"])
    
    return subjects

def _find_objects(verb) -> List:
    """Find all objects related to a verb."""
    objects = []
    
    for token in verb.rights:
        if token.dep_ == "dobj":
            objects.append(token)
            objects.extend([child for child in token.children 
                          if child.dep_ == "conj"])
        elif token.dep_ == "prep":
            objects.extend([child for child in token.children 
                          if child.dep_ == "pobj"])
    
    return objects

def find_relations():
    test_texts = [
    """
Bilbo Baggins celebrates his birthday and leaves the Ring to Frodo, his heir. 
Gandalf (a wizard) suspects it is a Ring of Power; seventeen years later, he confirms it was lost by the 
Dark Lord Sauron and counsels Frodo to take it away from the Shire. Gandalf leaves, promising to return, 
but fails to do so. Frodo sets out on foot with his cousin Pippin Took and gardener Sam Gamgee. 
They are pursued by Black Riders, but meet some Elves, whose singing to Elbereth wards off the Riders. 
The Hobbits take an evasive shortcut to Bucklebury Ferry, where they meet their friend Merry Brandybuck. 
Merry and Pippin reveal they know about the Ring and insist on joining Frodo on his journey.
""",
    """
They try to shake off the Black Riders by cutting through the Old Forest. Merry and Pippin are 
trapped by the malign Old Man Willow, but are rescued by Tom Bombadil. Leaving Tom's house, 
they are caught by a barrow-wight. Frodo, awakening from the barrow-wight's spell, calls Tom Bombadil, 
who frees them and gives them ancient swords from the wight's hoard. The Hobbits reach the village of Bree,
where they meet Strider, a Ranger. The innkeeper gives Frodo an old letter from Gandalf, 
which identifies Strider as a friend. Knowing the Black Riders will attempt to seize the Ring, 
Strider guides the group toward the Elvish sanctuary of Rivendell. 
"""
]

    
    print("\nAnalyzing multiple texts...")
    result = extract_relations_from_texts(test_texts)
        
    print("\nAll relations found:")
    for relation in result['relations']:
        print(f"- {relation['subject']} -> {relation['relation']} -> {relation['object']} [Text {relation['text_index']}]")
        print(f"  Sentence: {relation['sentence']}")
        
    print("\nRelation strings:")
    for rel_str in result['relation_strings']:
        print(f"- {rel_str}")

    return result["relation_strings"]