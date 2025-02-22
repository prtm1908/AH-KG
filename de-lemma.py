import nltk
from lemminflect import getInflection
from collections import Counter

# Download required NLTK data if not already available.
nltk.download('averaged_perceptron_tagger')

def annotate_triplets(triplets):
    """
    Given a list of knowledge graph triplets as tuples (subject, relationship, object),
    this function:
      1. Counts duplicate triplets to determine strength.
      2. Tokenizes each element and obtains Penn-Treebank tags.
      3. Delemmatizes each token using getInflection.
      4. Stores the POS tag, original lemma, and strength in metadata.
    
    Returns a list of dictionaries with keys: 'subject', 'relationship', 'object', 'metadata'.
    """
    # Count frequency of each triplet to determine strength.
    triplet_freq = Counter(triplets)
    annotated = []
    
    for subj, rel, obj in triplets:
        strength = triplet_freq[(subj, rel, obj)]
        
        # Tokenize each element (here using a simple whitespace split)
        subj_tokens = subj.split()
        rel_tokens  = rel.split()
        obj_tokens  = obj.split()
        
        # Get a representative tag (using the first token's tag) for each element.
        subj_tag = nltk.pos_tag([subj_tokens[0]])[0][1] if subj_tokens else ''
        rel_tag  = nltk.pos_tag([rel_tokens[0]])[0][1] if rel_tokens else ''
        obj_tag  = nltk.pos_tag([obj_tokens[0]])[0][1] if obj_tokens else ''
        
        def inflect_tokens(tokens):
            inflected = []
            for token in tokens:
                token_tag = nltk.pos_tag([token])[0][1]
                # Get inflected forms using getInflection
                infls = getInflection(token, token_tag, inflect_oov=True)
                # Use the first candidate if available
                inflected_token = infls[0] if infls else token
                inflected.append(inflected_token)
            return " ".join(inflected)
        
        inflected_subj = inflect_tokens(subj_tokens)
        inflected_rel  = inflect_tokens(rel_tokens)
        inflected_obj  = inflect_tokens(obj_tokens)
        
        annotated.append({
            "subject": inflected_subj,
            "relationship": inflected_rel,
            "object": inflected_obj,
            "metadata": {
                "subject_tag": subj_tag,
                "relationship_tag": rel_tag,
                "object_tag": obj_tag,
                "subject_lemma": subj,
                "relationship_lemma": rel,
                "object_lemma": obj,
                "strength": strength  # strength is the number of times this triplet appears
            }
        })
        
    return annotated

def update_subject_inflection(triplets_with_metadata):
    """
    Given annotated triplets (as produced by annotate_triplets),
    this function re-inflects the subject using the stored original subject lemma
    and its corresponding POS tag.
    
    It calls getInflection on the subject (as a whole) and updates the triple’s subject.
    Returns the updated list of triplets.
    """
    updated_triples = []
    
    for trip in triplets_with_metadata:
        meta = trip.get("metadata", {})
        # Retrieve the original subject lemma and its tag.
        subj_lemma = meta.get("subject_lemma", trip["subject"])
        subj_tag   = meta.get("subject_tag")
        
        if subj_tag:
            # Inflect the whole subject using the original lemma and tag.
            infls = getInflection(subj_lemma, subj_tag, inflect_oov=True)
            new_subj = infls[0] if infls else trip["subject"]
        else:
            new_subj = trip["subject"]
        
        # Create a copy of the triple with the updated subject.
        updated_trip = trip.copy()
        updated_trip["subject"] = new_subj
        updated_triples.append(updated_trip)
        
    return updated_triples

# --- Example usage ---
if __name__ == "__main__":
    # Example list of triplets in lemma form. The first triplet is repeated to demonstrate strength.
    triplets = [
        ("run", "be_part_of", "marathon"),
        ("jump", "lead_to", "success"),
        ("run", "be_part_of", "marathon")  # duplicate, strength should be 2
    ]
    
    # Annotate triplets with POS tags, delemmatized forms, and strength metadata.
    annotated = annotate_triplets(triplets)
    print("Annotated Triplets:")
    for item in annotated:
        print(item)
    
    # Update subject inflection based on metadata.
    updated = update_subject_inflection(annotated)
    print("\nUpdated Triplets (subject inflected):")
    for item in updated:
        print(item)
