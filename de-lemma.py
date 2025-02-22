import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from lemminflect import getInflection

# Ensure the required NLTK resources are downloaded.
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
nltk.download('omw-1.4')

def penn_to_wordnet(tag):
    """
    Convert a Penn Treebank POS tag to a WordNet POS tag.
    """
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN

def lemmatize_triplets(raw_triplets):
    """
    Process raw knowledge graph triplets (non-lemmatized) and convert each element into its lemma.
    
    For each element (subject, relationship, object):
      - Tokenize the string.
      - Convert tokens to lowercase before lemmatizing.
      - Retrieve Penn-Treebank POS tags.
      - Lemmatize each token using WordNetLemmatizer.
      - Join tokens back into a lemmatized string.
      - Store the representative (first token's) Penn-Treebank tag and the original raw text in a metadata field.
    
    Returns a list of dictionaries with keys: 'subject', 'relationship', 'object', and 'metadata'.
    """
    lemmatizer = WordNetLemmatizer()
    processed = []
    
    for subj_raw, rel_raw, obj_raw in raw_triplets:
        # Tokenize and convert tokens to lowercase for proper lemmatization.
        subj_tokens = subj_raw.split()
        subj_tokens_lower = [token.lower() for token in subj_tokens]
        rel_tokens = rel_raw.split()
        rel_tokens_lower = [token.lower() for token in rel_tokens]
        obj_tokens = obj_raw.split()
        obj_tokens_lower = [token.lower() for token in obj_tokens]
        
        # Get POS tags for the lowercase tokens.
        subj_pos = nltk.pos_tag(subj_tokens_lower)
        rel_pos  = nltk.pos_tag(rel_tokens_lower)
        obj_pos  = nltk.pos_tag(obj_tokens_lower)
        
        # Use the first token's tag as the representative tag.
        subj_tag = subj_pos[0][1] if subj_pos else ''
        rel_tag  = rel_pos[0][1] if rel_pos else ''
        obj_tag  = obj_pos[0][1] if obj_pos else ''
        
        def lemmatize_tokens(pos_tags):
            lemmas = []
            for token, pos in pos_tags:
                wn_pos = penn_to_wordnet(pos)
                lemma = lemmatizer.lemmatize(token, pos=wn_pos)
                lemmas.append(lemma)
            return " ".join(lemmas)
        
        subj_lemma = lemmatize_tokens(subj_pos)
        rel_lemma  = lemmatize_tokens(rel_pos)
        obj_lemma  = lemmatize_tokens(obj_pos)
        
        processed.append({
            "subject": subj_lemma,
            "relationship": rel_lemma,
            "object": obj_lemma,
            "metadata": {
                "subject_tag": subj_tag,
                "relationship_tag": rel_tag,
                "object_tag": obj_tag,
                "subject_raw": subj_raw,
                "relationship_raw": rel_raw,
                "object_raw": obj_raw
            }
        })
        
    return processed

def de_lemmatize_triplets(triplets_with_metadata):
    """
    Given triplets that have been lemmatized and annotated with metadata, 
    de-lemmatize (i.e. re-inflect) the subject using lemminflect's getInflection function.
    
    For each triplet:
      - Retrieve the lemmatized subject (which is now in its base form) and its stored Penn-Treebank tag.
      - Call getInflection on the base form to obtain the inflected version.
      - Update the subject with the returned candidate (if available).
    
    Returns the updated list of triplets.
    """
    updated_triplets = []
    
    for trip in triplets_with_metadata:
        meta = trip.get("metadata", {})
        subj_lemma = trip.get("subject", "")
        subj_tag = meta.get("subject_tag", "")
        
        if subj_tag:
            # getInflection expects a base form; it returns a tuple of candidate inflections.
            infl_candidates = getInflection(subj_lemma, subj_tag, inflect_oov=True)
            new_subject = infl_candidates[0] if infl_candidates else subj_lemma
        else:
            new_subject = subj_lemma
        
        updated_trip = trip.copy()
        updated_trip["subject"] = new_subject
        updated_triplets.append(updated_trip)
    
    return updated_triplets

# --- Example Usage ---
if __name__ == "__main__":
    # Raw triplets (non-lemmatized) as input.
    raw_triplets = [
        ("Running", "is part of", "a marathon"),
        ("Jumped", "leads to", "success"),
    ]
    
    # First, lemmatize the raw triplets and store metadata.
    lemmatized = lemmatize_triplets(raw_triplets)
    print("Lemmatized Triplets with Metadata:")
    for t in lemmatized:
        print(t)
    
    # Next, de-lemmatize (inflect) the subject using the inflection function.
    de_lemmatized = de_lemmatize_triplets(lemmatized)
    print("\nTriplets after De-lemmatizing the Subject:")
    for t in de_lemmatized:
        print(t)
