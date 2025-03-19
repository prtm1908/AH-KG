import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from lemminflect import getInflection

# Ensure the required NLTK resources are downloaded.
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
nltk.download('omw-1.4')

# Define a mapping dictionary for POS tags.
# We assume only a limited set of tags (up to 10) are used.
POS_TAG_MAPPING = {
    'NN': 0,    # singular noun
    'NNS': 1,   # plural noun
    'VB': 2,    # base form verb
    'VBD': 3,   # past tense verb
    'VBG': 4,   # gerund/participle verb
    'VBZ': 5,   # 3rd person singular verb
    'JJ': 6,    # adjective
    'RB': 7,    # adverb
    'DT': 8,    # determiner
    'PRP': 9    # pronoun
}

# Inverse mapping to recover the original tag from its code.
INVERSE_POS_TAG_MAPPING = {v: k for k, v in POS_TAG_MAPPING.items()}

def encode_pos_tag(tag):
    """
    Encodes a Penn Treebank POS tag as a 4-bit integer.
    Defaults to 'NN' if the tag is not in the mapping.
    """
    return POS_TAG_MAPPING.get(tag, POS_TAG_MAPPING['NN'])

def decode_pos_tag(code):
    """
    Decodes the 4-bit integer back into its Penn Treebank POS tag string.
    Defaults to 'NN' if the code is not in the inverse mapping.
    """
    return INVERSE_POS_TAG_MAPPING.get(code, 'NN')

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
      - Store the representative (first token's) POS tag as a 4-bit integer in metadata.
    
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
        subj_tag_str = subj_pos[0][1] if subj_pos else 'NN'
        rel_tag_str  = rel_pos[0][1] if rel_pos else 'NN'
        obj_tag_str  = obj_pos[0][1] if obj_pos else 'NN'
        
        # Encode the POS tags into a 4-bit integer.
        encoded_subj_tag = encode_pos_tag(subj_tag_str)
        encoded_rel_tag  = encode_pos_tag(rel_tag_str)
        encoded_obj_tag  = encode_pos_tag(obj_tag_str)
        
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
                "subject_tag": encoded_subj_tag,
                "relationship_tag": encoded_rel_tag,
                "object_tag": encoded_obj_tag
            }
        })
        
    return processed

def de_lemmatize_triplets(triplets_with_metadata):
    """
    Given triplets that have been lemmatized and annotated with metadata, 
    de-lemmatize (i.e. re-inflect) the subject using lemminflect's getInflection function.
    
    For each triplet:
      - Retrieve the lemmatized subject (base form) and its stored 4-bit POS tag.
      - Decode the stored tag back to its original Penn-Treebank string.
      - Call getInflection on the base form to obtain the inflected version.
      - Update the subject with the returned candidate (if available).
    
    Returns the updated list of triplets.
    """
    updated_triplets = []
    
    for trip in triplets_with_metadata:
        meta = trip.get("metadata", {})
        subj_lemma = trip.get("subject", "")
        # Retrieve the stored 4-bit POS code and decode it.
        subj_tag_code = meta.get("subject_tag", None)
        if subj_tag_code is not None:
            decoded_subj_tag = decode_pos_tag(subj_tag_code)
            # getInflection expects the base form and the tag string.
            infl_candidates = getInflection(subj_lemma, decoded_subj_tag, inflect_oov=True)
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
    
    # First, lemmatize the raw triplets and store metadata with encoded POS tags.
    lemmatized = lemmatize_triplets(raw_triplets)
    print("Lemmatized Triplets with Encoded POS Tag Metadata:")
    for t in lemmatized:
        print(t)
    
    # Next, de-lemmatize (inflect) the subject by decoding the POS tag.
    de_lemmatized = de_lemmatize_triplets(lemmatized)
    print("\nTriplets after De-lemmatizing the Subject:")
    for t in de_lemmatized:
        print(t)
