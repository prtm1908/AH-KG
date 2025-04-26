import spacy
from fastcoref import spacy_component
import torch
import requests
import json

def resolve_text_with_fastcoref(text):
    """
    Process text with spaCy and FastCoref to resolve coreferences.
    
    Args:
        text: Input text to process
        
    Returns:
        Resolved text with coreferences resolved
    """
    print("\nStarting text resolution with FastCoref")
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

def get_pos_tag_meaning(pos_tag):
    """
    Convert Stanford CoreNLP POS tag to its meaning.
    
    Args:
        pos_tag: The POS tag from Stanford CoreNLP
        
    Returns:
        The meaning of the POS tag
    """
    pos_tag_meanings = {
        # Conjunctions
        "CC": "Coordinating conjunction",
        
        # Numbers
        "CD": "Cardinal number",
        
        # Determiners
        "DT": "Determiner",
        
        # Existential
        "EX": "Existential there",
        
        # Foreign words
        "FW": "Foreign word",
        
        # Prepositions
        "IN": "Preposition or subordinating conjunction",
        
        # Adjectives
        "JJ": "Adjective",
        "JJR": "Adjective, comparative",
        "JJS": "Adjective, superlative",
        
        # List markers
        "LS": "List item marker",
        
        # Modals
        "MD": "Modal",
        
        # Nouns
        "NN": "Noun, singular or mass",
        "NNS": "Noun, plural",
        "NNP": "Proper noun, singular",
        "NNPS": "Proper noun, plural",
        
        # Predeterminers
        "PDT": "Predeterminer",
        
        # Possessives
        "POS": "Possessive ending",
        
        # Pronouns
        "PRP": "Personal pronoun",
        "PRP$": "Possessive pronoun",
        
        # Adverbs
        "RB": "Adverb",
        "RBR": "Adverb, comparative",
        "RBS": "Adverb, superlative",
        
        # Particles
        "RP": "Particle",
        
        # Symbols
        "SYM": "Symbol",
        
        # To
        "TO": "to",
        
        # Interjections
        "UH": "Interjection",
        
        # Verbs
        "VB": "Verb, base form",
        "VBD": "Verb, past tense",
        "VBG": "Verb, gerund or present participle",
        "VBN": "Verb, past participle",
        "VBP": "Verb, non-3rd person singular present",
        "VBZ": "Verb, 3rd person singular present",
        
        # Wh-determiners
        "WDT": "Wh-determiner",
        
        # Wh-pronouns
        "WP": "Wh-pronoun",
        "WP$": "Possessive wh-pronoun",
        
        # Wh-adverbs
        "WRB": "Wh-adverb"
    }
    
    return pos_tag_meanings.get(pos_tag, f"Unknown tag: {pos_tag}")

def get_pos_tags_with_stanford_corenlp(text):
    """
    Process text with Stanford CoreNLP to extract POS tags and dependencies.
    
    Args:
        text: Input text to process
        
    Returns:
        List of tokens with their POS tags and dependencies
    """
    print("\nStarting POS tag and dependency extraction with Stanford CoreNLP")
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
        
        # Process each sentence
        for sentence_idx, sentence in enumerate(result.get('sentences', [])):
            print(f"\nSentence {sentence_idx + 1}:")
            tokens = sentence.get('tokens', [])
            dependencies = sentence.get('enhancedPlusPlusDependencies', [])
            print(f"Dependencies: {dependencies}")
            
            # Create a dictionary to store token dependencies
            token_deps = {}
            for dep in dependencies:
                # Skip ROOT and punctuation dependencies
                if dep.get('dep') in ['ROOT', 'punct']:
                    continue
                    
                # Get the indices from the correct fields
                dependent_idx = dep.get('dependent', 0)
                governor_idx = dep.get('governor', 0)
                
                # Debug print for dependency processing
                print(f"Processing dependency: {dep.get('dependentGloss')} -> {dep.get('governorGloss')} ({dep.get('dep')})")
                print(f"Raw indices: dependent={dependent_idx}, governor={governor_idx}")
                
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
            
            # Print each token with its POS tag and dependency info
            for idx, token in enumerate(tokens):
                word = token.get('word', '')
                pos = token.get('pos', '')
                lemma = token.get('lemma', '')
                pos_meaning = get_pos_tag_meaning(pos)
                
                # Basic token info
                print(f"\nWord: {word:<15} POS: {pos:<5} Meaning: {pos_meaning:<30} Lemma: {lemma}")
                
                # If this token has dependencies
                if idx in token_deps:
                    print(f"Found dependencies for {word}:")
                    for dep_info in token_deps[idx]:
                        governor_word = tokens[dep_info['governor_idx']].get('word', '')
                        governor_pos = tokens[dep_info['governor_idx']].get('pos', '')
                        governor_pos_meaning = get_pos_tag_meaning(governor_pos)
                        relation = dep_info['relation']
                        
                        # Format the relationship type for better readability
                        relation_display = relation.replace(':', ' ').title()
                        
                        # Special handling for different types of dependencies
                        if relation in ['advmod', 'neg']:
                            print(f"    └─ Modifies ({relation_display}): {governor_word} ({governor_pos_meaning})")
                        elif relation == 'amod':
                            print(f"    └─ Modifies ({relation_display}): {governor_word} ({governor_pos_meaning})")
                        elif relation in ['nsubj', 'nsubjpass']:
                            print(f"    └─ Subject of: {governor_word} ({governor_pos_meaning})")
                        elif relation == 'obj':
                            print(f"    └─ Object of: {governor_word} ({governor_pos_meaning})")
                        elif relation.startswith('obl:'):
                            print(f"    └─ Oblique ({relation_display}): {governor_word} ({governor_pos_meaning})")
                        else:
                            print(f"    └─ {relation_display}: {governor_word} ({governor_pos_meaning})")
                else:
                    print(f"No dependencies found for {word}")
        
        return result
        
    except Exception as e:
        print(f"Error processing with Stanford CoreNLP: {str(e)}")
        return []

if __name__ == "__main__":
    # Sample text with coreferences
    sample_text = "The incredibly quick, brown fox, who had been diligently searching for hours, finally jumped swiftly over the lazy, sleeping dog and disappeared into Mr. Fitzwilliam's lush, green vegetable garden, much to everyone's surprise."
    
    print("Original text:")
    print(sample_text)
    
    # Step 1: Resolve coreferences with spaCy and FastCoref
    resolved_text = resolve_text_with_fastcoref(sample_text)
    print("\nResolved text:")
    print(resolved_text)
    
    # Step 2: Get POS tags with Stanford CoreNLP
    pos_tags = get_pos_tags_with_stanford_corenlp(resolved_text)
