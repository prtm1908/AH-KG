import pandas as pd
import numpy as np
import sys
import os
from pathlib import Path

# Add parent directory to path to import from RAG.py
sys.path.append(str(Path(__file__).parent.parent))

from RAG import (
    start_corenlp_server,
    process_text_to_kg,
    extract_query_entities,
    find_matching_nodes,
    extract_relevant_triplets_from_entities,
    OpenIEKnowledgeGraph
)

import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def process_paragraph_to_triplets(text: str, server_process) -> list:
    """Process a paragraph and return knowledge graph triplets."""
    try:
        # Create knowledge graph
        kg = OpenIEKnowledgeGraph()
        success = process_text_to_kg(text, kg)
        
        if not success:
            logger.warning("Failed to process text to knowledge graph")
            return []
        
        # Get all triplets from the graph
        triplets = []
        for u, v, data in kg.G.edges(data=True):
            triplet = {
                'subject': kg.G.nodes[u]['name'],
                'relation': data['relation'],
                'object': kg.G.nodes[v]['name'],
                'strength': data['strength']
            }
            triplets.append(triplet)
        
        return triplets
    
    except Exception as e:
        logger.error(f"Error processing paragraph: {str(e)}")
        return []

def get_relevant_triplets_for_question(text_kg: OpenIEKnowledgeGraph, question: str) -> list:
    """Get relevant triplets for a question using spaCy entities."""
    try:
        # Extract entities from question
        query_entities = extract_query_entities(question)
        
        # Find matching nodes
        matching_nodes = find_matching_nodes(text_kg, query_entities)
        
        if not matching_nodes:
            return []
        
        # Extract relevant triplets
        relevant_triplets = extract_relevant_triplets_from_entities(text_kg, matching_nodes)
        return relevant_triplets
    
    except Exception as e:
        logger.error(f"Error getting relevant triplets: {str(e)}")
        return []

def format_triplets(triplets: list) -> str:
    """Format triplets as a string with newline separation."""
    formatted = []
    for t in triplets:
        formatted.append(f"{t['subject']} | {t['relation']} | {t['object']} | {t['strength']}")
    return "\n".join(formatted)

def main():
    try:
        # Start CoreNLP server
        logger.info("Starting CoreNLP server...")
        server_process = start_corenlp_server()
        
        # Read the dataset
        logger.info("Reading QA dataset...")
        original_df = pd.read_csv('QA_dataset.csv')
        
        # Select 100 random rows
        logger.info("Selecting 100 random rows...")
        if len(original_df) > 100:
            original_df = original_df.sample(n=100, random_state=42)
        
        # Create a new DataFrame for triplets
        triplets_df = pd.DataFrame(index=original_df.index)
        triplets_df['knowledge_graph_triplets'] = ''
        triplets_df['question1_triplets'] = ''
        triplets_df['question2_triplets'] = ''
        triplets_df['question3_triplets'] = ''
        
        # Create output directory if it doesn't exist
        output_dir = 'processed_rows'
        os.makedirs(output_dir, exist_ok=True)
        
        # Process each row
        for idx, row in original_df.iterrows():
            logger.info(f"Processing row {idx + 1}/{len(original_df)}...")
            
            try:
                # Process paragraph to get knowledge graph triplets
                paragraph_triplets = process_paragraph_to_triplets(row['Paragraphs'], server_process)
                triplets_df.at[idx, 'knowledge_graph_triplets'] = format_triplets(paragraph_triplets)
                
                # Create knowledge graph for the paragraph
                kg = OpenIEKnowledgeGraph()
                process_text_to_kg(row['Paragraphs'], kg)
                
                # Get relevant triplets for each question
                for i in range(1, 4):
                    question = row[f'Question{i}']
                    relevant_triplets = get_relevant_triplets_for_question(kg, question)
                    triplets_df.at[idx, f'question{i}_triplets'] = format_triplets(relevant_triplets)
                
                # Save progress after each row
                current_result = pd.concat([
                    original_df.iloc[:idx+1],
                    triplets_df.iloc[:idx+1]
                ], axis=1)
                
                # Save to both a temporary file and the final file
                temp_output_path = os.path.join(output_dir, f'row_{idx+1}_processed.csv')
                final_output_path = 'QA_dataset_with_triplets.csv'
                
                current_result.to_csv(temp_output_path, index=False)
                current_result.to_csv(final_output_path, index=False)
                
                logger.info(f"Saved progress to {temp_output_path}")
                
            except Exception as e:
                logger.error(f"Error processing row {idx}: {str(e)}")
                continue
        
        logger.info("Processing completed!")
        
    except Exception as e:
        logger.error(f"Error in main process: {str(e)}")
    finally:
        # Stop CoreNLP server
        logger.info("Stopping CoreNLP server...")
        server_process.terminate()
        server_process.wait()

if __name__ == "__main__":
    main()
