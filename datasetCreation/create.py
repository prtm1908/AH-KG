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
    OpenIEKnowledgeGraph,
    get_nodes_at_depth_two
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
            # Get the original forms from the edge data
            triplet = {
                'subject': kg.G.nodes[u]['name'],  # Keep lemmatized form
                'relation': data['relation'],
                'object': kg.G.nodes[v]['name'],  # Keep lemmatized form
                'strength': data['strength'],
                'original_subject': data['original_subject'],
                'original_object': data['original_object']
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
        relevant_triplets = []
        
        # For each matching node, get its neighborhood
        for node in matching_nodes:
            logger.info(f"Getting neighborhood for node: {text_kg.G.nodes[node]['name']}")
            # Get all nodes and edges within 2 hops
            nodes, edges = get_nodes_at_depth_two(text_kg.G, node)
            
            # Convert edges to triplets with original forms
            for u, v, data in edges:
                triplet = {
                    'subject': text_kg.G.nodes[u]['name'],  # Keep lemmatized form
                    'relation': data['relation'],
                    'object': text_kg.G.nodes[v]['name'],  # Keep lemmatized form
                    'strength': data['strength'],
                    'original_subject': data['original_subject'],
                    'original_object': data['original_object']
                }
                
                if triplet not in relevant_triplets:
                    relevant_triplets.append(triplet)
        
        return relevant_triplets
    
    except Exception as e:
        logger.error(f"Error getting relevant triplets: {str(e)}")
        return []

def format_triplets(triplets: list) -> str:
    """Format triplets as a string with newline separation, using original forms."""
    formatted = []
    for t in triplets:
        # Use original forms directly from the triplet
        formatted.append(f"{t['original_subject']} -> {t['relation']} -> {t['original_object']}")
    formatted_str = "\n".join(formatted)
    print("\nFormatted triplets being saved to CSV:")
    print(formatted_str)
    return formatted_str

def main():
    try:
        # Start CoreNLP server once at the beginning
        logger.info("Starting CoreNLP server...")
        server_process = start_corenlp_server()
        
        try:
            # Check if there's an existing progress file
            if os.path.exists('QA_dataset_with_triplets.csv'):
                logger.info("Found existing progress, loading it...")
                df = pd.read_csv('QA_dataset_with_triplets.csv')
                
                # Find rows that haven't been processed yet (empty knowledge graph triplets)
                unprocessed_mask = df['knowledge_graph_triplets'].isna() | (df['knowledge_graph_triplets'] == '')
                unprocessed_rows = df[unprocessed_mask]
                
                logger.info(f"Found {len(unprocessed_rows)} unprocessed rows out of {len(df)} total rows")
                
                if len(unprocessed_rows) == 0:
                    logger.info("All rows have been processed already!")
                    return
                
                # Process each unprocessed row
                for idx in unprocessed_rows.index:
                    logger.info(f"Processing row {idx + 1}/{len(df)}...")
                    
                    try:
                        row = df.loc[idx]
                        # Process paragraph to get knowledge graph triplets
                        paragraph_triplets = process_paragraph_to_triplets(row['Paragraphs'], server_process)
                        df.at[idx, 'knowledge_graph_triplets'] = format_triplets(paragraph_triplets)
                        
                        # Create knowledge graph for the paragraph
                        kg = OpenIEKnowledgeGraph()
                        process_text_to_kg(row['Paragraphs'], kg)
                        
                        # Get relevant triplets for each question
                        for i in range(1, 4):
                            question = row[f'Question{i}']
                            relevant_triplets = get_relevant_triplets_for_question(kg, question)
                            df.at[idx, f'question{i}_triplets'] = format_triplets(relevant_triplets)
                        
                        # Save progress after each row
                        df.to_csv('QA_dataset_with_triplets.csv', index=False)
                        logger.info(f"Processed row {idx}")
                        
                    except Exception as e:
                        logger.error(f"Error processing row {idx}: {str(e)}")
                        continue
                
            else:
                # If no existing file, create new one with 100 random rows
                logger.info("No existing progress file found. Creating new one...")
                
                # Read the original dataset
                logger.info("Reading QA dataset...")
                original_df = pd.read_csv('QA_dataset.csv')
                
                # Select 100 random rows
                logger.info("Selecting 100 random rows...")
                if len(original_df) > 100:
                    df = original_df.sample(n=100)
                else:
                    df = original_df
                
                # Add empty columns for triplets
                df['knowledge_graph_triplets'] = ''
                df['question1_triplets'] = ''
                df['question2_triplets'] = ''
                df['question3_triplets'] = ''
                
                # Save the initial file
                df.to_csv('QA_dataset_with_triplets.csv', index=False)
                
                # Process each row
                for idx in df.index:
                    logger.info(f"Processing row {idx + 1}/{len(df)}...")
                    
                    try:
                        row = df.loc[idx]
                        # Process paragraph to get knowledge graph triplets
                        paragraph_triplets = process_paragraph_to_triplets(row['Paragraphs'], server_process)
                        df.at[idx, 'knowledge_graph_triplets'] = format_triplets(paragraph_triplets)
                        
                        # Create knowledge graph for the paragraph
                        kg = OpenIEKnowledgeGraph()
                        process_text_to_kg(row['Paragraphs'], kg)
                        
                        # Get relevant triplets for each question
                        for i in range(1, 4):
                            question = row[f'Question{i}']
                            relevant_triplets = get_relevant_triplets_for_question(kg, question)
                            df.at[idx, f'question{i}_triplets'] = format_triplets(relevant_triplets)
                        
                        # Save progress after each row
                        df.to_csv('QA_dataset_with_triplets.csv', index=False)
                        logger.info(f"Processed row {idx}")
                        
                    except Exception as e:
                        logger.error(f"Error processing row {idx}: {str(e)}")
                        continue
            
            logger.info("Processing completed!")
            
        finally:
            # Stop CoreNLP server in the inner try-finally to ensure it's stopped even if processing fails
            logger.info("Stopping CoreNLP server...")
            server_process.terminate()
            server_process.wait()
            
    except Exception as e:
        logger.error(f"Error in main process: {str(e)}")

if __name__ == "__main__":
    main()
