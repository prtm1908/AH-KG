import os
import subprocess
import tempfile
from typing import List, Dict, Optional
from pathlib import Path

def format_triplets_for_prompt(triplets: List[Dict[str, str]]) -> str:
    """
    Format knowledge graph triplets into a readable text format for the model.
    
    Args:
        triplets: List of dictionaries containing 'first_node', 'relation', and 'second_node'
        
    Returns:
        Formatted string containing the triplets
    """
    formatted_text = "Here is the available information:\n\n"
    
    for triplet in triplets:
        formatted_text += f"- {triplet['first_node']} {triplet['relation']} {triplet['second_node']}\n"
    
    return formatted_text

def create_prompt(query: str, triplets: List[Dict[str, str]]) -> str:
    """
    Create a prompt for the BD3-LM model combining the query and available information.
    
    Args:
        query: The user's question
        triplets: List of knowledge graph triplets
        
    Returns:
        Complete prompt for the model
    """
    context = format_triplets_for_prompt(triplets)
    
    prompt = f"""{context}

Question: {query}

Answer: Based on the information provided above, I will answer your question. If the information needed to answer your question is not present in the provided facts, I will clearly state that I cannot answer the question.

Let me analyze the available information and provide an answer:"""
    
    return prompt

def run_bd3lm_inference(prompt: str, block_size: int = 4, length: int = 512) -> str:
    """
    Run inference using the BD3-LM model.
    
    Args:
        prompt: The input prompt for the model
        block_size: Size of blocks for the model (4, 8, or 16)
        length: Generation length (must be multiple of block_size)
        
    Returns:
        Generated response from the model
    """
    # Get the absolute path to the bd3lms directory
    current_dir = Path(__file__).parent
    bd3lms_dir = current_dir / "bd3lms"
    
    if not bd3lms_dir.exists():
        raise FileNotFoundError(f"BD3-LM directory not found at {bd3lms_dir}")
    
    # Create a temporary directory for logs
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create input prompt file
        prompt_file = os.path.join(temp_dir, "input_prompt.txt")
        with open(prompt_file, "w", encoding="utf-8") as f:
            f.write(prompt)
        
        # Construct the command with absolute paths
        cmd = [
            "python", "-u", str(bd3lms_dir / "main.py"),
            "loader.eval_batch_size=1",
            "model=small",
            "algo=bd3lm",
            "algo.T=5000",
            "data=openwebtext-split",
            f"model.length={length}",
            f"block_size={block_size}",
            "wandb=null",
            "mode=sample_eval",
            f"eval.checkpoint_path=kuleshov-group/bd3lm-owt-block_size{block_size}",
            "model.attn_backend=sdpa",
            "sampling.nucleus_p=0.9",
            "sampling.kv_cache=true",
            f"sampling.input_text={prompt_file}",
            f"sampling.logdir={temp_dir}"
        ]
        
        # Run the command
        try:
            result = subprocess.run(
                cmd,
                cwd=str(bd3lms_dir),  # Run from the bd3lms directory
                capture_output=True,
                text=True
            )
            
            if result.returncode != 0:
                print(f"Error running BD3-LM: {result.stderr}")
                return "I apologize, but I encountered an error while processing your query."
            
            # Extract the generated response from the output
            # The model's output will be in the last generated text
            output_lines = result.stdout.split("\n")
            response = ""
            for line in output_lines:
                if line.startswith("Answer:"):
                    response = line[7:].strip()
                    break
            
            return response if response else "I apologize, but I could not generate a meaningful response."
            
        except Exception as e:
            print(f"Error running BD3-LM: {str(e)}")
            return "I apologize, but I encountered an error while processing your query."

def answer_query_with_knowledge(query: str, triplets: List[Dict[str, str]], 
                              block_size: int = 4, length: int = 512) -> str:
    """
    Main function to answer a query using knowledge graph triplets and BD3-LM.
    
    Args:
        query: The user's question
        triplets: List of knowledge graph triplets
        block_size: Size of blocks for the model (4, 8, or 16)
        length: Generation length (must be multiple of block_size)
        
    Returns:
        Answer to the query based on available information
    """
    # Create the prompt
    prompt = create_prompt(query, triplets)
    
    # Run inference
    response = run_bd3lm_inference(prompt, block_size, length)
    
    return response

# Example usage
if __name__ == "__main__":
    # Example triplets
    sample_triplets = [
        {"first_node": "John", "relation": "loves", "second_node": "football"},
        {"first_node": "John", "relation": "enjoys", "second_node": "basketball"},
        {"first_node": "Mary", "relation": "reads", "second_node": "books"},
        {"first_node": "Mary", "relation": "studies", "second_node": "library"}
    ]
    
    # Example query
    sample_query = "What does John like to do?"
    
    # Get answer
    answer = answer_query_with_knowledge(sample_query, sample_triplets)
    print(f"Query: {sample_query}")
    print(f"Answer: {answer}")
