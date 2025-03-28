from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
import torch

def load_model_and_tokenizer(model_name: str = "kuleshov-group/bd3lm-owt-block_size16"):
    """
    Load the BD3-LM model and tokenizer from Hugging Face.
    
    Args:
        model_name: Name of the model to load
        
    Returns:
        model: The loaded model
        tokenizer: The loaded tokenizer
    """
    print(f"Loading model and tokenizer from {model_name}...")
    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_name, config=config, trust_remote_code=True)
    return model, tokenizer

def generate_text(model, tokenizer, prompt: str, max_length: int = 512, 
                 temperature: float = 0.7, top_p: float = 0.9) -> str:
    """
    Generate text using the loaded model.
    
    Args:
        model: The loaded model
        tokenizer: The loaded tokenizer
        prompt: Input prompt text
        max_length: Maximum length of generated text
        temperature: Sampling temperature
        top_p: Nucleus sampling parameter
        
    Returns:
        Generated text
    """
    # Tokenize the input
    inputs = tokenizer(prompt, return_tensors="pt")
    
    # Generate text
    outputs = model.generate(
        **inputs,
        max_length=max_length,
        temperature=temperature,
        top_p=top_p,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id
    )
    
    # Decode and return the generated text
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated_text

def main():
    # Example usage
    model_name = "kuleshov-group/bd3lm-owt-block_size16"
    
    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(model_name)
    
    # Example prompt
    prompt = "Here is the available information:\n\n- John loves football\n- John enjoys basketball\n- Mary reads books\n- Mary studies library\n\nQuestion: What does John like to do?\n\nAnswer: Based on the information provided above, I will answer your question. If the information needed to answer your question is not present in the provided facts, I will clearly state that I cannot answer the question.\n\nLet me analyze the available information and provide an answer:"
    
    # Generate response
    response = generate_text(model, tokenizer, prompt)
    print("\nGenerated Response:")
    print(response)

if __name__ == "__main__":
    main() 