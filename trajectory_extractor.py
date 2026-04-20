import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import numpy as np
import json
import re
from tqdm import tqdm

MODEL_NAME = "Qwen/Qwen2-0.5B-Instruct"
DATASET_NAME = "gsm8k"
SPLIT = "test"  # Using 'test' for quicker iteration, 'train' for actual training
MAX_SAMPLES = 10  # Reduced for faster iteration on CPU

def load_model():
    print(f"Loading {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, 
        device_map="auto", 
        torch_dtype=torch.float32,
        output_hidden_states=True
    )
    return model, tokenizer

def load_data():
    print(f"Loading {DATASET_NAME}...")
    dataset = load_dataset(DATASET_NAME, "main", split=SPLIT)
    return dataset

def extract_answer(text):
    # GSM8K answers are usually the last number
    # Simple regex to find the last number
    numbers = re.findall(r"[-+]?\d*\.\d+|\d+", text.replace(",", ""))
    if not numbers:
        return None
    return float(numbers[-1])

def compute_stability(model, tokenizer, prompt):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Hidden states: tuple of (batch, seq, dim) for each layer
    # We want the last token of the sequence
    hidden_states = outputs.hidden_states
    
    # 1. Cosine Similarity between layers (for the last token)
    # Shape: (num_layers, batch, seq, dim) -> (num_layers, dim)
    # Note: hidden_states includes embeddings at index 0
    layer_states = [h[0, -1, :] for h in hidden_states[1:]] # Skip embedding layer (index 0)
    layer_states = torch.stack(layer_states) # (num_layers, dim)
    
    # Normalize vectors
    normed_states = F.normalize(layer_states, p=2, dim=-1)
    
    # Compute cosine sim between i and i+1
    # (L-1, dim) * (L-1, dim) -> (L-1,)
    cosine_sims = (normed_states[:-1] * normed_states[1:]).sum(dim=-1).cpu().numpy().tolist()
    
    # 2. Final Prediction Entropy
    logits = outputs.logits[0, -1, :] # (vocab_size,)
    probs = F.softmax(logits, dim=-1)
    entropy = -torch.sum(probs * torch.log(probs + 1e-9)).item()
    
    return cosine_sims, entropy

def generate_answer(model, tokenizer, prompt):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=256, pad_token_id=tokenizer.eos_token_id)
    full_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Extract only the newly generated part
    # A bit hacky, cleaner is to slice input len
    input_len = inputs.input_ids.shape[1]
    generated_text = tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True)
    return generated_text

def run_extraction():
    model, tokenizer = load_model()
    dataset = load_data()
    
    results = []
    
    # Clear file first
    with open("trajectory_data.jsonl", "w") as f:
        pass

    print(f"Processing {MAX_SAMPLES} samples...")
    for i, data in tqdm(enumerate(dataset), total=min(len(dataset), MAX_SAMPLES)):
        if i >= MAX_SAMPLES:
            break
            
        question = data['question']
        ground_truth_str = data['answer']
        
        # Format prompt
        chat = [{"role": "user", "content": question}]
        prompt = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
        
        # 1. Extract Stability Metrics (on Prompt)
        cosine_sims, entropy = compute_stability(model, tokenizer, prompt)
        
        # 2. Generate Answer (to check correctness)
        pred_text = generate_answer(model, tokenizer, prompt)
        
        # 3. Verify Correctness
        pred_val = extract_answer(pred_text)
        truth_val = extract_answer(ground_truth_str)
        
        is_correct = False
        if pred_val is not None and truth_val is not None:
             if abs(pred_val - truth_val) < 1e-4:
                 is_correct = True
                 
        result = {
            "idx": i,
            "cosine_sims": cosine_sims,
            "final_entropy": entropy,
            "is_correct": is_correct,
            "pred_val": pred_val,
            "truth_val": truth_val
        }
        results.append(result)
        
        # Append to file
        with open("trajectory_data.jsonl", "a") as f:
            f.write(json.dumps(result) + "\n")
            
        status = "CORRECT" if is_correct else "WRONG"
        print(f"Sample {i}: {status} (Entropy: {entropy:.2f})")

    print("Done! Saved to trajectory_data.jsonl")

if __name__ == "__main__":
    run_extraction()
