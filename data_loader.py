from datasets import load_dataset

def load_gsm8k():
    print("Loading GSM8K dataset...")
    dataset = load_dataset("gsm8k", "main")
    print(f"Loaded GSM8K: {dataset}")
    return dataset

if __name__ == "__main__":
    data = load_gsm8k()
    print("Train split size:", len(data['train']))
    print("Test split size:", len(data['test']))
    print("Sample entry:", data['train'][0])
