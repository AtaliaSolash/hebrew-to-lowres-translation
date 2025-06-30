import transformers
import torch
from tqdm import tqdm
import pickle

# Initialize model
model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"
pipeline = transformers.pipeline(
    "text-generation",
    model=model_id,
    model_kwargs={
        "torch_dtype": torch.bfloat16,
        "device_map": "auto",
    }
)

def translate_text(text, source_lang, target_lang):
    """
    Translation function that returns only the assistant's translation.
    """
    messages = [
        {"role": "system", "content": f"You are a professional translator. Always translate exactly to {target_lang} language only and return only the translation."},
        {"role": "user", "content": f"Translate this {source_lang} text to {target_lang}: {text}"}
    ]
    
    outputs = pipeline(
        messages,
        max_new_tokens=600,
        temperature=0.3,
        top_p=0.9,
        pad_token_id=pipeline.tokenizer.eos_token_id
    )
    
    # Extract just the assistant's response content
    messages = outputs[0]['generated_text']
    for message in messages:
        if message['role'] == 'assistant':
            return message['content']
    
    return None

# Load dictionary
print("Loading dictionary...")
with open("/home/shirmash/research_methods/final_he_fi_dict.pkl", "rb") as f:
    he_fi_dict = pickle.load(f)

# Initialize Ukrainian translation dictionary
ukrainian_dict = {}

# Create checkpoint file
checkpoint_file = "llama3_translation_checkpoint_pivot_uk.pkl"

# Process all items
total_items = len(he_fi_dict)
items = list(he_fi_dict.items())
print(f"Total items to process: {total_items}")

try:
    # Main progress bar for overall progress
    with tqdm(total=total_items, desc="Hebrew → English → Ukrainian") as pbar:
        for idx, (key, value) in enumerate(items):
            hebrew_text = value[0]
            
            try:
                # Pivot translation: Hebrew -> English -> Ukrainian
                english_text = translate_text(hebrew_text, "Hebrew", "English")
                ukrainian_text = translate_text(english_text, "English", "Ukrainian")
                
                # Store translations
                ukrainian_dict[key] = [hebrew_text,krainian_text]
                
                # Update progress
                pbar.update(1)
                
                # Save checkpoint every 50 items
                if (idx + 1) % 50 == 0:
                    with open(checkpoint_file, "wb") as f:
                        pickle.dump(ukrainian_dict, f)
                    print(f"Checkpoint saved at item {idx + 1}/{total_items}")
                    
            except Exception as e:
                print(f"\nError processing key {key}: {e}")
                ukrainian_dict[key] = [hebrew_text ,None]
                pbar.update(1)

finally:
    # Save final results (even if interrupted)
    print("\nSaving final Ukrainian translations...")
    with open("/home/shirmash/research_methods/translations/he_uk_dict_llama3_pivot.pkl", "wb") as f:
        pickle.dump(ukrainian_dict, f)
    
    print("Ukrainian translation completed!")
    print(f"Total items processed: {len(ukrainian_dict)}")
