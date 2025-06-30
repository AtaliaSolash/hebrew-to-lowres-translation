import pickle
from tqdm import tqdm
import time
from openai import OpenAI
import os

# Initialize the OpenAI client
client = OpenAI(api_key='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')  # Replace with your actual API key

def translate_with_gpt4(text, source_lang, target_lang):
    """
    Translate text using GPT-4 API
    """
    try:
        prompt = f"Translate this {source_lang} text to {target_lang}. Only provide the translation, no explanations:\n\n{text}"
        
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a professional translator."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3  # Lower temperature for more consistent translations
        )
        
        return response.choices[0].message.content.strip()
    
    except Exception as e:
        print(f"Translation error: {str(e)}")
        return None

# Load dictionaries
with open("/home/shirmash/research_methods/final_he_fi_dict.pkl", "rb") as f:
    he_fi_dict = pickle.load(f)

with open("/home/shirmash/research_methods/final_he_uk_dict.pkl", "rb") as f:
    he_uk_dict = pickle.load(f)

# Prepare the source dictionary for translations
he_dict = {key: value[0] for key, value in he_fi_dict.items() if value}

# Initialize dictionaries for translations
finnish_dict = {}
ukrainian_dict = {}

# Create a checkpoint file to save progress
checkpoint_file = "/home/shirmash/research_methods/translation_checkpoint.pkl"

# Load checkpoint if exists
start_index = 0
if os.path.exists(checkpoint_file):
    with open(checkpoint_file, "rb") as f:
        checkpoint_data = pickle.load(f)
        finnish_dict = checkpoint_data["finnish"]
        ukrainian_dict = checkpoint_data["ukrainian"]
        start_index = checkpoint_data["index"]

# Convert dictionary items to list for indexing
items = list(he_dict.items())[start_index:]

# Perform translations with tqdm for progress logging
for i, (key, hebrew_text) in enumerate(tqdm(items, desc="Translating texts", unit="entry")):
    try:
        # Step 1: Hebrew to English
        english_text = translate_with_gpt4(hebrew_text, "Hebrew", "English")
        time.sleep(1)  # Rate limiting
        
        if english_text:
            # Step 2a: English to Finnish
            finnish_text = translate_with_gpt4(english_text, "English", "Finnish")
            time.sleep(1)  # Rate limiting
            
            # Step 2b: English to Ukrainian
            ukrainian_text = translate_with_gpt4(english_text, "English", "Ukrainian")
            
            # Add to respective dictionaries
            finnish_dict[key] = [hebrew_text, finnish_text]
            ukrainian_dict[key] = [hebrew_text, ukrainian_text]
            
            # Optional: Print intermediate translations for verification
            # print(f"\nKey: {key}")
            # print(f"Hebrew: {hebrew_text}")
            # print(f"English: {english_text}")
            # print(f"Finnish: {finnish_text}")
            # print(f"Ukrainian: {ukrainian_text}\n")
            
            # Save checkpoint every 10 translations
            if i % 10 == 0:
                checkpoint_data = {
                    "finnish": finnish_dict,
                    "ukrainian": ukrainian_dict,
                    "index": start_index + i + 1
                }
                with open(checkpoint_file, "wb") as f:
                    pickle.dump(checkpoint_data, f)
        
        # Add delay to respect rate limits
        time.sleep(1)
        
    except Exception as e:
        print(f"Error processing key {key}: {e}")
        finnish_dict[key] = [hebrew_text, None]
        ukrainian_dict[key] = [hebrew_text, None]
        
        # Save checkpoint on error
        checkpoint_data = {
            "finnish": finnish_dict,
            "ukrainian": ukrainian_dict,
            "index": start_index + i + 1
        }
        with open(checkpoint_file, "wb") as f:
            pickle.dump(checkpoint_data, f)

# Save the final results to pickle files
with open("/home/shirmash/research_methods/he_fi_dict_gpt4_pivot_en.pkl", "wb") as f:
    pickle.dump(finnish_dict, f)

with open("/home/shirmash/research_methods/he_uk_dict_gpt4_pivot_en.pkl", "wb") as f:
    pickle.dump(ukrainian_dict, f)

# Remove checkpoint file after successful completion
if os.path.exists(checkpoint_file):
    os.remove(checkpoint_file)

print("Translation completed!")