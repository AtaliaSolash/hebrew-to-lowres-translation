import pickle
from tqdm import tqdm
from transformers import pipeline

# Load dictionaries
with open("/home/shirmash/research_methods/final_he_fi_dict.pkl", "rb") as f:
    he_fi_dict = pickle.load(f)

with open("/home/shirmash/research_methods/final_he_uk_dict.pkl", "rb") as f:
    he_uk_dict = pickle.load(f)

he_dict = {key: value[0] for key, value in he_fi_dict.items() if value}

# Load the models for pivot translation
model_name = "facebook/nllb-200-distilled-600M"

# Pipeline for Hebrew to English
pipe_he_to_en = pipeline("translation", max_length=1024, model=model_name, 
                        src_lang="heb_Hebr", tgt_lang="eng_Latn")

# Pipelines from English to target languages
pipe_en_to_fi = pipeline("translation", max_length=1024, model=model_name, 
                        src_lang="eng_Latn", tgt_lang="fin_Latn")
pipe_en_to_uk = pipeline("translation", max_length=1024, model=model_name, 
                        src_lang="eng_Latn", tgt_lang="ukr_Cyrl")

# Initialize dictionaries for translations
finnish_dict = {}
ukrainian_dict = {}

# Perform translations with tqdm for progress logging
for key, hebrew_text in tqdm(he_dict.items(), desc="Translating texts", unit="entry"):
    try:
        # First step: Hebrew to English
        english_translation = pipe_he_to_en(hebrew_text)[0]['translation_text']

        # Second step: English to target languages
        finnish_translation = pipe_en_to_fi(english_translation)[0]['translation_text']
        ukrainian_translation = pipe_en_to_uk(english_translation)[0]['translation_text']

        # Store both the original text and the translations
        finnish_dict[key] = [hebrew_text, finnish_translation]
        ukrainian_dict[key] = [hebrew_text, ukrainian_translation]
        
    except Exception as e:
        print(f"Error processing key {key}: {e}")

# Save the results to pickle files with 'pivot' in the filename
with open("/home/shirmash/research_methods/he_fi_dict_nllb_200_pivot.pkl", "wb") as f:
    pickle.dump(finnish_dict, f)

with open("/home/shirmash/research_methods/he_uk_dict_nllb_200_pivot.pkl", "wb") as f:
    pickle.dump(ukrainian_dict, f)