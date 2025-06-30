import pickle
from tqdm import tqdm
from transformers import pipeline

# Load dictionaries
with open("/home/shirmash/research_methods/final_he_fi_dict.pkl", "rb") as f:
    he_fi_dict = pickle.load(f)

with open("/home/shirmash/research_methods/final_he_uk_dict.pkl", "rb") as f:
    he_uk_dict = pickle.load(f)

# Prepare the source dictionary for translations
he_dict = {key: value[0] for key, value in he_fi_dict.items() if value}

# Load all required Helsinki NLP models for the pivot translation
pipe_he_to_en = pipeline("translation", model="Helsinki-NLP/opus-mt-tc-big-he-en", max_length=512)
pipe_en_to_fi = pipeline("translation", model="Helsinki-NLP/opus-mt-en-fi", max_length=512)
pipe_en_to_uk = pipeline("translation", model="Helsinki-NLP/opus-mt-en-uk", max_length=512)

# Initialize dictionaries for translations
finnish_dict = {}
ukrainian_dict = {}

# Perform translations with tqdm for progress logging
for key, hebrew_text in tqdm(he_dict.items(), desc="Translating texts", unit="entry"):
    try:
        # Step 1: Hebrew to English
        english_translation = pipe_he_to_en(hebrew_text)[0]['translation_text']

        # Step 2a: English to Finnish
        finnish_translation = pipe_en_to_fi(english_translation)[0]['translation_text']

        # Step 2b: English to Ukrainian
        ukrainian_translation = pipe_en_to_uk(english_translation)[0]['translation_text']

        # Add to respective dictionaries
        finnish_dict[key] = [hebrew_text, finnish_translation]
        ukrainian_dict[key] = [hebrew_text, ukrainian_translation]

    except Exception as e:
        print(f"Error processing key {key}: {e}")
        # Optionally store failed translations with None
        finnish_dict[key] = [hebrew_text, None]
        ukrainian_dict[key] = [hebrew_text, None]


# Save the results to pickle files with a name indicating pivot translation
with open("/home/shirmash/research_methods/he_fi_dict_helsinki_pivot.pkl", "wb") as f:
    pickle.dump(finnish_dict, f)

with open("/home/shirmash/research_methods/he_uk_dict_helsinki_pivot.pkl", "wb") as f:
    pickle.dump(ukrainian_dict, f)