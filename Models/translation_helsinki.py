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

# Load Helsinki NLP models for each language pair
pipe_he_to_fi = pipeline("translation", model="Helsinki-NLP/opus-mt-he-fi", max_length=512)
pipe_he_to_uk = pipeline("translation", model="Helsinki-NLP/opus-mt-he-uk", max_length=512)

# Initialize dictionaries for translations
finnish_dict = {}
ukrainian_dict = {}

# Perform translations with tqdm for progress logging
for key, hebrew_text in tqdm(he_dict.items(), desc="Translating texts", unit="entry"):
    try:
        # Finnish Translation
        finnish_translation = pipe_he_to_fi(hebrew_text)[0]['translation_text']

        # Ukrainian Translation
        ukrainian_translation = pipe_he_to_uk(hebrew_text)[0]['translation_text']

        # Add to respective dictionaries
        finnish_dict[key] = [hebrew_text, finnish_translation]
        ukrainian_dict[key] = [hebrew_text, ukrainian_translation]
    except Exception as e:
        print(f"Error processing key {key}: {e}")

# Check the number of translations performed
print(f"Total Finnish Translations: {len(finnish_dict)}")
print(f"Total Ukrainian Translations: {len(ukrainian_dict)}")

# Save the results to pickle files
with open("/home/shirmash/research_methods/he_fi_dict_helsinki_direct.pkl", "wb") as f:
    pickle.dump(finnish_dict, f)

with open("/home/shirmash/research_methods/he_uk_dict_helsinki_direct.pkl", "wb") as f:
    pickle.dump(ukrainian_dict, f)


