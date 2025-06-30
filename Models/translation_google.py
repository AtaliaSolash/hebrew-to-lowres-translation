import pickle
from tqdm import tqdm
from google.cloud import translate_v2 as translate
import time

# Load dictionaries
with open("/home/shirmash/research_methods/final_he_fi_dict.pkl", "rb") as f:
    he_fi_dict = pickle.load(f)

with open("/home/shirmash/research_methods/final_he_uk_dict.pkl", "rb") as f:
    he_uk_dict = pickle.load(f)

# Prepare the source dictionary for translations
he_dict = {key: value[0] for key, value in he_fi_dict.items() if value}

# Initialize Google Translate client
translate_client = translate.Client.from_service_account_json("/home/shirmash/research_methods/shir_key.json")

# Initialize dictionaries for translations
finnish_dict = {}
ukrainian_dict = {}

# Perform translations with tqdm for progress logging
for key, hebrew_text in tqdm(he_dict.items(), desc="Translating texts", unit="entry"):
    try:
        # Add small delay to avoid rate limiting
        time.sleep(0.2)
        
        # Finnish Translation
        result_fi = translate_client.translate(
            hebrew_text,
            source_language="he",
            target_language="fi"
        )
        
        # Ukrainian Translation
        result_uk = translate_client.translate(
            hebrew_text,
            source_language="he",
            target_language="uk"
        )

        # Add to respective dictionaries
        finnish_dict[key] = [hebrew_text, result_fi['translatedText']]
        ukrainian_dict[key] = [hebrew_text, result_uk['translatedText']]
    except Exception as e:
        print(f"Error processing key {key}: {e}")

# Check the number of translations performed
print(f"Total Finnish Translations: {len(finnish_dict)}")
print(f"Total Ukrainian Translations: {len(ukrainian_dict)}")

# Save the results to pickle files
with open("/home/shirmash/research_methods/he_fi_dict_google_direct.pkl", "wb") as f:
    pickle.dump(finnish_dict, f)

with open("/home/shirmash/research_methods/he_uk_dict_google_direct.pkl", "wb") as f:
    pickle.dump(ukrainian_dict, f)