import pickle
from tqdm import tqdm
from google.cloud import translate_v2 as translate
import time

# Load dictionaries
with open("/home/shirmash/research_methods/final_he_fi_dict.pkl", "rb") as f:
    he_fi_dict = pickle.load(f)

# Prepare the source dictionary for translations
he_dict = {key: value[0] for key, value in he_fi_dict.items() if value}

# Initialize Google Translate client
translate_client = translate.Client.from_service_account_json("/home/shirmash/research_methods/shir_key.json")

# Initialize Finnish translation dictionary
finnish_dict = {}

# Perform translations with tqdm for progress logging
for key, hebrew_text in tqdm(he_dict.items(), desc="Translating to Finnish", unit="entry"):
    try:
        # Add small delay to avoid rate limiting
        time.sleep(0.1)
        
        # Step 1: Hebrew to English
        result_en = translate_client.translate(
            hebrew_text,
            source_language="he",
            target_language="en"
        )
        english_text = result_en['translatedText']
        
        # Add another small delay between translations
        time.sleep(0.3)
        
        # Step 2: English to Finnish
        result_fi = translate_client.translate(
            english_text,
            source_language="en",
            target_language="fi"
        )

        # Add to Finnish dictionary
        finnish_dict[key] = [hebrew_text, result_fi['translatedText']]
        
    except Exception as e:
        print(f"Error processing key {key}: {e}")
        finnish_dict[key] = [hebrew_text, None]

# Save the Finnish translations to a pickle file
with open("/home/shirmash/research_methods/he_fi_dict_google_pivot.pkl", "wb") as f:
    pickle.dump(finnish_dict, f)

print("Hebrew to Finnish translations completed!")
