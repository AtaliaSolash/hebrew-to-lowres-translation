import pickle
from tqdm import tqdm
from transformers import pipeline

# Load dictionaries
with open("/home/shirmash/research_methods/final_he_fi_dict.pkl", "rb") as f:
    he_fi_dict = pickle.load(f)

with open("/home/shirmash/research_methods/final_he_uk_dict.pkl", "rb") as f:
    he_uk_dict = pickle.load(f)

he_dict = {key: value[0] for key, value in he_fi_dict.items() if value}

# Load the model
model_name = "facebook/nllb-200-distilled-600M"
pipe_he_to_fi = pipeline("translation", max_length=1024, model=model_name, src_lang="heb_Hebr", tgt_lang="fin_Latn")
pipe_he_to_uk = pipeline("translation", max_length=1024, model=model_name, src_lang="heb_Hebr", tgt_lang="ukr_Cyrl")

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


# Save the results to pickle files
with open("/home/shirmash/research_methods/he_fi_dict_nllb_200_direct.pkl", "wb") as f:
    pickle.dump(finnish_dict, f)

with open("/home/shirmash/research_methods/he_uk_dict_nllb_200_direct.pkl", "wb") as f:
    pickle.dump(ukrainian_dict, f)





