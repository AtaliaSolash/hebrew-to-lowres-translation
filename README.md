# Enhancing Translation From Hebrew to Low-Resource Languages Through an English Intermediary Approach

This project explores the use of pivot-based machine translation, where English is used as an intermediary language to improve translation quality between Hebrew-Finnish and Hebrew-Ukrainian language pairs. The approach is evaluated using state-of-the-art translation models and multiple translation quality metrics.

### Research Objective:
1. Enhance translation quality from Hebrew to low-resource languages
2. Compare effectiveness between direct and pivot translation approaches

### Translation Pipeline: 
1. Direct Method: Hebrew → Target Language 
2. Pivot Method: Hebrew → English → Target Language

### Dataset:
TED 2020 subset of the OPUS corpus.
Example data can be found in the TED2020_xml_data folder.

### Data Preprocessing:
Each sentence pair was embedded using the multilingual-e5-large model, and cosine similarity was calculated to identify and remove mismatched or noisy pairs. Pairs with a similarity score below 0.75 were excluded to maintain high semantic alignment. Additionally, sentence pairs containing English words were removed to avoid bias in pivot-based translation steps. Finally, a subset of 20,000 aligned sentence pairs was selected and shared across both language pairs for consistent evaluation.

### Models Used:
1. NLLB-200 (Meta)
2. Helsinki-NLP (OPUS-MT)
3. Google Translate API
4. LLaMA 3.1

### Evaluation Metrics:
1. BLEU — N-gram precision
2. METEOR — Semantic similarity
3. COMET — Neural evaluation

### Conclustions:
1. Pivot-based translation generally outperformed direct translation in Hebrew-Finnish and Hebrew-Ukrainian pairs. 
2. The choice of translation method and evaluation metrics should align with task-specific priorities.

