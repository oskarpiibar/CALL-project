from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pandas as pd
from tqdm import tqdm

error_types_relative_df = pd.read_pickle('error_types_relative_frequencies.pkl')

inconclusive_count = 0

def classify_native_language(new_text_profile, error_types_relative_df, confidence_threshold):
    global inconclusive_count 
    
    new_text_vector = pd.DataFrame(new_text_profile, index=[0]).reindex(columns=error_types_relative_df.columns, fill_value=0)
    
    new_text_vector = new_text_vector.fillna(0)
    error_types_relative_df = error_types_relative_df.fillna(0)

    similarities = cosine_similarity(new_text_vector, error_types_relative_df)
    
    max_similarity = np.max(similarities)
    
    if max_similarity < confidence_threshold:
        inconclusive_count += 1  
        return "Inconclusive"
    
    most_similar_index = np.argmax(similarities)
    return error_types_relative_df.index[most_similar_index]

def classify_texts(df, error_types_relative_df, confidence_threshold=0.95):
    results = []

    for index, row in tqdm(df.iterrows(), total=len(df), desc="Classifying texts"):
        new_text_profile = row.drop(['text', 'correct'], errors='ignore').to_dict()

        predicted_language = classify_native_language(new_text_profile, error_types_relative_df, confidence_threshold)

        results.append({
            'text': row.get('text', None), 
            'correct': row.get('correct', None),      
            'predicted_native_language': predicted_language
        })

    return pd.DataFrame(results)

df_to_classify = pd.read_pickle('error_types_relative_frequencies_with_text.pkl')  # Replace with your actual file

classified_texts_df = classify_texts(df_to_classify, error_types_relative_df)

target_languages = ['Spanish', 'Russian', 'Chinese (Mandarin)']

for language in target_languages:
    language_df = classified_texts_df[classified_texts_df['predicted_native_language'] == language]
    
    if not language_df.empty:
        filename = f'classified_texts_{language}_95.csv'
        language_df.to_csv(filename, index=False)
        print(f"Saved {len(language_df)} rows for {language} to {filename}")

print(f"Number of inconclusive predictions: {inconclusive_count}")