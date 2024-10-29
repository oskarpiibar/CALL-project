import pandas as pd
from collections import Counter
from ast import literal_eval
from tqdm import tqdm

df_big = pd.read_csv("analysis/intermediate_results.csv")
df_small = pd.read_csv("analysis/POS sequences/error_prone_POS_sequences_small.csv")

df_small['POS Sequence'] = df_small['POS Sequence'].apply(lambda x: literal_eval(x) if isinstance(x, str) else x)
df_big['classified_errors'] = df_big['classified_errors'].apply(lambda x: literal_eval(x) if isinstance(x, str) else x)
df_big['corrected_pos'] = df_big['corrected_pos'].apply(lambda x: literal_eval(x) if isinstance(x, str) else x)
df_big['original_pos'] = df_big['original_pos'].apply(lambda x: literal_eval(x) if isinstance(x, str) else x)

sequence_to_native = dict(zip(df_small['POS Sequence'], df_small['Native']))
matching_rows = []

def get_surrounding_sequence_full_match(pos_list, error_word, error_pos_tag):
    for i, item in enumerate(pos_list):
        if isinstance(item, tuple) and len(item) == 2:
            word, pos = item
            if word == error_word and pos == error_pos_tag:
                if i > 0 and i < len(pos_list) - 1:
                    sequence = (pos_list[i - 1][1], pos, pos_list[i + 1][1])
                    return sequence
    return None

for _, row in tqdm(df_big.iterrows(), total=len(df_big), desc="Processing Rows"):
    if row['classified_errors'] != []:
        corrected_pos = row['corrected_pos']
        original_pos = row['original_pos']
        text = row['text']
        correct = row['correct']
        
        for error in row['classified_errors']:
            if len(error) >= 3:

                if error[1] == 'ADDED':
                    # For `ADDED`, the word is the last item, and POS tag is the second-to-last
                    error_word = error[-1]
                    error_pos_tag = error[-2]
                    sequence = get_surrounding_sequence_full_match(corrected_pos, error_word, error_pos_tag)
                
                elif error[-1] == 'DELETED':
                    # For `DELETED`, the word is the second item and POS tag is the third
                    error_word = error[1]
                    error_pos_tag = error[2]
                    sequence = get_surrounding_sequence_full_match(original_pos, error_word, error_pos_tag)
                
                else:
                    # For other actions (e.g., replacement), use the second item as `word` and third as `POS`
                    error_word = error[-2]
                    error_pos_tag = error[-1]
                    sequence = get_surrounding_sequence_full_match(corrected_pos, error_word, error_pos_tag)

                if sequence and sequence in sequence_to_native:
                    matching_rows.append({
                        "text": text,
                        "correct": correct,
                        "sequence": sequence,
                        "native": sequence_to_native[sequence]
                    })

new_df = pd.DataFrame(matching_rows)

target_languages = ['Russian', 'Chinese (Mandarin)', 'Spanish']

for language in target_languages:
    # Filter the DataFrame to include only rows where predicted_native_language matches the language
    language_df = new_df[new_df['native'] == language]
    
    if not language_df.empty:
        filename = f'sequence_extracted_texts_{language}.csv'
        language_df.to_csv(filename, index=False)
        print(f"Saved {len(language_df)} rows for {language} to {filename}")
