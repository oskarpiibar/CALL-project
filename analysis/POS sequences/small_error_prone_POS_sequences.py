import pandas as pd
from collections import Counter, defaultdict
from ast import literal_eval
from tqdm import tqdm

df = pd.read_csv("analysis/classified_error_results.csv")

df['classified_errors'] = df['classified_errors'].apply(lambda x: literal_eval(x) if isinstance(x, str) else x)
df['corrected_pos'] = df['corrected_pos'].apply(lambda x: literal_eval(x) if isinstance(x, str) else x)
df['original_pos'] = df['original_pos'].apply(lambda x: literal_eval(x) if isinstance(x, str) else x)

language_pos_sequence_counters = defaultdict(Counter)

def get_surrounding_sequence_full_match(pos_list, error_word, error_pos_tag):
    for i, item in enumerate(pos_list):
        if isinstance(item, tuple) and len(item) == 2:
            word, pos = item
            if word == error_word and pos == error_pos_tag:
                if i > 0 and i < len(pos_list) - 1:
                    sequence = (pos_list[i - 1][1], pos, pos_list[i + 1][1])
                    return sequence
    return None

for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing Rows"):
    if row['classified_errors'] != []:
        native_language = row['native']
        corrected_pos = row['corrected_pos']
        original_pos = row['original_pos']
        
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

                if sequence:
                    language_pos_sequence_counters[native_language][sequence] += 1

output_data = []
for language, counter in language_pos_sequence_counters.items():
    top_sequences = counter.most_common(4)
    for sequence, count in top_sequences:
        output_data.append({
            "Native": language,
            "POS Sequence": sequence,
            "Count": count
        })

output_df = pd.DataFrame(output_data)
output_df.to_csv("analysis/POS sequences/4error_prone_POS_sequences_small.csv", index = False)
print("CSV file 'error_prone_POS_sequences_small.csv' created successfully.")