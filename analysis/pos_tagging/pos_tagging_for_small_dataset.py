import spacy
import pandas as pd
import difflib
from tqdm import tqdm
from collections import Counter
import ast

nlp = spacy.load("en_core_web_sm")
tqdm.pandas()

df = pd.read_csv('top_3_with_corrections.csv')  

def pos_tagging(text):
    doc = nlp(text)
    return [(token.text, token.pos_) for token in doc]

df['original_pos'] = df['text'].progress_apply(pos_tagging)
df['corrected_pos'] = df['correct'].progress_apply(pos_tagging)

df.to_csv('small_dataset_POS.csv', index=False)
print("POS tagging done and saved") 

df = pd.read_csv('small_dataset_POS.csv', encoding='utf-8')

def classify_errors(original_pos, corrected_pos):
    classified_errors = []
    
    original_tokens = [token for token, pos in original_pos]
    corrected_tokens = [token for token, pos in corrected_pos]
    
    diff = list(difflib.ndiff(original_tokens, corrected_tokens))
    
    original_index, corrected_index = 0, 0
    i = 0
    
    while i < len(diff):
        if diff[i][0] == '-' and i + 1 < len(diff) and diff[i + 1][0] == '+':
            if original_index < len(original_pos) and corrected_index < len(corrected_pos):
                original_word, original_pos_tag = original_pos[original_index]
                corrected_word, corrected_pos_tag = corrected_pos[corrected_index]
                error_type = f"{corrected_pos_tag} Error"
                classified_errors.append((error_type, original_word, original_pos_tag, corrected_word, corrected_pos_tag))
                original_index += 1
                corrected_index += 1
            i += 2
        elif diff[i][0] == '-' and original_index < len(original_pos):
            original_word, original_pos_tag = original_pos[original_index]
            error_type = f"{original_pos_tag} Error"
            classified_errors.append((error_type, original_word, original_pos_tag, 'DELETED'))
            original_index += 1
            i += 1
        elif diff[i][0] == '+' and corrected_index < len(corrected_pos):
            corrected_word, corrected_pos_tag = corrected_pos[corrected_index]
            error_type = f"{corrected_pos_tag} Error"
            classified_errors.append((error_type, 'ADDED', corrected_pos_tag, corrected_word))
            corrected_index += 1
            i += 1
        else:
            original_index += 1
            corrected_index += 1
            i += 1
    
    return classified_errors

df['original_pos'] = df['original_pos'].apply(eval)
df['corrected_pos'] = df['corrected_pos'].apply(eval)

df['classified_errors'] = df.progress_apply(lambda row: classify_errors(row['original_pos'], row['corrected_pos']), axis=1)

known_error_types = [
    'Pronoun Error', 'Verb Error', 'Noun Error', 'Adjective Error', 'Preposition Error', 'Other Error',
    'DET Error', 'PROPN Error', 'PART Error', 'ADP Error', 'AUX Error', 'ADJ Error', 'PRON Error', 
    'NOUN Error', 'VERB Error', 'NUM Error', 'PUNCT Error', 'SYM Error', 'X Error'
]

def safe_literal_eval(value):
    if isinstance(value, str):
        try:
            return ast.literal_eval(value)
        except Exception as e:
            print(f"Error evaluating value: {value}. Error: {e}")
            return []  
    return value  

df['classified_errors'] = df['classified_errors'].apply(safe_literal_eval)

def count_error_types(classified_errors):
    error_counter = Counter()
    
    for error_tuple in classified_errors:
        if isinstance(error_tuple, tuple) and len(error_tuple) > 0:
            error_type = error_tuple[0]  # First element in the tuple is the error type
            if error_type in known_error_types:
                error_counter[error_type] += 1
            else:
                print(f"Unknown error type found: {error_type}")
        else:
            print(f"Malformed entry found: {error_tuple}")
    
    return error_counter

df['error_counts'] = df['classified_errors'].progress_apply(count_error_types)

error_pattern_by_language = df.groupby('native')['error_counts'].apply(lambda x: sum(x, Counter()))
error_pattern_df = pd.DataFrame(list(error_pattern_by_language.items()), columns=['native', 'error_counts'])

df.to_csv('classified_error_results.csv', index=False)
error_pattern_df.to_csv('error_analysis_by_language.csv', index=False)

print("Error classification and analysis completed and saved.") 

from ast import literal_eval

df = pd.read_csv('classified_error_results.csv')

def to_counter(entry):
    if isinstance(entry, str):
        if entry.strip() == '':  
            return Counter()
        try:
            if entry.startswith("Counter("):
                entry = entry[8:-1] 
            return Counter(ast.literal_eval(entry))
        except (ValueError, SyntaxError) as e:
            print(f"Warning: Could not convert entry to Counter: {entry}, Error: {e}")
            return Counter()
    return entry if isinstance(entry, Counter) else Counter()  # Handle other non-Counter entries

df['error_counts'] = df['error_counts'].apply(to_counter)
df = df[['native', 'error_counts']]
df.to_csv('inspect', index=False)

grouped_df = df.groupby('native')['error_counts'].apply(lambda x: sum(x, Counter())).reset_index()

grouped_df.to_csv('classified_error_counts_by_language.csv', index=False)

print("The grouped error counts by native language have been saved to 'classified_error_counts_by_language.csv'.")

def calculate_relative_frequency(counter_dict):
    total_errors = sum(counter_dict.values())
    if total_errors > 0:
        return {error_type: count / total_errors for error_type, count in counter_dict.items()}
    return counter_dict  

df = pd.read_csv('classified_error_counts_by_language.csv')

excluded_errors = ["X Error"]
df_filtered = df[~df['level_1'].isin(excluded_errors)]

df_filtered.to_csv('inspect', index=True)

total_errors = df_filtered.groupby('native')['error_counts'].transform('sum')

df_filtered['relative_frequency'] = df_filtered['error_counts'] / total_errors

pivot_df = df_filtered.pivot_table(index='native', columns='level_1', values='relative_frequency', fill_value=0)

total_errors = df.groupby('native')['error_counts'].transform('sum')

pivot_df.to_csv('error_types_relative_frequencies.csv', index=True)
pivot_df.to_pickle('error_types_relative_frequencies.pkl')

print(f"Relative frequencies saved")