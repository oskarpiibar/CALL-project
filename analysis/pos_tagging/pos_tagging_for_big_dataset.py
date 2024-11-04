import spacy
import pandas as pd
from collections import Counter
from tqdm import tqdm
import difflib

nlp = spacy.load("en_core_web_sm")

try:
    df = pd.read_csv('big_dataset.csv', encoding='utf-8')
    print("Dataset loaded successfully with UTF-8 encoding.")
except UnicodeDecodeError:
    df = pd.read_csv('big_dataset.csv', encoding='ISO-8859-1')
    print("Dataset loaded with ISO-8859-1 encoding due to UTF-8 decoding error.")

df.dropna(subset=['correct'], inplace=True)
df = df[df['correct'].str.strip() != ""]

if df['correct'].isna().any() or (df['correct'].str.strip() == "").any():
    raise ValueError("NaN or empty values found in 'correct' column after cleaning.")

tqdm.pandas()

def pos_tagging(text, index):
    try:
        if isinstance(text, str) and text.strip():
            doc = nlp(text)
            return [(token.text, token.pos_) for token in doc]
        else:
            raise ValueError(f"Invalid text input at row {index}: {text}")
    except Exception as e:
        print(f"Skipping row {index} due to error: {e}")
        return None

try:
    df = pd.read_csv('intermediate_results.csv', encoding='utf-8')
    print("Loaded intermediate results from 'intermediate_results.csv'")
except FileNotFoundError:
    print("No intermediate results found, starting from scratch.")

def verify_pos_structure(pos_list):
    if isinstance(pos_list, str):
        try:
            pos_list = ast.literal_eval(pos_list)
        except (ValueError, SyntaxError) as e:
            print(f"Error parsing POS list: {pos_list}. Error: {e}")
            return False
    return all(isinstance(item, tuple) and len(item) == 2 for item in pos_list)

if 'original_pos' not in df.columns or df['original_pos'].isna().any():
    df['original_pos'] = df.progress_apply(lambda row: pos_tagging(row['text'], row.name), axis=1)
    df['original_valid'] = df['original_pos'].progress_apply(verify_pos_structure)
    df = df[df['original_valid']].drop(columns=['original_valid'])
    df.dropna(subset=['original_pos'], inplace=True)
    df.to_csv('intermediate_results.csv', index=False)
    print("Saved intermediate results after 'original_pos' tagging.")

if 'corrected_pos' not in df.columns or df['corrected_pos'].isna().any():
    df['corrected_pos'] = df.progress_apply(lambda row: pos_tagging(row['correct'], row.name), axis=1)
    df['corrected_valid'] = df['corrected_pos'].progress_apply(verify_pos_structure)
    df = df[df['corrected_valid']].drop(columns=['corrected_valid'])
    df.dropna(subset=['corrected_pos'], inplace=True)
    df.to_csv('intermediate_results.csv', index=False)
    print("Saved intermediate results after 'corrected_pos' tagging.")

def classify_errors_from_diff(original_pos, corrected_pos):
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

try:
    if 'classified_errors' not in df.columns or df['classified_errors'].isna().any():
        df['original_pos'] = df['original_pos'].apply(eval)
        df['corrected_pos'] = df['corrected_pos'].apply(eval)
        df['classified_errors'] = df.apply(lambda row: classify_errors_from_diff(row['original_pos'], row['corrected_pos']), axis=1)
        df.to_csv('intermediate_results.csv', index=False)
        print("Saved intermediate results after error detection.")
except Exception as e:
    print(f"Error during error detection: {e}")
    raise


import ast

known_error_types = ['PRON Error', 'SYM Error', 'VERB Error', 'ADP Error', 'PART Error',
                      'AUX Error', 'SCONJ Error', 'NUM Error', 'SPACE Error', 'ADV Error',
                        'NOUN Error', 'INTJ Error', 'PUNCT Error', 'CCONJ Error',
                          'DET Error', 'ADJ Error', 'PROPN Error']

df['classified_errors'] = df['classified_errors'].apply(ast.literal_eval)


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


try:
    if 'error_counts' not in df.columns or df['error_counts'].isna().any():
        df['error_counts'] = df['classified_errors'].progress_apply(count_error_types)
        
        df['error_counts'] = df['error_counts'].apply(dict)
        
        df.to_csv('intermediate_results.csv', index=False)
        print("Saved intermediate results after error counting.")
except Exception as e:
    print(f"Error during error counting: {e}")
    raise


try:
    df[['text', 'correct', 'classified_errors', 'error_counts']].to_csv('error_analysis_big_dataset_clean.csv', index=False)
    print("Final results saved to 'error_analysis_big_dataset_clean.csv'")
except Exception as e:
    print(f"Error saving final CSV: {e}")
    raise

try:
    df_cleaned = pd.read_csv('error_analysis_big_dataset_clean.csv')
except Exception as e:
    print(f"Error loading cleaned dataset: {e}")
    raise

try:
    df = pd.read_csv('error_analysis_big_dataset_clean.csv')

    def convert_to_counter_dict(counter_str):
        try:
            return eval(counter_str)
        except Exception as e:
            print(f"Error converting string to dict: {e}")
            return {}

    df['error_counts'] = df['error_counts'].apply(convert_to_counter_dict)

    error_types_df = pd.DataFrame(df['error_counts'].tolist()).fillna(0)

    error_types_df = error_types_df.applymap(lambda x: int(x) if isinstance(x, (int, float)) else 0)

    error_types_relative_df = error_types_df.div(error_types_df.sum(axis=1), axis=0)

    error_types_relative_df.to_pickle('error_types_relative_frequencies_big_dataset.pkl')
    error_types_relative_df.to_csv('error_types_relative_frequencies_big_dataset.csv')

except Exception as e:
    print(f"Error during error counts analysis: {e}")
    raise
