import spacy
import pandas as pd
from collections import Counter
from tqdm import tqdm
import difflib

# Load SpaCy English model
nlp = spacy.load("en_core_web_sm")

# Load dataset with encoding check
try:
    df = pd.read_csv('big_dataset.csv', encoding='utf-8')
    print("Dataset loaded successfully with UTF-8 encoding.")
except UnicodeDecodeError:
    df = pd.read_csv('big_dataset.csv', encoding='ISO-8859-1')
    print("Dataset loaded with ISO-8859-1 encoding due to UTF-8 decoding error.")

# Drop rows where the 'correct' column has NaN values or is empty/whitespace
df.dropna(subset=['correct'], inplace=True)
df = df[df['correct'].str.strip() != ""]

# Verify no NaN or empty values remain in the 'correct' column
if df['correct'].isna().any() or (df['correct'].str.strip() == "").any():
    raise ValueError("NaN or empty values found in 'correct' column after cleaning.")

# Add tqdm progress_apply to monitor progress
tqdm.pandas()

# Function to split text into tokens and apply POS tagging
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

# Load intermediate results if available, with verification
try:
    df = pd.read_csv('intermediate_results.csv', encoding='utf-8')
    print("Loaded intermediate results from 'intermediate_results.csv'")
except FileNotFoundError:
    print("No intermediate results found, starting from scratch.")

# Function to verify POS structure
def verify_pos_structure(pos_list):
    if isinstance(pos_list, str):
        try:
            pos_list = ast.literal_eval(pos_list)
        except (ValueError, SyntaxError) as e:
            print(f"Error parsing POS list: {pos_list}. Error: {e}")
            return False
    return all(isinstance(item, tuple) and len(item) == 2 for item in pos_list)

# Apply POS tagging and structure validation on 'original_pos'
if 'original_pos' not in df.columns or df['original_pos'].isna().any():
    df['original_pos'] = df.progress_apply(lambda row: pos_tagging(row['text'], row.name), axis=1)
    df['original_valid'] = df['original_pos'].progress_apply(verify_pos_structure)
    df = df[df['original_valid']].drop(columns=['original_valid'])
    df.dropna(subset=['original_pos'], inplace=True)
    df.to_csv('intermediate_results.csv', index=False)
    print("Saved intermediate results after 'original_pos' tagging.")

# Apply POS tagging and structure validation on 'corrected_pos'
if 'corrected_pos' not in df.columns or df['corrected_pos'].isna().any():
    df['corrected_pos'] = df.progress_apply(lambda row: pos_tagging(row['correct'], row.name), axis=1)
    df['corrected_valid'] = df['corrected_pos'].progress_apply(verify_pos_structure)
    df = df[df['corrected_valid']].drop(columns=['corrected_valid'])
    df.dropna(subset=['corrected_pos'], inplace=True)
    df.to_csv('intermediate_results.csv', index=False)
    print("Saved intermediate results after 'corrected_pos' tagging.")

# Function to compare POS tags, accounting for insertions/deletions
def classify_errors_from_diff(original_pos, corrected_pos):
    classified_errors = []
    
    # Tokenize words from the original and corrected POS lists
    original_tokens = [token for token, pos in original_pos]
    corrected_tokens = [token for token, pos in corrected_pos]
    
    # Get the diff of tokens
    diff = list(difflib.ndiff(original_tokens, corrected_tokens))
    
    original_index, corrected_index = 0, 0
    
    # Iterate through diff and classify errors
    i = 0
    while i < len(diff):
        if diff[i][0] == '-' and i + 1 < len(diff) and diff[i + 1][0] == '+':
            # Substitution case
            if original_index < len(original_pos) and corrected_index < len(corrected_pos):
                original_word, original_pos_tag = original_pos[original_index]
                corrected_word, corrected_pos_tag = corrected_pos[corrected_index]
                error_type = f"{corrected_pos_tag} Error"
                classified_errors.append((error_type, original_word, original_pos_tag, corrected_word, corrected_pos_tag))
                original_index += 1
                corrected_index += 1
            i += 2  # Skip the next item since it’s part of the substitution
        elif diff[i][0] == '-' and original_index < len(original_pos):
            # Deletion case
            original_word, original_pos_tag = original_pos[original_index]
            error_type = f"{original_pos_tag} Error"
            classified_errors.append((error_type, original_word, original_pos_tag, 'DELETED'))
            original_index += 1
            i += 1
        elif diff[i][0] == '+' and corrected_index < len(corrected_pos):
            # Addition case
            corrected_word, corrected_pos_tag = corrected_pos[corrected_index]
            error_type = f"{corrected_pos_tag} Error"
            classified_errors.append((error_type, 'ADDED', corrected_pos_tag, corrected_word))
            corrected_index += 1
            i += 1
        else:
            # No change
            original_index += 1
            corrected_index += 1
            i += 1
    
    return classified_errors

# Apply error detection
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

# List of known error types
known_error_types = ['PRON Error', 'SYM Error', 'VERB Error', 'ADP Error', 'PART Error',
                      'AUX Error', 'SCONJ Error', 'NUM Error', 'SPACE Error', 'ADV Error',
                        'NOUN Error', 'INTJ Error', 'PUNCT Error', 'CCONJ Error',
                          'DET Error', 'ADJ Error', 'PROPN Error']

df['classified_errors'] = df['classified_errors'].apply(ast.literal_eval)


def count_error_types(classified_errors):
    error_counter = Counter()
    
    # Loop through each tuple in classified_errors
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


# Apply counting function to each row in classified_errors column
try:
    if 'error_counts' not in df.columns or df['error_counts'].isna().any():
        df['error_counts'] = df['classified_errors'].progress_apply(count_error_types)
        
        # Convert Counter to dictionary for saving
        df['error_counts'] = df['error_counts'].apply(dict)
        
        # Save results
        df.to_csv('intermediate_results.csv', index=False)
        print("Saved intermediate results after error counting.")
except Exception as e:
    print(f"Error during error counting: {e}")
    raise


# Final save for analysis
try:
    df[['text', 'correct', 'classified_errors', 'error_counts']].to_csv('error_analysis_big_dataset_clean.csv', index=False)
    print("Final results saved to 'error_analysis_big_dataset_clean.csv'")
except Exception as e:
    print(f"Error saving final CSV: {e}")
    raise

# Optional: Load the cleaned dataset again if necessary
try:
    df_cleaned = pd.read_csv('error_analysis_big_dataset_clean.csv')
except Exception as e:
    print(f"Error loading cleaned dataset: {e}")
    raise

# Error counts analysis
try:
    # Load the CSV file (without the 'native' column)
    df = pd.read_csv('error_analysis_big_dataset_clean.csv')

    # Function to safely convert the error counts from string to dictionary
    def convert_to_counter_dict(counter_str):
        try:
            return eval(counter_str)
        except Exception as e:
            print(f"Error converting string to dict: {e}")
            return {}

    # Apply the conversion to ensure error counts are dicts
    df['error_counts'] = df['error_counts'].apply(convert_to_counter_dict)

    # Convert the error counts into a DataFrame for easier plotting
    error_types_df = pd.DataFrame(df['error_counts'].tolist()).fillna(0)

    # Normalize data to ensure consistent data types
    error_types_df = error_types_df.applymap(lambda x: int(x) if isinstance(x, (int, float)) else 0)

    # Calculate relative frequencies by dividing each error count by the total number of errors for that row
    error_types_relative_df = error_types_df.div(error_types_df.sum(axis=1), axis=0)

    # Save the relative frequencies DataFrame to a pickle file
    error_types_relative_df.to_pickle('error_types_relative_frequencies_big_dataset.pkl')
    error_types_relative_df.to_csv('error_types_relative_frequencies_big_dataset.csv')

except Exception as e:
    print(f"Error during error counts analysis: {e}")
    raise
