import difflib
import pandas as pd
from tqdm import tqdm
import sys
sys.setrecursionlimit(3000)

tqdm.pandas()


print("running")

# Function to classify errors based on the diff
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

# Load your CSV file
df = pd.read_csv('intermediate_results.csv')
print ("file read")
# Convert string representations of POS tags back to list of tuples
df['original_pos'] = df['original_pos'].apply(eval)
print("eval thingy")
df['corrected_pos'] = df['corrected_pos'].apply(eval)
df['classified_errors'] = df.progress_apply(lambda row: classify_errors_from_diff(row['original_pos'], row['corrected_pos']), axis=1)

# Save the result to a new CSV file
df.to_csv('classified_error_results.csv', index=False)

print("Error classification completed and saved to 'classified_error_results.csv'.")



