import spacy
import pandas as pd

# Load SpaCy English model
nlp = spacy.load("en_core_web_sm")

# Load your dataset
df = pd.read_csv('top_3_with_corrections.csv')  # Replace with your actual data file

# Function to split text into sentences and apply POS tagging
def pos_tagging(text):
    doc = nlp(text)
    return [(token.text, token.pos_) for sent in doc.sents for token in sent]

# Apply POS tagging on original and corrected text
df['original_pos'] = df['text'].apply(pos_tagging)
df['corrected_pos'] = df['correct'].apply(pos_tagging)

# Example function to compare POS tags
def compare_pos(original, corrected):
    # Compare token by token for error detection
    errors = []
    for (o_token, o_pos), (c_token, c_pos) in zip(original, corrected):
        if o_pos != c_pos:
            errors.append((o_token, o_pos, c_token, c_pos))  # Track original vs corrected
    return errors

# Apply error detection and group by native language
df['errors'] = df.apply(lambda row: compare_pos(row['original_pos'], row['corrected_pos']), axis=1)

# Group by native language
errors_by_language = df.groupby('native')['errors'].apply(list)

# Output or save results for further analysis
print(errors_by_language)

def classify_errors(original_pos, corrected_pos):
    errors = []
    for (o_token, o_pos), (c_token, c_pos) in zip(original_pos, corrected_pos):
        if o_pos != c_pos:
            # Check the main categories of POS tags for classification
            if o_pos.startswith('VERB') or c_pos.startswith('VERB'):
                errors.append(('Verb Error', o_token, o_pos, c_token, c_pos))
            elif o_pos.startswith('NOUN') or c_pos.startswith('NOUN'):
                errors.append(('Noun Error', o_token, o_pos, c_token, c_pos))
            elif o_pos.startswith('ADJ') or c_pos.startswith('ADJ'):
                errors.append(('Adjective Error', o_token, o_pos, c_token, c_pos))
            elif o_pos.startswith('ADP') or c_pos.startswith('ADP'):
                errors.append(('Preposition Error', o_token, o_pos, c_token, c_pos))
            elif o_pos.startswith('PRON') or c_pos.startswith('PRON'):
                errors.append(('Pronoun Error', o_token, o_pos, c_token, c_pos))
            else:
                errors.append(('Other Error', o_token, o_pos, c_token, c_pos))
    return errors


# Apply classification
df['classified_errors'] = df.apply(lambda row: classify_errors(row['original_pos'], row['corrected_pos']), axis=1)

# Group by native language and classified errors
classified_errors_by_language = df.groupby('native')['classified_errors'].apply(list)

# Display the result
print(classified_errors_by_language)

from collections import Counter
import pandas as pd

# List of known error types
known_error_types = [
    'Pronoun Error', 'Verb Error', 'Noun Error', 'Adjective Error', 'Preposition Error', 'Other Error'
]

# Function to count errors for each native language by error type
def count_error_types(classified_errors):
    error_counter = Counter()
    for error_list in classified_errors:
        for error in error_list:
            # Check if the error matches any of the known error types
            for error_type in known_error_types:
                if error_type in error:  # If the known error type is part of the string, count it
                    error_counter[error_type] += 1
                    break  # Exit after matching the first error type
            else:
                # If no known error type is found, you can print for debugging or handle it differently
                print(f"Unknown error structure: {error}")
    return error_counter


# Apply counting for each native languageerror_pattern_df = pd.DataFrame(list(error_pattern_by_language.items()), columns=['native', 'error_counts'])

df['error_counts'] = df['classified_errors'].apply(count_error_types)

# Example of aggregating counts for all languages
error_pattern_by_language = df.groupby('native')['error_counts'].apply(lambda x: sum(x, Counter()))

# Convert to DataFrame for easier manipulation
error_pattern_df = pd.DataFrame(list(error_pattern_by_language.items()), columns=['native', 'error_counts'])

# Display the error patterns for analysis
# Safely apply the dict conversion by handling NaN or float values
error_pattern_df['error_counts'] = error_pattern_df['error_counts'].apply(lambda x: dict(x) if isinstance(x, Counter) else {})
print(error_pattern_df)

# Create a DataFrame that contains the native language, classified errors, and error counts
analysis_df = df[['native', 'classified_errors', 'error_counts']]

analysis_df.to_csv('error_analysis_by_language_unclean.csv', index=False)
df = pd.read_csv('error_analysis_by_language_unclean.csv')

df['native'] = df['native'].str.split('<br/>')
df = df.explode('native')

# Optional: Reset the index if you need a clean DataFrame
df.reset_index(drop=True, inplace=True)

# Remove rows with empty classified errors
df_cleaned = df[df['classified_errors'].apply(lambda x: len(eval(x)) > 0)]

# Group by the native language and aggregate classified_errors and error_counts
df_grouped = df_cleaned.groupby('native').agg({
    'classified_errors': lambda x: sum([eval(item) for item in x], []),  # Combine all classified errors
    'error_counts': lambda x: sum([Counter(eval(item)) for item in x], Counter())  # Sum the error counts
}).reset_index()

# Convert Counter back to a dictionary for error_counts
df_grouped['error_counts'] = df_grouped['error_counts'].apply(dict)

# Save the grouped dataframe
df_grouped.to_csv('error_analysis_by_language_clean.csv', index=False)


