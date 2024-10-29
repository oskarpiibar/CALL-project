from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pandas as pd
from tqdm import tqdm

# Load the relative frequencies DataFrame
error_types_relative_df = pd.read_pickle('error_types_relative_frequencies.pkl')

# Initialize a counter for predictions below the confidence threshold
inconclusive_count = 0

# Function to classify new text based on cosine similarity with a 95% confidence threshold
def classify_native_language(new_text_profile, error_types_relative_df, confidence_threshold=0.85):
    global inconclusive_count  # Use global to modify the counter variable
    
    # Convert the new text profile to a DataFrame with the same structure as error_types_relative_df
    new_text_vector = pd.DataFrame(new_text_profile, index=[0]).reindex(columns=error_types_relative_df.columns, fill_value=0)
    
    # Fill any NaN values in new_text_vector or error_types_relative_df with 0
    new_text_vector = new_text_vector.fillna(0)
    error_types_relative_df = error_types_relative_df.fillna(0)

    # Calculate cosine similarity between the new text profile and all native language profiles
    similarities = cosine_similarity(new_text_vector, error_types_relative_df)
    
    # Get the highest similarity score
    max_similarity = np.max(similarities)
    
    # If the highest similarity score is below the threshold, return "Inconclusive"
    if max_similarity < confidence_threshold:
        inconclusive_count += 1  # Increment the counter for inconclusive predictions
        return "Inconclusive"
    
    # Otherwise, return the native language corresponding to the most similar profile
    most_similar_index = np.argmax(similarities)
    return error_types_relative_df.index[most_similar_index]

# Function to iterate through the dataframe with 'incorrect' and 'correct' text columns
def classify_texts(df, error_types_relative_df, confidence_threshold=0.85):
    results = []

    # Add tqdm progress bar for the iteration
    for index, row in tqdm(df.iterrows(), total=len(df), desc="Classifying texts"):
        # Drop 'incorrect' and 'correct' if they exist
        new_text_profile = row.drop(['text', 'correct'], errors='ignore').to_dict()

        # Classify the text using the relative error frequencies
        predicted_language = classify_native_language(new_text_profile, error_types_relative_df, confidence_threshold)

        # Append the result including incorrect, correct text, and predicted language
        results.append({
            'text': row.get('text', None),  # Use None if 'incorrect' is missing
            'correct': row.get('correct', None),      # Use None if 'correct' is missing
            'predicted_native_language': predicted_language
        })

    # Convert the results to a DataFrame for easier handling
    return pd.DataFrame(results)

# Assuming 'df_to_classify' is the DataFrame with 'incorrect' and 'correct' columns
# and the relative frequencies of errors.
df_to_classify = pd.read_pickle('error_types_relative_frequencies_with_text.pkl')  # Replace with your actual file

# Classify each row and return a DataFrame with predictions
classified_texts_df = classify_texts(df_to_classify, error_types_relative_df)

# List of target languages for filtering
target_languages = ['Spanish', 'Russian', 'Chinese (Mandarin)']

# Iterate through the target languages and save each one to a separate CSV file
for language in target_languages:
    # Filter the DataFrame to include only rows where the predicted language matches the current target language
    language_df = classified_texts_df[classified_texts_df['predicted_native_language'] == language]
    
    # If there are rows for this language, save them to a CSV
    if not language_df.empty:
        filename = f'classified_texts_{language}.csv'
        language_df.to_csv(filename, index=False)
        print(f"Saved {len(language_df)} rows for {language} to {filename}")

# Print the number of inconclusive predictions
print(f"Number of inconclusive predictions (below 95% confidence): {inconclusive_count}")