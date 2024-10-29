from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pandas as pd
from tqdm import tqdm

# Load the ground truth DataFrame (ground.csv) and the DataFrame to classify (to_classify.csv)
ground_df = pd.read_csv('ground.csv')
to_classify_df = pd.read_csv('to_classify.csv')

# Extract only the numeric metric columns from ground_df for similarity calculation
# Assuming ground_df has columns 'error_beginning_pct', 'error_middle_pct', 'error_end_pct', and 'native_language'
metric_columns = ['error_beginning_pct', 'error_middle_pct', 'error_end_pct']
ground_df_features = ground_df[metric_columns]
ground_labels = ground_df['native']  # Column containing the language labels

# Initialize a counter for predictions below the confidence threshold
inconclusive_count = 0

# Function to classify new text based on cosine similarity with a confidence threshold
def classify_error_position(new_position_profile, ground_df_features, ground_labels, confidence_threshold=0.95):
    global inconclusive_count
    
    # Convert the new position profile to a DataFrame with the same structure as ground_df_features
    new_position_vector = pd.DataFrame(new_position_profile, index=[0]).reindex(columns=ground_df_features.columns, fill_value=0)
    
    # Fill any NaN values in new_position_vector or ground_df_features with 0
    new_position_vector = new_position_vector.fillna(0)
    ground_df_features = ground_df_features.fillna(0)

    # Calculate cosine similarity between the new position profile and all profiles in ground_df_features
    similarities = cosine_similarity(new_position_vector, ground_df_features)
    
    # Get the highest similarity score
    max_similarity = np.max(similarities)
    
    # If the highest similarity score is below the threshold, return "Inconclusive"
    if max_similarity < confidence_threshold:
        inconclusive_count += 1
        return "Inconclusive"
    
    # Otherwise, return the language corresponding to the most similar profile
    most_similar_index = np.argmax(similarities)
    return ground_labels.iloc[most_similar_index]

# Function to iterate through the dataframe to classify, comparing each row to ground_df_features
def classify_texts(df, ground_df_features, ground_labels, confidence_threshold=0.90):
    results = []

    # Add tqdm progress bar for the iteration
    for index, row in tqdm(df.iterrows(), total=len(df), desc="Classifying texts"):
        # Extract position metrics for this row
        new_position_profile = row[metric_columns].to_dict()

        # Classify the text using the error position metrics
        predicted_language = classify_error_position(new_position_profile, ground_df_features, ground_labels, confidence_threshold)

        # Append the result including text, correct version, and the predicted match
        results.append({
            'text': row['text'],
            'correct': row['correct'],
            'predicted_native_language': predicted_language
        })

    # Convert the results to a DataFrame for easier handling
    return pd.DataFrame(results)

# Classify each row in to_classify.csv and return a DataFrame with predictions
classified_texts_df = classify_texts(to_classify_df, ground_df_features, ground_labels)

# Save the classification results to a CSV file
classified_texts_df.to_csv('classified_texts_by_error_position.csv', index=False)
print(f"Classification complete. Results saved to 'classified_texts_by_error_position.csv'")

# Define the list of target languages for filtering (assume they were pre-identified in ground_df)
target_languages = ['Russian', 'Chinese (Mandarin)', 'Spanish']

# Save separate files for each target language, based on the matched texts
for language in target_languages:
    # Filter the DataFrame to include only rows where predicted_native_language matches the language
    language_df = classified_texts_df[classified_texts_df['predicted_native_language'] == language]
    
    # If there are rows for this language, save them to a CSV
    if not language_df.empty:
        filename = f'classified_texts_{language}.csv'
        language_df.to_csv(filename, index=False)
        print(f"Saved {len(language_df)} rows for {language} to {filename}")

# Print the number of inconclusive predictions
print(f"Number of inconclusive predictions (below 80% confidence): {inconclusive_count}")
