import pandas as pd
import os

def combine_csv_by_language(folder_paths):
    combined_data = {
        'Chinese (Mandarin)': [],
        'Russian': [],
        'Spanish': []
    }

    # Read and combine CSV files from each folder
    for folder in folder_paths:
        for filename in os.listdir(folder):
            if filename.endswith('.csv'):
                language = filename.split('_')[-1].replace('.csv', '')  # Extract language from filename
                file_path = os.path.join(folder, filename)
                df = pd.read_csv(file_path)

                # Append DataFrame to the corresponding language list
                if language in combined_data:
                    combined_data[language].append(df)

    # Combine DataFrames for each language and remove duplicates
    combined_dfs = {}
    for language, dataframes in combined_data.items():
        if dataframes:  # Check if there are any DataFrames for the language
            combined_dfs[language] = pd.concat(dataframes, ignore_index=True).drop_duplicates()

    return combined_dfs

# Specify folder paths (relative to the location of this script)
folder_paths = [
    '../metric_error_position',  # Up one level to access the folders
    '../metric_rel_frequency',
    '../POS_sequences',
]

# Combine CSV files from the specified folders by language
final_combined_dfs = combine_csv_by_language(folder_paths)

# Save the combined DataFrames to new CSV files
for language, combined_df in final_combined_dfs.items():
    output_file_path = f'combined_classified_texts_{language}.csv'
    combined_df.to_csv(output_file_path, index=False)
    print(f'Combined CSV file for {language} saved as: {output_file_path}')
