import pandas as pd
from transformers import pipeline
from tqdm import tqdm
import os

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Load the CSV file into a DataFrame
df = pd.read_csv('preprocessed_dataset.csv')

df = df.head(100)

# Load the correction model from Hugging Face
grammar_correction_model = pipeline(task="text2text-generation", model="hassaanik/grammar-correction-model")

# Apply text correction in batches
batch_size = 10  # Adjust the batch size depending on your system's capacity
tqdm.pandas()

def batch_process(texts):
    corrected_texts = grammar_correction_model(texts, max_new_tokens=50)
    return [text['generated_text'] for text in corrected_texts]

# Apply the correction in batches
df['correct_text'] = df['text'].progress_apply(lambda x: grammar_correction_model([x], max_new_tokens=50)[0]['generated_text'])

# Save the corrected dataset to a new CSV file
df.to_csv('corrected_test.csv', index=False)

print(df.head())
