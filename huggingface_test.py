from transformers import pipeline
import pandas as pd
from tqdm import tqdm

tqdm.pandas()

df = pd.read_csv('test_set.csv')

# Load the Grammar Correction T5 Model from Hugging Face
grammar_correction_model = pipeline(task="text2text-generation", model="hassaanik/grammar-correction-model")

df['correct_text'] = df['incorrect'].progress_apply(lambda x: grammar_correction_model(x, return_tensors=False)[0]['generated_text'])

df.to_csv('test_set.csv', index=False)

print(df.head())


# # Input text with grammatical errors
# input_text = "They is going to spent time together."
# # Get corrected output and details
# result = grammar_correction_model(input_text, max_length=200, num_beams=5, no_repeat_ngram_size=2)
# # Print the corrected output
# print(result)