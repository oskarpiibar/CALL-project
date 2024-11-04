# import pandas as pd
# from textblob import TextBlob
# from tqdm import tqdm

# # Load the CSV file into a DataFrame
# df = pd.read_csv('test_set.csv')

# # Add a progress bar using tqdm
# tqdm.pandas(desc="Correcting Text")

# # Apply TextBlob's correction in a more efficient way
# df['correct'] = df['incorrect'].progress_apply(lambda x: str(TextBlob(x).correct()))

# # Save the corrected DataFrame to a new CSV file
# df.to_csv('textblob_corrected.csv', index=False)

# print(df.head())

# Happytransformer (T5)
import pandas as pd
from happytransformer import HappyTextToText, TTSettings
from tqdm import tqdm
import re

df = pd.read_csv('test_set.csv')

happy_tt = HappyTextToText("T5", "vennify/t5-base-grammar-correction")

args = TTSettings(num_beams=5, min_length=1)

def correct_grammar(text):
    sentences = re.split(r'(?<=[.!?]) +', text)
    corrected_sentences = []
    
    for sentence in sentences:
        result = happy_tt.generate_text(f"grammar: {sentence}", args=args)
        corrected_sentences.append(result.text)  # Collect all corrected sentences
    
    return ' '.join(corrected_sentences)

tqdm.pandas()

df['correct_text'] = df['incorrect'].progress_apply(correct_grammar)

df.to_csv('test_corrected_test.csv', index=False)

print(df.head())


# os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# df = pd.read_csv('test_set.csv')

# grammar_correction_model = pipeline(task="text2text-generation", model="hassaanik/grammar-correction-model")

# Good for grammar correction. Used in various grammar correction tasks
# grammar_correction_model = pipeline("text2text-generation", model="prithivida/grammar_error_correcter_v1", from_pt=True)

# Grammar correction in noisy transcriptions 
# grammar_correction_model = pipeline("text2text-generation", model="flexudy/t5-small-wav2vec2-grammar-fixer")

# Fine tuned for grammar correction
# grammar_correction_model = pipeline("fill-mask", model="bert-base-cased-finetuned-grammar-check")

# Initial grammar correction tool

# import pandas as pd
# from transformers import pipeline
# from tqdm import tqdm

# # Specifically trained for grammar correction
# grammar_correction_model = pipeline(task = "text2text-generation", model="vennify/t5-base-grammar-correction")

# tqdm.pandas()

# df['correct_text'] = df['incorrect'].progress_apply(lambda x: grammar_correction_model(x, return_tensors=False)[0]['generated_text'])

# # Save the corrected dataset to a new CSV file
# df.to_csv('corrected_test.csv', index=False)

# print(df.head())