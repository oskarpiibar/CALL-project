from langdetect import detect, DetectorFactory
from langdetect.lang_detect_exception import LangDetectException
import re
import ftfy
import pandas as pd
from tqdm import tqdm
from nltk.stem import WordNetLemmatizer
import nltk
nltk.download('words')
nltk.download('omw-1.4')
nltk.download('wordnet')

lemmatizer = WordNetLemmatizer()

import nltk
from nltk.corpus import words

english_words = set(words.words())

DetectorFactory.seed = 0

# Load the dataset
df_og = pd.read_csv('processed_dataset.csv')
df = df_og.copy()
print(df.head(20))

# Function to remove symbols
def remove_symbols(text):
    return re.sub(r'[^A-Za-z0-9\s.,!?\'"()\-]', '', text)

# Function to retain rows with at least 30% English content and remove non-English words
def filter_non_english_rows(text):
    words_list = text.split()
    english_only_words = []
    for word in words_list:
        # Strip punctuation from beginning and end, handle possessives like "life's"
        clean_word = word.strip(".,!?\"'()[]{}").lower()
        
        # Lemmatize to handle plurals, e.g., "stories" -> "story"
        base_word = lemmatizer.lemmatize(clean_word)
        
        # Check if base or possessive form is in dictionary
        if (base_word in english_words or
            clean_word.rstrip("'s") in english_words or
            clean_word in english_words):
            english_only_words.append(word)  # Keep original word with punctuation

    # Calculate English percentage and determine if row is mostly English
    if len(words_list) == 0:
        return ''  # Return empty string if text is empty
    english_percentage = len(english_only_words) / len(words_list)
    if english_percentage >= 0.3:
        return ' '.join(english_only_words)  # Return only English words
    else:
        return None  # Mark row for removal if below 30% English content



tqdm.pandas(desc="Processing dataset")

df['text'] = df['text'].progress_apply(ftfy.fix_text)
df['text'] = df['text'].progress_apply(remove_symbols)

df['text'] = df['text'].progress_apply(filter_non_english_rows)
df = df.dropna(subset=['text']).reset_index(drop=True)

# Display final DataFrame
print(df.head(20))

df.to_csv('processed_dataset3.csv', index=False)


