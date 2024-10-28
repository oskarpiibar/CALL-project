from langdetect import detect, DetectorFactory
from langdetect.lang_detect_exception import LangDetectException
import re
import ftfy
import pandas as pd
from tqdm import tqdm
from nltk import tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Set seed for langdetect to ensure consistency in results
DetectorFactory.seed = 0

# Initialize lemmatizer and stopwords
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# Load the dataset
df_og = pd.read_csv('processed_dataset.csv')
df = df_og.copy()


# Function to remove symbols
def remove_symbols(text):
    return re.sub(r'[^A-Za-z0-9\s.,:;!?\'"/\-]', '', text)

def add_remove_space_near_punctuation(text):
    text = re.sub(r'\s+([.!?])', r'\1', text)  # Remove space before punctuation
    text = re.sub(r'([.!?])([a-zA-Z])', r'\1 \2', text)  # Add space after punctuation
    return text

def remove_short_sentences(text):
    if not isinstance(text, str) or text is None:
        return ''
    sentences = tokenize.sent_tokenize(text)
    new_text = ' '.join([sentence for sentence in sentences if len(sentence.split()) >= 3])
    return new_text.strip()

# Custom function to detect if a word contains non-English characters
def contains_non_english_chars(word):
    return bool(re.search(r'[^\x00-\x7F]', word))  # Check for non-ASCII characters

# Function to lemmatize and remove stopwords
def lemmatize_and_remove_stopwords(text):
    words = re.split(r'\s+', text.lower())  # Lowercase and split
    cleaned_words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words and len(word) > 2]
    return ' '.join(cleaned_words)

# Function to filter non-English sentences and apply lemmatization
def filter_non_english_sentences(text):
    if not isinstance(text, str) or text is None:
        return None

    sentences = tokenize.sent_tokenize(text)
    valid_sentences = []

    for sentence in sentences:
        try:
            # First check if the sentence is mostly English
            if detect(sentence) != 'en':
                continue  # Skip if the whole sentence is not English
        except LangDetectException:
            continue  # Skip if language detection fails

        words_list = re.split(r'\s+', sentence)
        sentence_is_english = True
        cleaned_words = []

        # Perform a word-level check
        for word in words_list:
            clean_word = word.strip(".;:%,!?\"'()[]{}")

            # Lemmatize and check if the word contains non-English characters
            lemmatized_word = lemmatizer.lemmatize(clean_word.lower())
            if contains_non_english_chars(lemmatized_word) or :
                sentence_is_english = False
                break  # Discard the entire sentence if a non-English word is found
            cleaned_words.append(word)

        if sentence_is_english:
            valid_sentences.append(' '.join(cleaned_words))

    return ' '.join(valid_sentences) if valid_sentences else None

# Process the dataset
tqdm.pandas(desc="Processing dataset")

# Ensure all text fields are strings before applying text processing
df['text'] = df['text'].astype(str)

# Apply text cleaning and filtering
df['text'] = df['text'].progress_apply(ftfy.fix_text)
df['text'] = df['text'].progress_apply(remove_symbols)
df['text'] = df['text'].progress_apply(add_remove_space_near_punctuation)
df['text'] = df['text'].progress_apply(lemmatize_and_remove_stopwords)
df['text'] = df['text'].progress_apply(filter_non_english_sentences)
df['text'] = df['text'].progress_apply(remove_short_sentences)

# Replace empty strings with NaN and then drop rows with NaN in 'text' column
df['text'].replace('', pd.NA, inplace=True)
df = df.dropna(subset=['text']).reset_index(drop=True)

# Display final DataFrame
print("Processed DataFrame (First 100 rows):")
print(df.head(20))

# Save the processed DataFrame to a CSV for checking
df.to_csv('processed_dataset3.csv', index=False)
