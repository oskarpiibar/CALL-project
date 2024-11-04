from langdetect import detect, DetectorFactory 
from langdetect.lang_detect_exception import LangDetectException
import re
import ftfy
import pandas as pd
from tqdm import tqdm
from nltk.stem import WordNetLemmatizer
from nltk import tokenize
import nltk
from nltk.corpus import words, names
import geonamescache  

nltk.download('words')
nltk.download('names')
nltk.download('punkt_tab')
nltk.download('omw-1.4')
nltk.download('wordnet')

lemmatizer = WordNetLemmatizer()

english_words = set(words.words())

male_names = set(names.words('male.txt'))
female_names = set(names.words('female.txt'))
all_names = male_names.union(female_names)
english_words.update(name.lower() for name in all_names)

gc = geonamescache.GeonamesCache()
cities = set()
for city_info in gc.get_cities().values():
    city_name = city_info['name'].lower()
    cities.add(city_name)
english_words.update(cities)

languages = {
    "estonian", "english", "spanish", "french", "german", "chinese", "arabic", "russian",
    "japanese", "korean", "portuguese", "italian", "dutch", "turkish", "hindi",
    "bengali", "urdu", "swahili", "hebrew", "greek", "latin", "persian", "tamil", "thai"
}
english_words.update(languages)

nationalities = {
    "american", "british", "canadian", "french", "german", "italian", "spanish",
    "russian", "chinese", "japanese", "korean", "indian", "brazilian", "mexican",
    "nigerian", "australian", "swedish", "dutch", "turkish", "saudi", "argentinian",
    "egyptian", "south african", "polish", "pakistani", "bangladeshi", "filipino"
}
english_words.update(nationalities)

DetectorFactory.seed = 0

df_og = pd.read_csv('dataset_cleand-1-1.csv')
df = df_og.copy()
print("Original DataFrame:")
print(df.head(20))

def remove_symbols(text):
    return re.sub(r'[^A-Za-z0-9\s.,:;!?\'"/\-]', '', text)

def add_remove_space_near_puncuation(text):
    text = re.sub(r'\s+([.!?])', r'\1', text) 
    text = re.sub(r'([.!?])([a-zA-Z])', r'\1 \2', text) 
    return text

def remove_short_senteces(text):
    if not isinstance(text, str) or text is None:
        return ''
    sentences = tokenize.sent_tokenize(text)
    new_text = ''
    for sentence in sentences:
        if len(sentence.split()) >= 3:
            new_text += sentence + ' '
    return new_text.strip()

def filter_non_english_rows(text):
    words_list = re.split(r'\s+|/', text)
    english_only_words = []
    for word in words_list:
        original_word = word.strip(".;:%,!?\"'()[]{}")
        clean_word = original_word.lower()
        
        base_word = lemmatizer.lemmatize(clean_word)
        
        if (base_word in english_words or
            clean_word.rstrip("'s") in english_words or
            clean_word in english_words or
            original_word.istitle() or
            "'" in original_word): 
            english_only_words.append(word)  

    if len(words_list) == 0:
        return ''  
    english_percentage = len(english_only_words) / len(words_list)
    if english_percentage >= 0.3:
        return ' '.join(english_only_words)  
    else:
        return None  
    

def filter_non_english_sentences(text):
    sentences = re.split(r'(?<=[.!?]) +', text)
    english_sentences = []
    for sentence in sentences:
        try:
            if detect(sentence) == 'en':  
                english_sentences.append(sentence)
        except LangDetectException:
            pass
    return ' '.join(english_sentences)

tqdm.pandas(desc="Processing dataset")

df['text'] = df['text'].progress_apply(ftfy.fix_text)
df['text'] = df['text'].progress_apply(remove_symbols)
df['text'] = df['text'].progress_apply(add_remove_space_near_puncuation)
df['text'] = df['text'].progress_apply(filter_non_english_sentences)
df['text'] = df['text'].progress_apply(remove_short_senteces)
df = df.dropna(subset=['text']).reset_index(drop=True)

df.to_csv('processed_dataset3.csv', index=False)
