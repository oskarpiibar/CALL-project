from langdetect import detect, detect_langs, DetectorFactory 
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
import fasttext
from huggingface_hub import hf_hub_download

nltk.download('words')
nltk.download('names')
nltk.download('punkt_tab')
nltk.download('omw-1.4')
nltk.download('wordnet')

DetectorFactory.seed = 0

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


def remove_symbols(text):
    return re.sub(r'[^A-Za-z0-9\s.,;:!?\'"()%\-]', '', text)


def handle_apostrophes(word):
    # 'm (I'm)
    # 't (don't, can't, shouldn't)
    # 're (they're)
    # 've (would've, could've, I've)
    # 's (he's, John's)
    # 'd (he'd, they'd)
    # 'll (he'll, she'll)
    # ' (Smiths', Jesus')
    en_apostrophe_endings = ['ve', 't', 're', 's', 'd', 'll', 'm']
    word = re.sub(r"'{2,}", "'", word)
    if "'" in word:
        splitted = word.split("'")
        if splitted[1] in en_apostrophe_endings or splitted[1] == '':
            return True
    return False


def filter_non_english_sentences(text):
    sentences = re.split(r'(?<=[.!?]) +', text)
    english_sentences = []
    for sentence in sentences:
        print(sentence)
        try:
            if detect(sentence) == 'en':  
                english_sentences.append(sentence)
        except LangDetectException:
            pass
    return ' '.join(english_sentences)


def additional_fasttext_language_detection(text):
    english_sentences = []
    sentences = re.split(r'(?<=[.!?]) +', text)
    for sentence in sentences:
        prediction = model.predict(sentence, k=2)
        predicted_language = prediction[0][0].split('__label__')[1]
        confidence_score = prediction[1][0]
        if predicted_language == 'eng_Latn' and confidence_score >= 0.99:
            english_sentences.append(sentence)

    return ' '.join(english_sentences)

def filter_non_english_rows(text):
    words_list = re.split(r'\s+|/', text)
    english_only_words = []
    for word in words_list:
        original_word = word.strip(".,%:;!?\"'()[]{}")
        clean_word = original_word.lower()
        
        base_word = lemmatizer.lemmatize(clean_word)
        
        if (base_word in english_words or
            clean_word.rstrip("'s") in english_words or
            clean_word in english_words or
            original_word.istitle() or
            str.isdigit(original_word) or
            "'" in original_word): 
            english_only_words.append(word)

    if len(words_list) == 0:
        return '' 
    english_percentage = len(english_only_words) / len(words_list)
    if english_percentage >= 0.3:
        return ' '.join(english_only_words)  
    else:
        return None  
    
def add_space_after_punctuation(text):
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

# string = "hi;hello there ?every life's a 50% : 50% movie. We got different stars and stories. We got different nights and mornings.Â°â€¢â˜†Â°â€¢â˜†Â°â€¢â˜†Â°â€¢â˜†Â°â€¢â˜†Â°â€¢â˜†Â°â€¢â˜†â€¢"
# string = "58 ans, shouldn't However this matter don't I'm okay de Metz dans France. Je pratique l'Espagnol, l'Anglais, me manque des gens Tous les les retour, j'aiderai la pratique Franais"
string = "tere mina olen oskar ja oskan lugeda Hello, today's a nice day and I'm enjoying the pretty weather. Hi today's a nice day in Barcelona. I'll be there in Estonia if you'd like me to visit the Smiths' family. Blabla veel eesti keelt."

new = add_space_after_punctuation(string)
new = ftfy.fix_text(new)
print(f'Fixing characters: {new}')
new2 = remove_symbols(new)
print(f'Symbols: {new2}')
new3 = remove_short_senteces(new2)
print(f'Short sentences: {new3}')
new4 = filter_non_english_sentences(new3)
print(f'English words: {new4}')

if 'matter' in english_words:
    print('yes')
else:
    print('no')

model_path = hf_hub_download(repo_id="facebook/fasttext-language-identification", filename="model.bin")
model = fasttext.load_model(model_path)
stringsss = "c'est tout :................. Prof  la retraite, je cherche  amliorer mon italien. I'd be more than happy to chat with you. Si quieres aprender Spanish or practicarlo just text me, don't be shy... Je pratique l'Espagnol, l'Anglais, me manque des however there's always a chacne gens. Yo siempre disfruto tomando un cafe contigo, Celia. however what are the reasons for you Ana y su madre comen un carne delicioso."
print(additional_fasttext_language_detection(stringsss))