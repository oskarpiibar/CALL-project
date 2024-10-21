import re
import ftfy
import nltk
from nltk.corpus import words
from nltk import tokenize, word_tokenize
from nltk.stem import WordNetLemmatizer
import langdetect

nltk.download('words')
nltk.download('punkt_tab')
nltk.download('omw-1.4')
nltk.download('wordnet')

lemmatizer = WordNetLemmatizer()

english_words = set(words.words())


def remove_symbols(text):
    return re.sub(r'[^A-Za-z0-9\s.,;:!?\'"()%\-]', '', text)


def filter_non_english(text):
    text = text.lower()
    words_list = re.split(r'\s+|/', text)
    english_only_words = []
    for word in words_list:
        original_word = word.strip(".,%:;!?\"'()[]{}")
        base_word = lemmatizer.lemmatize(original_word)
        try:
            if langdetect.detect(original_word or base_word) == 'en':
                english_only_words.append(word)
            elif ((original_word.rstrip("'s") in english_words) or 
                (str.isdigit(original_word)) or
                ("'" in original_word)):
                english_only_words.append(word)
        except langdetect.lang_detect_exception.LangDetectException:
            pass  # If language detection fails, skip the word

    print(english_only_words)
    if len(words_list) == 0:
        return ''  # Return empty string if text is empty
    english_percentage = len(english_only_words) / len(words_list)
    if english_percentage >= 0.3:
        return ' '.join(english_only_words)  # Return only English words
    else:
        return None  # Mark row for removal if below 30% English content

def filter_non_english_rows(text):
    words_list = re.split(r'\s+|/', text)
    english_only_words = []
    for word in words_list:
        # Keep the original word for proper nouns
        original_word = word.strip(".,%:;!?\"'()[]{}")
        clean_word = original_word.lower()
        
        # Lemmatize to handle plurals, e.g., "stories" -> "story"
        base_word = lemmatizer.lemmatize(clean_word)
        
        if (base_word in english_words or
            clean_word.rstrip("'s") in english_words or
            clean_word in english_words or
            original_word.istitle() or
            str.isdigit(original_word) or
            "'" in original_word):  # Allow contractions with apostrophes like don't
            english_only_words.append(word)  # Keep original word with punctuation

    # Calculate English percentage and determine if row is mostly English
    if len(words_list) == 0:
        return ''  # Return empty string if text is empty
    english_percentage = len(english_only_words) / len(words_list)
    if english_percentage >= 0.3:
        return ' '.join(english_only_words)  # Return only English words
    else:
        return None  # Mark row for removal if below 30% English content
    
def add_space_after_punctuation(text):
    text = re.sub(r'\s+([.!?])', r'\1', text) #remove space before a punctuation
    text = re.sub(r'([.!?])([a-zA-Z])', r'\1 \2', text) #add space after punctuation
    return text

def remove_short_senteces(text):
    if not isinstance(text, str) or text is None:
        return ''
    sentences = tokenize.sent_tokenize(text)
    new_text = ''
    for sentence in sentences:
        if len(sentence.split()) >= 3:  # Keep only sentences with more than 3 actual words
            new_text += sentence + ' '
    return new_text.strip()

# string = "hi;hello there ?every life's a 50% : 50% movie. We got different stars and stories. We got different nights and mornings.Â°â€¢â˜†Â°â€¢â˜†Â°â€¢â˜†Â°â€¢â˜†Â°â€¢â˜†Â°â€¢â˜†Â°â€¢â˜†â€¢"
string = "Je Monique 58 ans, shouldn't However this matter don't I'm okay de Metz dans Nord-Est de la France. Je pratique l'Espagnol, l'Anglais, me manque des gens Tous les les retour, j'aiderai la pratique Franais"

new = add_space_after_punctuation(string)
new = ftfy.fix_text(new)
print(f'Fixing characters: {new}')
new2 = remove_symbols(new)
print(f'Symbols: {new2}')
new3 = remove_short_senteces(new2)
print(f'Short sentences: {new3}')
new4 = filter_non_english_rows(new3)
print(f'English words: {new4}')

if 'Ciao' in english_words:
    print('yes')
else:
    print('no')