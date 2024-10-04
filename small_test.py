import re
import ftfy
import nltk
from nltk.corpus import words
from nltk.stem import WordNetLemmatizer

nltk.download('words')
nltk.download('omw-1.4')
nltk.download('wordnet')

lemmatizer = WordNetLemmatizer()

english_words = set(words.words())


def remove_symbols(text):
    return re.sub(r'[^A-Za-z0-9\s.,!?\'"()\-]', '', text)

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

string = "Every life's a movie We got different stars and stories We got different nights and mornings.Â°â€¢â˜†Â°â€¢â˜†Â°â€¢â˜†Â°â€¢â˜†Â°â€¢â˜†Â°â€¢â˜†Â°â€¢â˜†â€¢"

new = ftfy.fix_text(string)
print(new)
new2 = remove_symbols(new)
print(new2)
new3 = filter_non_english_rows(new2)
print(new3)

if "stories" in english_words:
    print('yes')
else:
    print('no')