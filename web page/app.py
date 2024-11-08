from flask import Flask, request, jsonify
from flask import Flask, render_template, request
import pickle
from happytransformer import TTSettings
from flask_cors import CORS
import spacy
from collections import Counter
import difflib

app = Flask(__name__)

CORS(app)

def correct_text(model, text):
    with open(model, 'rb') as file:
        happy_tt = pickle.load(file)
    beam_settings = TTSettings(num_beams=5, min_length=1, max_length=50)
    learner_sentence = f"grammar: {text}"
    corrected_sentence = happy_tt.generate_text(learner_sentence, args=beam_settings).text
    print(corrected_sentence)
    return corrected_sentence

def analyze_mistakes(original_text, corrected_text):
    """Analyzes mistakes between the original and corrected text."""
    try:
        # Load the SpaCy English model
        nlp = spacy.load("en_core_web_sm")

        # Perform POS tagging on both texts
        original_doc = nlp(original_text)
        corrected_doc = nlp(corrected_text)

        # Create lists of (word, pos) tuples
        original_tags = [(token.text, token.pos_) for token in original_doc]
        corrected_tags = [(token.text, token.pos_) for token in corrected_doc]

        # Get the diff of tokens
        original_tokens = [token for token, pos in original_tags]
        corrected_tokens = [token for token, pos in corrected_tags]
        diff = list(difflib.ndiff(original_tokens, corrected_tokens))

        # Initialize counters and results
        classified_errors = []
        mistake_counter = Counter()
        error_indices = []

        original_index, corrected_index = 0, 0
        pos_mapping = {
            "ADJ": "adjective", "ADP": "adposition", "ADV": "adverb", "AUX": "auxiliary",
            "CCONJ": "coordinating conjunction", "DET": "determiner", "INTJ": "interjection",
            "NOUN": "noun", "NUM": "numeral", "PART": "particle", "PRON": "pronoun",
            "PROPN": "proper noun", "PUNCT": "punctuation", "SCONJ": "subordinating conjunction",
            "SYM": "symbol", "VERB": "verb", "X": "other"
        }

        i = 0
        while i < len(diff):
            if diff[i][0] == '-' and i + 1 < len(diff) and diff[i + 1][0] == '+':
                # Substitution case
                if original_index < len(original_tags) and corrected_index < len(corrected_tags):
                    original_word, original_pos_tag = original_tags[original_index]
                    corrected_word, corrected_pos_tag = corrected_tags[corrected_index]
                    error_type = pos_mapping.get(corrected_pos_tag, corrected_pos_tag).lower()
                    classified_errors.append(f"Fixed {error_type} error: '{original_word}' -> '{corrected_word}' at index {original_index}")
                    mistake_counter[error_type] += 1
                    error_indices.append(original_index)
                    original_index += 1
                    corrected_index += 1
                i += 2
            elif diff[i][0] == '-' and original_index < len(original_tags):
                # Deletion case
                original_word, original_pos_tag = original_tags[original_index]
                error_type = pos_mapping.get(original_pos_tag, original_pos_tag).lower()
                classified_errors.append(f"Deleted {error_type}: '{original_word}' at position {original_index}")
                mistake_counter[error_type] += 1
                error_indices.append(original_index)
                original_index += 1
                i += 1
            elif diff[i][0] == '+' and corrected_index < len(corrected_tags):
                # Addition case
                corrected_word, corrected_pos_tag = corrected_tags[corrected_index]
                error_type = pos_mapping.get(corrected_pos_tag, corrected_pos_tag).lower()
                classified_errors.append(f"Added {error_type}: '{corrected_word}' at position {corrected_index}")
                mistake_counter[error_type] += 1
                error_indices.append(corrected_index)
                corrected_index += 1
                i += 1
            else:
                # No change
                original_index += 1
                corrected_index += 1
                i += 1

        # Analyze the location of mistakes as percentages
        total_mistakes = sum(mistake_counter.values())
        if total_mistakes > 0:
            error_location = {"beginning": 0, "middle": 0, "end": 0}
            for index in error_indices:
                if index < len(original_tags) * 0.1:  # First 10% as beginning
                    error_location["beginning"] += 1
                elif index < len(original_tags) * 0.9:  # Middle 80%
                    error_location["middle"] += 1
                else:  # Last 10% as end
                    error_location["end"] += 1

            classified_errors.append("Error Locations (%):")
            for location, count in error_location.items():
                location_percentage = (count / total_mistakes) * 100
                classified_errors.append(f"{location.capitalize()}: {location_percentage:.2f}% of mistakes")

            # Add relative frequency of mistakes by type
            classified_errors.append("Relative Frequency of Mistakes by Type:")
            for mistake_type, count in mistake_counter.items():
                frequency = (count / total_mistakes) * 100
                classified_errors.append(f"{mistake_type.capitalize()}: {frequency:.2f}%")

        return classified_errors

    except Exception as e:
        print(f"Error in mistake analysis: {e}")
        return None
    
@app.route('/')
def home():
 return render_template('page.html')

@app.route('/correct_text', methods=['GET', 'POST'])
def correct_text_route():
    print("Route accessed")
    try:
        data = request.get_json()
        text = data.get('text')
        language = data.get('language')
        
        if not text or not language:
            return jsonify({'error': 'Missing text or language input.'}), 400

        if language == "Spanish":
            model = '../models/model_spanish'
        elif language == "Chinese (Mandarin)":
            model = '../models/model_chinese'

        else:
            return jsonify({'error': f"Model for '{language}' not found."}), 400
        
        # Perform text correction
        print("accessing correcting method")
        corrected_text = correct_text(model, text)
        
        if corrected_text is None:
            return jsonify({'error': 'Failed to process the text.'}), 500

        # Perform mistake analysis
        mistakes = analyze_mistakes(text, corrected_text)
        
        if mistakes is None:
            return jsonify({'corrected_text': corrected_text, 'error': 'Error in mistake analysis.'}), 500

        return jsonify({'corrected_text': corrected_text, 'mistakes': mistakes})
    
    except Exception as e:
        print(f"Unexpected error: {e}")
        return jsonify({'error': 'Internal server error.'}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
