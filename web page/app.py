from flask import Flask, request, jsonify
from flask import Flask, redirect, render_template, request, abort
import pickle
from happytransformer import TTSettings
from flask_cors import CORS

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
        # Placeholder for actual analysis logic
        # This should return a list of identified mistakes
        mistakes = ["Example mistake 1", "Example mistake 2"]
        
        return mistakes
    except Exception as e:
        print(f"Error in mistake analysis: {e}")
        return None
    
@app.route('/')
def home():
 return render_template('page.html')

@app.route('/hello')
def hello():
    return "Hello, World!"

@app.route('/test_post', methods=['GET','POST'])
def test_post():
    return jsonify({"message": "POST request successful"})

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
            model = '../models/spanish_model1'
        elif language == "Russian":
            model = ""
        elif language == "Chinese (Mandarin)":
            model = ""

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
