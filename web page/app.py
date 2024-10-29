from flask import Flask, request, jsonify
import torch  


app = Flask(__name__)

model_spanish = torch.load("path/to/spanish_model.pt")
model_russian = torch.load("path/to/russian_model.pt")
model_chinese = torch.load("path/to/chinese_model.pt")


def correct_text(model, text):
    # method name needs to be changed to the name they put
    corrected_text = model.correct(text)
    return corrected_text

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

@app.route('/correct_text', methods=['POST'])
def correct_text_route():
    try:
        data = request.get_json()
        text = data.get('text')
        language = data.get('language')
        
        if not text or not language:
            return jsonify({'error': 'Missing text or language input.'}), 400

        if language == "Spanish":
            model = model_spanish
        elif language == "Russian":
            model = model_russian

        elif language == "Chinese (Mandarin)":
            model = model_chinese

        else:
            return jsonify({'error': f"Model for '{language}' not found."}), 400
        
        # Perform text correction
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
    app.run(debug=True)
