async function correctText() {
    const language = document.getElementById("language-select").value;
    const text = document.getElementById("text-input").value;

    try {
        const response = await fetch('/correct_text', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ text: text, language: language })
        });

        if (response.ok) {
            const result = await response.json();
            document.getElementById("output").innerText = result.corrected_text;
            displayMistakes(analyzeMistakes(text, result.corrected_text));
        } else {
            document.getElementById("output").innerText = "Error: Unable to process your text.";
        }
    } catch (error) {
        console.error('Error:', error);
        document.getElementById("output").innerText = "Error: Unable to connect to server.";
    }
}


// Function to analyze mistakes (you can modify this logic as needed)
function analyzeMistakes(originalText, correctedText) {
    // Placeholder logic for example
    return [
        "Missing article before noun",
        "Verb tense correction",
        "Incorrect preposition usage"
    ];
}

// Function to display mistakes in the 'mistakes-output' list
function displayMistakes(mistakes) {
    const mistakesOutput = document.getElementById("mistakes-output");
    mistakesOutput.innerHTML = ""; // Clear previous mistakes

    mistakes.forEach(mistake => {
        const li = document.createElement("li");
        li.textContent = mistake;
        mistakesOutput.appendChild(li);
    });
}
