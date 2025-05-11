from flask import Flask, render_template, request
import pickle
import spacy

# Initialize Flask app
app = Flask(__name__)

# Load spaCy multilingual model
nlp = spacy.load("xx_ent_wiki_sm")

# Load saved model, vectorizer, and label encoder
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

with open("label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)

# Clean text using spaCy
def clean_text(text):
    doc = nlp(text)
    return " ".join([token.text for token in doc if not token.is_stop and not token.is_punct])

# Routes
@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    if request.method == "POST":
        input_text = request.form["text"]
        cleaned_text = clean_text(input_text)
        vec = vectorizer.transform([cleaned_text])
        pred = model.predict(vec)
        lang = label_encoder.inverse_transform(pred)[0]
        prediction = f"Detected Language: {lang}"
    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)
