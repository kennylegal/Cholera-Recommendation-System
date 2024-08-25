from flask import Flask, request, render_template
import joblib

app = Flask(__name__)

# Define the custom tokenizer function used in the original model training
def custom_tokenizer(text):
    return text  # Since the text was already a list of symptoms

# Load the model and preprocessing objects
rf_model = joblib.load('model/MyRecommendationModel.pkl')
vectorizer = joblib.load('model/vectorizer.pkl')
mlb = joblib.load('model/label_binarizer.pkl')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get symptoms from form
    symptoms = request.form['symptoms']
    
    # Preprocess the input symptoms
    symptoms_list = symptoms.split(', ')
    symptoms_vector = vectorizer.transform([symptoms_list]).toarray()

    # Predict the treatment
    treatment_prediction = rf_model.predict(symptoms_vector)
    treatment_list = mlb.inverse_transform(treatment_prediction)

    # Render the results
    return render_template('result.html', treatments=treatment_list[0])

if __name__ == '__main__':
    app.run(debug=True)
