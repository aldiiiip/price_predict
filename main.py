import pandas as pd
import pickle
from flask import Flask, request, jsonify, render_template
import json

app = Flask(__name__)

# Load your model pipeline
with open('price_model.pickle', 'rb') as f:
    model_pipeline = pickle.load(f)

# Load the locations from the JSON file
with open('value_names.json', 'r') as file:
    values = json.load(file)

# Function to predict rent price
def predict_rent_price(model_pipeline, location, property_type, rooms, parking, bathroom, size, furnished):
    data = {
        'location': location,
        'property_type': property_type,
        'rooms': rooms,
        'parking': parking,
        'bathroom': bathroom,
        'size': size,
        'furnished': furnished
    }
    input_df = pd.DataFrame([data])
    prediction = model_pipeline.predict(input_df)[0]
    return prediction


@app.route('/', methods=['GET', 'POST'])
def index():
    predicted_rent = None
    form_data = {}  # Initialize a dictionary to store form data

    if request.method == 'POST':
        # Get form data
        form_data['location'] = request.form['location']
        form_data['property_type'] = request.form['property_type']
        form_data['rooms'] = request.form['rooms']
        form_data['parking'] = request.form['parking']
        form_data['bathroom'] = request.form['bathroom']
        form_data['size'] = request.form['size']
        form_data['furnished'] = request.form['furnished']

        # Predict the rent price
        predicted_rent = round(predict_rent_price(
            model_pipeline,
            form_data['location'],
            form_data['property_type'],
            float(form_data['rooms']),
            float(form_data['parking']),
            float(form_data['bathroom']),
            float(form_data['size']),
            form_data['furnished']
        )/10)*10

    # Pass form_data to template
    return render_template('index.html', 
                           locations=values['locations'], 
                           property_types=values['property_types'], 
                           furnished=values['furnished'], 
                           prediction=predicted_rent, 
                           form_data=form_data)

if __name__ == "__main__":
    app.run(debug = True)


