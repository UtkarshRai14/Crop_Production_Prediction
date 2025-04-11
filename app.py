from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import pickle

app = Flask(__name__)
app.secret_key = 'your_secret_key_here'

# Load the trained CatBoost model
with open('catboost_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Dummy options – replace with your actual values
crops = ['Wheat', 'Rice', 'Maize', 'Barley']
states = ['Maharashtra', 'Punjab', 'Karnataka', 'Bihar']
seasons = ['Kharif', 'Rabi', 'Summer', 'Winter', 'Whole Year']

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    warning = None
    form_data = None

    if request.method == 'POST':
        try:
            area_input = request.form['Area']
            area = float(area_input)

            if area == 0:
                prediction = 0
            else:
                data = {
                    'Crop': request.form['Crop'],
                    'Crop_Year': int(request.form['Crop_Year']),
                    'Season': request.form['Season'],
                    'State': request.form['State'],
                    'Area': np.log1p(area),
                    'Annual_Rainfall': np.log1p(float(request.form['Annual_Rainfall'])),
                    'Fertilizer': np.log1p(float(request.form['Fertilizer'])),
                    'Pesticide': np.log1p(float(request.form['Pesticide']))
                }

                input_df = pd.DataFrame([data])
                pred_log = model.predict(input_df)[0]
                prediction = round(np.expm1(pred_log), 2)

                if prediction <= 0:
                    warning = (
                        "⚠️ The predicted yield is zero or negative. "
                        "This usually means the inputs may not be realistic. "
                        "Please double-check and try again."
                    )

            form_data = request.form

        except Exception as e:
            print("Prediction Error:", e)
            prediction = None
            warning = "⚠️ Something went wrong. Please check your input values."
            form_data = request.form

    return render_template(
        'index.html',
        prediction=prediction,
        warning=warning,
        form_data=form_data,
        crops=crops,
        states=states,
        seasons=seasons
    )

if __name__ == '__main__':
    app.run(debug=True)
