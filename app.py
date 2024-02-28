from flask import Flask, render_template, request
import pickle
import pandas as pd
import numpy as np

app = Flask(__name__)

model = pickle.load(open('PricePrediction.pkl','rb'))
car = pd.read_csv('Cleaned_Car_data.csv')

@app.route('/')
def home():
    companies = sorted(car['company'].unique())
    car_models=sorted(car['name'].unique())
    year=sorted(car['year'].unique(),reverse=True)
    fuel_type=car['fuel_type'].unique()

    companies.insert(0,'Select Company')
    return render_template('home.html', companies=companies, car_models=car_models, years=year,fuel_types=fuel_type)

@app.route('/about')
def about():
    return render_template('about.html')


@app.route('/predict')
def predict():
    company = request.args.get('company')
    cmodel = request.args.get('model')
    year = request.args.get('year')
    kml = request.args.get('kml')
    fueltype  = request.args.get('fueltype')

    prediction = model.predict(pd.DataFrame(
                columns=['name', 'company', 'year', 'kms_driven', 'fuel_type'], 
                data=np.array([cmodel,company,year,kml,fueltype]).reshape(1, 5)))
    

    return render_template('predict.html', company = company, cmodel = cmodel, year = year, kml = kml,fuel_type = fueltype, prediction = str(np.round(prediction[0],2)))

if __name__ == "__main__":
    app.run(debug=True, port=8000)