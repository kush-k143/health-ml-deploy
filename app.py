import numpy as np
from flask import Flask , jsonify , request , render_template
import pickle

app = Flask(__name__ , template_folder = 'frontend/templates',
                        static_url_path = '',
                        static_folder = 'frontend/static')


model_cancer = open('model_cancer.pkl','rb')
classifier_cancer = pickle.load(model_cancer)

model_diabetes = open('model_diabetes.pkl','rb')
classifier_diabetes = pickle.load(model_diabetes)

model_heart = open('model_heart.pkl','rb')
classifier_heart = pickle.load(model_heart)

model_kidney = open('model_kidney.pkl','rb')
classifier_kidney = pickle.load(model_kidney)

@app.route('/')

def home():
    return render_template("home.html")

@app.route('/cancer')

def cancer():
    return render_template("cancer.html")

@app.route('/diabetes')

def diabetes():
    return render_template("diabetes.html")

@app.route('/heart')

def heart():
    return render_template("heart.html")

@app.route('/kidney')

def kidney():
    return render_template("kidney.html")

@app.route('/cancer_predict' , methods=['POST'])

def cancer_predict():
    radius_mean = float(request.form['radius_mean'])
    texture_mean = float(request.form['texture_mean'])
    perimeter_mean = float(request.form['perimeter_mean'])
    area_mean = float(request.form['area_mean'])
    smoothness_mean = float(request.form['smoothness_mean'])
    compactness_mean = float(request.form['compactness_mean'])
    concavity_mean = float(request.form['concavity_mean'])
    concave_points_mean = float(request.form['concave points_mean'])
    symmetry_mean = float(request.form['symmetry_mean'])
    fractal_dimension_mean = float(request.form['fractal_dimension_mean'])
    radius_se = float(request.form['radius_se'])
    texture_se = float(request.form['texture_se'])
    perimeter_se = float(request.form['perimeter_se'])
    area_se = float(request.form['area_se'])
    smoothness_se = float(request.form['smoothness_se'])
    compactness_se = float(request.form['compactness_se'])
    concavity_se = float(request.form['concavity_se'])
    concave_points_se = float(request.form['concave points_se'])
    symmetry_se = float(request.form['symmetry_se'])
    fractal_dimension_se = float(request.form['fractal_dimension_se'])
    radius_worst = float(request.form['radius_worst'])
    texture_worst = float(request.form['texture_worst'])
    perimeter_worst = float(request.form['perimeter_worst'])
    area_worst = float(request.form['area_worst'])
    smoothness_worst = float(request.form['smoothness_worst'])
    compactness_worst = float(request.form['compactness_worst'])
    concavity_worst = float(request.form['concavity_worst'])
    concave_points_worst = float(request.form['concave points_worst'])
    symmetry_worst = float(request.form['symmetry_worst'])
    fractal_dimension_worst = float(request.form['fractal_dimension_worst'])
    
    data_cancer = np.array([[radius_mean, texture_mean, perimeter_mean, area_mean,
        smoothness_mean, compactness_mean, concavity_mean,
        concave_points_mean, symmetry_mean, fractal_dimension_mean,
        radius_se, texture_se, perimeter_se, area_se, smoothness_se,
        compactness_se, concavity_se, concave_points_se, symmetry_se,
        fractal_dimension_se, radius_worst, texture_worst,
        perimeter_worst, area_worst, smoothness_worst,
        compactness_worst, concavity_worst, concave_points_worst,
        symmetry_worst, fractal_dimension_worst]])
    
    my_prediction = classifier_cancer.predict(data_cancer)
    final_prediction = my_prediction[0]
        
    return render_template('cancer-predict.html', prediction_cancer=final_prediction)

@app.route('/diabetes_predict' , methods=['POST'])

def diabetes_predict():
    Pregnancies = int(request.form['Pregnancies'])
    Glucose = float(request.form['Glucose'])
    BloodPressure = float(request.form['BloodPressure'])
    SkinThickness = float(request.form['SkinThickness'])
    Insulin = float(request.form['Insulin'])
    BMI = float(request.form['BMI'])
    DiabetesPedigreeFunction = float(request.form['DiabetesPedigreeFunction'])
    Age = float(request.form['Age'])

    data_diabetes = np.array([[Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin,
                            BMI, DiabetesPedigreeFunction, Age]])
    
    my_prediction = classifier_diabetes.predict(data_diabetes)
    final_prediction = my_prediction[0]
        
    return render_template('diabetes-predict.html', prediction_diabetes=final_prediction)

@app.route('/heart_predict', methods=['POST'])

def heart_predict():
    age = int(request.form['age'])
    cp = float(request.form['cp'])
    trestbps = float(request.form['trestbps'])
    chol = float(request.form['chol'])
    fbs = float(request.form['fbs'])
    restecg = float(request.form['restecg'])
    thalach = float(request.form['thalach'])
    exang = float(request.form['exang'])
    oldpeak = float(request.form['oldpeak'])
    slope = float(request.form['slope'])
    ca = float(request.form['ca'])
    thal = float(request.form['thal'])

    data_heart = np.array([[age, cp, trestbps, chol, fbs, restecg, thalach, exang,
                            oldpeak, slope, ca, thal]])
    
    my_prediction = classifier_heart.predict(data_heart)
    final_prediction = my_prediction[0]

    return render_template('heart-predict.html' , prediction_heart = final_prediction)

@app.route('/kidney_predict', methods=['POST'])

def kidney_predict():
    age = int(request.form['age'])
    bp = float(request.form['bp'])
    sg = float(request.form['sg'])
    al = float(request.form['al'])
    bgr = float(request.form['bgr'])
    bu = float(request.form['bu'])
    sc = float(request.form['sc']) 
    hemo = float(request.form['hemo'])
    pcv = float(request.form['pcv'])
    pcc_present = float(request.form['pcc_present'])
    htn = float(request.form['htn'])
    dm = float(request.form['dm'])
    cad = float(request.form['cad'])
    appet = float(request.form['appet'])
    ane = float(request.form['ane'])

    data_kidney = np.array([[age, bp, sg, al, bgr, bu, sc, hemo, pcv,
                            pcc_present, htn, dm, cad, appet, ane]])
    
    my_prediction = classifier_kidney.predict(data_kidney)
    final_prediction = my_prediction[0]

    return render_template('kidney-predict.html' , prediction_kidney = final_prediction)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
