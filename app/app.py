from flask import Flask, request, render_template
import pickle
import pandas as pd
import os
from app.model.prediction import text_prediction, batch_prediction

# creating a Flask application
app = Flask(__name__, template_folder="templates")

# Load the model
vectorizer = pickle.load(open('model/TFIDF_Vectorizer.pkl', 'rb'))
model = pickle.load(open('model/XGB.pkl', 'rb'))


@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        data = request.form.get('word')

        # Make prediction
        prec = text_prediction(data, vectorizer, model)
        pred = prec.label_prediction()
        confidence = prec.confidence()
        pred_prob = prec.prediction_probability()

        return render_template('index.html', prediction=pred, confidence=confidence, tables=[pred_prob.to_html(classes='data')], titles=pred_prob.columns.values)
    else:
        return render_template('index.html', prediction='', probability='')


@app.route('/batch_prediction', methods=["GET", "POST"])
def file_prediction():
    if request.method == 'POST':
        f = request.files['file']
        if f:
            # Delete file if exists
            try:
                os.remove(f.filename)
            except OSError:
                pass

            # Save uploaded file for processing
            f.save(f.filename)
            print("File Saved")

            # Read file into Pandas Dataframe
            data = pd.read_csv(f.filename, index_col=0).reset_index(drop=True)

            prec = batch_prediction(data, vectorizer, model)
            pred = prec.label_prediction()
            result = prec.save_prediction()

            return render_template('batch_index.html', prediction=pred, tables=[result.to_html(classes='data')], titles=result.columns.values)
    return render_template('batch_index.html', prediction='')


if __name__ == '__main__':
    app.run(port=3000, debug=True)