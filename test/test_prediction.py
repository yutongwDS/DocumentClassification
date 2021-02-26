from app.model.prediction import text_prediction, batch_prediction
import pandas as pd
import pickle

# Load the model
vectorizer = pickle.load(open('../app/model/TFIDF_Vectorizer.pkl', 'rb'))
model = pickle.load(open('../app/model/XGB.pkl', 'rb'))

def test_text_prediction():
    """
    Test text_prediction class
    """
    data = "word1 word2 word3"
    prec = text_prediction(data, vectorizer, model)
    label = prec.label_prediction()
    conf = prec.confidence()
    assert isinstance(label, str)
    assert isinstance(conf, float)

def test_batch_prediction():
    """
    Test batch_prediction class
    """
    data = pd.read_csv("test_sample.csv", index_col=0)
    data.dropna(inplace=True)
    prec = batch_prediction(data, vectorizer, model)
    label = prec.label_prediction()
    conf = prec.confidence()
    result = prec.save_prediction()
    assert len(label) == len(data)
    assert len(conf) == len(data)
    assert result.shape[1] == 2

