from flask import Flask, render_template, url_for, request
import re 
import pickle
import numpy as np
from numpy.random.mtrand import standard_cauchy


standard_aspect = pickle.load(open('standard_aspect.pkl', 'rb'))
standard_text = pickle.load(open('standard_text.pkl','rb'))

vectorize_aspect = pickle.load(open('vectorize_aspect.pkl', 'rb'))
vectorize_text = pickle.load(open('vectorize_text.pkl','rb'))

model = pickle.load(open('XGboost.pkl', 'rb'))

def decontracted(phrase):
    phrase = re.sub(r"won't", "will not", phrase)
    phrase = re.sub(r"can\'t", "can not", phrase)
    phrase = re.sub(r"n\'t", " not", phrase)
    phrase = re.sub(r"\'re", " are", phrase)
    phrase = re.sub(r"\'s", " is", phrase)
    phrase = re.sub(r"\'d", " would", phrase)
    phrase = re.sub(r"\'ll", " will", phrase)
    phrase = re.sub(r"\'t", " not", phrase)
    phrase = re.sub(r"\'ve", " have", phrase)
    phrase = re.sub(r"\'m", " am", phrase)
    return phrase

def preprocess(sentance):
    sentance = re.sub(r"http\S+", "", sentance)  #removing html tags
    sentance = decontracted(sentance) #decontrast
    sentance = ' '.join(e.lower() for e in sentance.split()) #lowering
    sentance = re.sub('[^a-zA-Z]', ' ', sentance) #removing puncuation and numericals
    sentance = re.sub(r'\s+', ' ', sentance) #remving wide spaces
    return sentance

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def predict():

    if request.method == 'POST':

        message = request.form['message']
        aspect = request.form['speaker']

        message = preprocess(message)
        aspect = preprocess(aspect)
        message = vectorize_text.transform([message]).toarray()
        message = standard_text.transform(message)

        aspect = vectorize_aspect.transform([aspect]).toarray()
        aspect = standard_aspect.transform(aspect)

        final_test = np.hstack((message, aspect))
        
        my_prediction = model.predict(final_test)
        prediction = model.predict_proba(final_test)

    return render_template('result.html',prediction = [my_prediction, prediction])

if __name__ == '__main__':
	app.run(debug=True)
