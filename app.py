from flask import Flask, render_template, url_for, request
# import pandas as pd
import re 
import pickle
import numpy as np
from transformers import DistilBertTokenizer, TFDistilBertModel
from transformers import  DistilBertConfig
from sklearn.feature_extraction.text import CountVectorizer
import tqdm
# import tensorflow as tf


distil_bert = 'distilbert-base-uncased'
config = DistilBertConfig.from_pretrained(distil_bert, output_hidden_states=True)
tokenizer = DistilBertTokenizer.from_pretrained(distil_bert)
bert_model =  TFDistilBertModel.from_pretrained(distil_bert, config=config)

encode = pickle.load(open('encode_instance.pkl', 'rb'))
standard = pickle.load(open('standard.pkl','rb'))

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
        aspect = request.form['aspect']
        # message = [message]
        # aspect = [aspect]
        message = preprocess(message)
        aspect = encode.transform([aspect]).toarray()
        aspect = standard.transform(aspect)
        e = tokenizer.encode(message)
        input = np.array(e).reshape(-1,1)
        output = bert_model(input)
        output = output[1][-1][0][0]
        output = np.array(output).reshape(-1,768)
        aspect = np.array(aspect).reshape(-1, 100)
        final_test = np.hstack((output, aspect))
        model = pickle.load(open('Bert_SVM.pkl','rb'))
        prediction = model.predict_proba(final_test)
        my_prediction = np.argmax(prediction, axis=1)

    return render_template('result.html',prediction = [my_prediction, prediction])

if __name__ == '__main__':
	app.run(debug=True)

