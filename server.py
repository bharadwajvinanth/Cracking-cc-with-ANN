# -*- coding: utf-8 -*-
"""
Created on Thu Nov 26 21:41:27 2020

@author: Vinanth S Bharadwaj
"""
from flask import Flask, request, render_template
from caesarcipher import CaesarCipher
from tensorflow.keras.models import load_model
import numpy as np


app = Flask(__name__)

@app.route("/")
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
	query = request.form['cipher']
	model = load_model('model.h5')
	count_test={}
	for i in range(1,27):
		count_test[i]=0
	for letter in query:
		if letter.isalpha():
			count_test[ord(letter)-97+1] = count_test.get(ord(letter)-97+1, 0) + 1
	temp_list=[]
	for key,value in count_test.items():
		temp_list.append(value)
	test_array = np.array(temp_list)
	test_array = test_array.reshape(1,26)
	key = (np.argmax(model.predict(test_array))+1)
	d= CaesarCipher(query,offset=key)
	decoded_string = d.decoded
	return render_template('display.html', string = query , key = key, decoded_string = decoded_string)

if __name__ == '__main__':
    app.run(debug=True)
