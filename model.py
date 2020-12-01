# -*- coding: utf-8 -*-
"""
Created on Thu Nov 26 22:24:43 2020

@author: Vinanth S Bharadwaj
"""
from nltk.corpus import brown
import regex as re
import random
from caesarcipher import CaesarCipher
import tensorflow as tf
from sklearn.model_selection import train_test_split
import numpy as np
import string

def model(query):
	sentences = brown.sents()
	remove = string.punctuation
	pattern = r"[{}]".format(remove) 
	sents_cleaned=[]
	for i in sentences:
	    temp=[]
	    for j in i:
	        j=re.sub(pattern,"", j)
	        if j!="":
	            temp.append(j)
	    sents_cleaned.append(temp)
	X=[]
	for i in sents_cleaned:
	    str = ' '.join(i)
	    rand = random.randint(1,26)
	    cipher = CaesarCipher(str,offset=rand)
	    temp=[]
	    count={}
	    for i in range(1,27):
	        count[i]=0
	    encoded = cipher.encoded
	    encoded = encoded.lower()
	    for letter in encoded:
	        if letter.isalpha():
	            count[ord(letter)-97+1] = count.get(ord(letter)-97+1, 0) + 1 
	    temp.append(count)
	    temp.append(rand)
	    X.append(temp)
	temp = []
	train_list=[]
	for i in range(0,len(X)):
	    for key,value in X[i][0].items():
	        train_list.append(value)  
	    temp.append(X[i][1]-1)

	train_array = np.array(train_list) 
	labels = np.array(temp)

	train_array = train_array/26
	set = train_array.reshape(len(X),26)
	labels.reshape(len(X),1)

	data_train, data_test, labels_train, labels_test = train_test_split(set, labels, test_size=0.20, random_state=42)

	classifier = tf.keras.Sequential()
	#First Hidden Layer
	classifier.add(tf.keras.layers.Dense(26, activation='relu',kernel_initializer='random_normal', input_dim=26))

	#Second  Hidden Layer
	classifier.add(tf.keras.layers.Dense(26, activation='sigmoid', kernel_initializer='random_normal'))

	# #Output Layer
	classifier.add(tf.keras.layers.Dense(26, activation='softmax', kernel_initializer='random_normal'))

	classifier.compile(optimizer ='adam',loss='sparse_categorical_crossentropy',metrics =['accuracy'])

	classifier.fit(data_train,labels_train, batch_size=10, epochs=15)
	
	count_test={}
	for i in range(1,27):
		count_test[i]=0
	for letter in query:
		if letter.isalpha():
			count[ord(letter)-97+1] = count.get(ord(letter)-97+1, 0) + 1
	temp_list=[]
	for key,value in count_test.items():
		temp_list.append(value)
	test_array = np.array(temp_list)
	test_array = test_array.reshape(1,26)
	return classifier.predict(test_array)
