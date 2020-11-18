#!/usr/bin/env python
# coding: utf-8

# In[3]:


from flask import Flask, render_template, request
from sklearn.ensemble import RandomForestClassifier
import pandas as pd 
import pickle

import pickle


app = Flask(__name__) 

@app.route('/') #Routes the app to below task when the URL is called 

def home(): 

    return render_template('home.html')
def predict_fun(): 

    NB_spam_model = open('RF_classifier.pkl','rb') 
    clf = pickle.load(NB_spam_model)

    if request.method == 'POST': 

        message = request.form['message'] 
        data = [message] 
        vect = cv.transform(data).toarray()
        my_prediction = clf.predict(vect) 

    return render_template('result.html',prediction = my_prediction)   

#Calling the main function and running the flask app 

if __name__ == '__main__':
    app.run(debug=True)


# In[ ]:




