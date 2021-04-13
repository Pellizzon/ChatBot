# To add a new cell, type ' '
# To add a new markdown cell, type '  [markdown]'
 
import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
import pickle

 

if __name__ == "__main__":
    #Load model
    with open("chatbot_model.sav", 'rb') as f:
        vectorizer, model = pickle.load(f)

    #Test model
    nova_sentenca = 'Acende a luz por favor!'
    counts_da_nova_sentenca = vectorizer.transform([nova_sentenca])
    print(model.predict(counts_da_nova_sentenca))

    #TODO: partial_fit




     


