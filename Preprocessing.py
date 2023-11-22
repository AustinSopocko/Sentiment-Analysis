import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
import math
from sklearn.feature_extraction.text import CountVectorizer
from collections import defaultdict
import pandas as pd
import string
from nltk.corpus import stopwords
import nltk
import spacy

#nltk.download('stopwords')
#nltk.download('wordnet')
lemmatizer = nltk.stem.WordNetLemmatizer()
tokeniser = nltk.tokenize.WhitespaceTokenizer()


def pre_processing():
    ReviewsFile = pd.read_csv('Test.csv') #reads dataset
    nlp = spacy.blank('en')
    ReviewsFile['text'] = ReviewsFile['text'].astype(str).str.lower() #makes all reviews lowercase
    punct = "\n\r" + string.punctuation
    ReviewsFile['text'] = ReviewsFile['text'].str.translate(str.maketrans('', '', punct)) #removes all punctuation
    stop = stopwords.words('english')
    ReviewsFile['text'] = ReviewsFile['text'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)])) # removes stopwords
    ReviewsFile['text'].apply(lemmatisation)  #lemmatisation : changing words with same meaning into one word
    return ReviewsFile



def lemmatisation(text):
    return [lemmatizer.lemmatize(w) for w in tokeniser.tokenize(text)]
