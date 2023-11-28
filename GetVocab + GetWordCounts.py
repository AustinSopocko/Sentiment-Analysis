import numpy as np
import math
from sklearn.feature_extraction.text import CountVectorizer
from collections import defaultdict

reviewsFile = pre_processing()
reviews = reviewsFile['text'].values
labels = reviewsFile['label'].values

def getVocabandFrequencies():
    vec = CountVectorizer(max_features = 10000) #10000 is amount of unique words, this is a random number I chose , will probably change later 
    docTermMatrix = vec.fit_transform(reviews)
    allWords = vec.get_feature_names_out() #gets list of  all words seen in reviews
    docTermMatrix = docTermMatrix.toarray()
    wordCounts = {} # dictionary for word and their frequency
    for k in range(2):
        wordCounts[k] = defaultdict(lambda: 0) # if word not seen, automatic value in dictionary is 0
    for i in range(docTermMatrix.shape[0]): #iterates through the rows of the 2D array
        currentLabel = labels[i]
        for j in range(len(allWords)):
            wordCounts[currentLabel][allWords[j]] += docTermMatrix[i][j] #increments wordcount for word with that label
    return allWords, wordCounts
