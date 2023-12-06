import numpy as np
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

reviewsFile = pre_processing()
reviews = reviewsFile['text'].values
labels = reviewsFile['label'].values

#split data
train_reviews, test_reviews, train_labels, test_labels = train_test_split(reviews,labels)

def getVocabandFrequencies():
    vec = CountVectorizer(max_features = 25000)
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



tokenizer = nltk.tokenize.WhitespaceTokenizer()


def laplace(wordCounts,allWords,currentWord, reviewTotals, currentLabel):
    x = wordCounts[currentLabel][currentWord] + 1 # gets frequency of word + 1
    y = reviewTotals[currentLabel] + len(allWords) #gets number of reviews for that label and the size of all words seen
    return math.log(x/y)

def groupDataByLabel(reviews,labels,labelOptions): #groups reviews by what label it is (0,1)
    dataDict = {}
    for label in labelOptions:
        dataDict[label] = reviews[np.where(label == labels)]
    return dataDict #this dictionary will have key (0 or 1) and values will be all reviews of negative sentiment and all reviews of positive sentiment



def fit(reviews, labels,labelOptions):
    reviewTotals = {}
    conditionalProbabilities = {}
    groupedReviews = groupDataByLabel(reviews, labels, labelOptions)
    for label, data in groupedReviews.items():
        reviewTotals[label] = len(data) #gets number of reviews for that label
        conditionalProbabilities[label] = math.log(reviewTotals[label]/len(reviews)) #divides number of reviews of that label by total number of reviews.
    return reviewTotals, conditionalProbabilities

def predict(allWords, reviewTotals,wordCounts,labelOptions,reviews,conditionalProbabilities): #allWords is all the words already seen
    labelScores = {}
    result = []

    for review in reviews:
        labelScores = {currentLabel: conditionalProbabilities[currentLabel] for currentLabel in labelOptions}
        #turns review into set of words
        wordSet = tokenizer.tokenize(review)
        for word in wordSet:
            if word in allWords: #if we have already seen this word in training phase
                for label in labelOptions: #looping through each label
                    laplaceScore = laplace(wordCounts,allWords,word,reviewTotals,label)
                    labelScores[label] += laplaceScore
        result.append(max(labelScores, key=labelScores.get))

    return result

labelOptions = [0,1]

train_reviews, test_reviews, train_labels, test_labels = train_test_split(reviews,labels)

allWords, wordCounts = getVocabandFrequencies()

reviewTotals, conditionalProbabilities = fit(train_reviews,train_labels,labelOptions)


prediction = predict(allWords,reviewTotals,wordCounts,labelOptions,test_reviews,conditionalProbabilities)

print("Accuracy on test set - ", accuracy_score(prediction,test_labels))






