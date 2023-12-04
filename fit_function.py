import math
import numpy as np


def fit(reviews, labels,labelOptions):
    reviewTotals = {}
    conditionalProbabilities = {}
    groupedReviews = groupDataByLabel(reviews, labels, labelOptions)
    for label, data in groupedReviews.items(): #loops through each label and value pair in dictionary
        reviewTotals[label] = len(data) #gets number of reviews for that label
        conditionalProbabilities[label] = math.log(reviewTotals[label]/len(reviews)) #divides number of reviews of that label by total number of reviews.
    return reviewTotals, conditionalProbabilities

def groupDataByLabel(reviews,labels,labelOptions): #groups reviews by what label it is (0,1)
    dataDict = {}
    for label in labelOptions: #for each sentiment label (0,1)
        dataDict[label] = reviews[np.where(label == labels)]
    return dataDict
    #this dictionary will have key (0 or 1) and the values will be all reviews of negative sentiment and all reviews of positive sentiment

