from math import log

def laplace(wordFrequency, sumAllWordFrequencies, sizeAllWords):
    probability = (wordFrequency + 1) / (sumAllWordFrequencies + sizeAllWords)
    return log(probability)