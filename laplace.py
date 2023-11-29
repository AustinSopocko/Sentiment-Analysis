import math
def laplace(wordCounts,allWords,currentWord, reviewTotals, currentLabel):
    x = wordCounts[currentLabel][currentWord] + 1 # gets frequency of word + 1
    y = reviewTotals[currentLabel] + len(allWords) #gets number of reviews for that label and the size of all words seen
    return math.log(x/y)
