import nltk
def predict(allWords, reviewTotals,wordCounts,labelOptions,reviews,conditionalProbabilities): #allWords is all the words already seen
    labelScores = {}
    result = []

    for review in reviews:
        labelScores = {currentLabel: conditionalProbabilities[currentLabel] for currentLabel in labelOptions}
        #turns review into set of words
        tokenizer = nltk.tokenize.WhitespaceTokenizer()
        wordSet = tokenizer(review)
        for word in wordSet:
            if word in allWords: #if we have already seen this word in training phase
                for label in labelOptions: #looping through each label
                    laplaceScore = laplace(wordCounts,allWords,word,reviewTotals,label)
                    labelScores[label] += laplaceScore
        result.append(max(labelScores, key=labelScores.get))

    return result

