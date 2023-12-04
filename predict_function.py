import nltk
def predict(allWords, reviewTotals,wordCounts,labelOptions,reviews,conditionalProbabilities): #allWords is all the words already seen
    labelScores = {}
    result = []

    for review in reviews:  #for each review
        labelScores = {currentLabel: conditionalProbabilities[currentLabel] for currentLabel in labelOptions}
        #turns review into set of words
        tokenizer = nltk.tokenize.WhitespaceTokenizer()
        wordSet = tokenizer(review)
        for word in wordSet: #loops through words in review
            if word in allWords: #if we have already seen this word in training phase
                for label in labelOptions: #for each sentiment (0,1)
                    laplaceScore = laplace(wordCounts,allWords,word,reviewTotals,label) #gets laplace score for that word
                    labelScores[label] += laplaceScore #adds score for label
        result.append(max(labelScores, key=labelScores.get)) #adds label with the highest score for that review to result array  

    return result

