from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

labelOptions = [0,1]

train_reviews, test_reviews, train_labels, test_labels = train_test_split(reviews,labels) # split data

allWords, wordCounts = getVocabandFrequencies()

reviewTotals, conditionalProbabilities = fit(train_reviews,train_labels,labelOptions)

prediction = predict(allWords,reviewTotals,wordCounts,labelOptions,test_reviews,conditionalProbabilities)

print("Accuracy on test set - ", accuracy_score(prediction,test_labels))
