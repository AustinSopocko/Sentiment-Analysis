from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

labelOptions = [0,1] # this is an array of all possible sentiment labels

train_reviews, test_reviews, train_labels, test_labels = train_test_split(reviews,labels) # split data into testing and training

allWords, wordCounts = getVocabandFrequencies() #gets set of all words, and dictionary of a word and its corresponding frequency

reviewTotals, conditionalProbabilities = fit(train_reviews,train_labels,labelOptions) #reviewTotals is a dictionary with key : label and value : number of reviews
#conditional probabilities is a dictionary with key : label and value : log probability of label

prediction = predict(allWords,reviewTotals,wordCounts,labelOptions,test_reviews,conditionalProbabilities) #returns array of new reviews and corresponding predicted label

print("Accuracy on test set - ", accuracy_score(prediction,test_labels)) #gets accuracy score of model on the data

