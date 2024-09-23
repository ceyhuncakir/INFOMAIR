from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os

labels = []
sentences = []
sentence_dict = {}

with open('data/dialog_acts.dat', 'r') as file:
    for line in file:
        label, sentence = line.strip().split(' ', 1)
        if sentence not in sentence_dict:
            sentence_dict[sentence] = label

sentences = list(sentence_dict.keys())
labels = list(sentence_dict.values())

X_train, X_test, y_train, y_test = train_test_split(sentences, labels, test_size=0.15, random_state=42)

print("Vectorizing")
vec = TfidfVectorizer()
X_train = vec.fit_transform(X_train)
X_test = vec.transform(X_test)

print("Fitting")
clf = svm.SVC()
clf.fit(X_train, y_train)

joblib.dump(vec, os.path.join('models', 'tfidf_vectorizer.pkl'))
joblib.dump(clf, os.path.join('models', 'svm_classifier.pkl'))

print("Evaluating")
y_pred = clf.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
print(classification_report(y_test, y_pred, zero_division=0))
