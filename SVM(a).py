import nltk
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import svm
from sklearn.metrics import classification_report
from sklearn import metrics
import numpy as np

def alpha_filter(w):
  # pattern to match a word of non-alphabetical characters
    pattern = re.compile('^[^a-z]+$')
    if (pattern.match(w)):
        return True
    else:
        return False

def pre_process(r):                               #processing the comments, turn them into the format that can be analyzed
  r = nltk.word_tokenize(r)                           #tokenization
  r = [w.lower() for w in r]                          #turn them into lower case
  r = [word for word in r if not any(c.isdigit() for c in word)]            #delete all the numbers
  stop = nltk.corpus.stopwords.words('english')                     #delete all the stop words
  stop.append('@USER')
  stop.append('user')
  r = [x for x in r if x not in stop]
  r = [w for w in r if not alpha_filter(w)]
  porter = nltk.PorterStemmer()                            #stemming
  r = [porter.stem(t) for t in r]
  wnl = nltk.WordNetLemmatizer()                          #lemmatizing
  r= [wnl.lemmatize(t) for t in r]
  text = " ".join(r)
  return text

def put_in_list(raw):
    a = []
    for i in raw:
        a.append(i)
    return a

f0 = pd.read_table("D:/cs/P2/Project/olid-training-v1.0.txt")
f1 = pd.read_table("D:/cs/P2/testset-levela.txt")
f2 = pd.read_table("D:/cs/P2/labels-levela.txt")

vectorizer = TfidfVectorizer(min_df = 5,
                             max_df = 0.8,
                             sublinear_tf = True,
                             use_idf = True)

traintexts = put_in_list(f0['tweet'])
trainlabels = put_in_list(f0['subtask_a'])
testtexts = put_in_list(f1['tweet'])
testlabels = put_in_list(f2['id,tag'])

for m in range(len(testlabels)):
    testlabels[m] = testlabels[m][-3:]

for i in range(len(traintexts)):                  #pre-process the tweets
    traintexts[i] = pre_process(traintexts[i])
for i in range(len(testtexts)):
    testtexts[i] = pre_process(testtexts[i])

train_vectors = vectorizer.fit_transform(traintexts)      #vectorize the tweets
test_vectors = vectorizer.transform(testtexts)

classifier = svm.SVC(kernel='linear')                 #build the model
model = classifier.fit(train_vectors, trainlabels)     #training
res = classifier.predict(test_vectors)                 #predicting
count = 0
for i in res:
    if(i=='OFF'):
        count = count + 1
print(res)
print('The number of aggresive tweets is: ',count)
report = classification_report(testlabels, res, output_dict=True)
f1_score = metrics.f1_score(testlabels,res, average='weighted', labels=np.unique(res))
print('The weighted avg summary is: ',report['weighted avg'])
print('The macro avg summary is: ',report['macro avg'])





