from sklearn import preprocessing,metrics
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import ensemble
import re
import nltk
import pandas as pd
from sklearn.metrics import classification_report
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
  stop.append('@user')
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
f1 = pd.read_table("D:/cs/P2/testset-levelc.txt")
f2 = pd.read_table("D:/cs/P2/labels-levelc.txt")

trainlabels = put_in_list(f0['subtask_c'])
text_indexlist_null = []
for i in range(len(trainlabels)-1,-1,-1):
    if (trainlabels[i] != 'OTH' and trainlabels[i] != 'IND' and trainlabels[i] != 'GRP'):
        trainlabels.pop(i)
        text_indexlist_null.append(i)
traintexts = put_in_list(f0['tweet'])
for i in range(len(traintexts)-1,-1,-1):
    if(i in text_indexlist_null):
        del traintexts[i]
testlabels = put_in_list(f2['id,tag'])
for h in range(len(testlabels)):
    testlabels[h] = testlabels[h][-3:]
testtexts = put_in_list(f1['tweet'])


for i in range(len(traintexts)):
    traintexts[i] = pre_process(traintexts[i])
for i in range(len(testtexts)):
    testtexts[i] = pre_process(testtexts[i])

trainDF = pd.DataFrame()
trainDF['text'] = traintexts
train_x = trainDF['text']
trainDF['label'] = trainlabels
train_y = trainDF['label']

testDF = pd.DataFrame()
testDF['text'] = testtexts
valid_x = testDF['text']
testDF['label'] = testlabels
valid_y = testDF['label']

encoder = preprocessing.LabelEncoder()
train_y = encoder.fit_transform(train_y)
valid_y = encoder.fit_transform(valid_y)

count_vect = CountVectorizer(analyzer='word', token_pattern=r'\w{1,}')
count_vect.fit(trainDF['text'])

#使用向量计数器对象转换训练集和验证集
xtrain_count = count_vect.transform(train_x)
xvalid_count = count_vect.transform(valid_x)

def train_model(classifier, feature_vector_train, label, feature_vector_valid, is_neural_net=False):
# fit the training dataset on the classifier
  classifier.fit(feature_vector_train, label)

# predict the labels on validation dataset
  predictions = classifier.predict(feature_vector_valid)

  if is_neural_net:
    predictions = predictions.argmax(axis=-1)


  return  predictions

predictions = train_model(ensemble.RandomForestClassifier(), xtrain_count, train_y, xvalid_count)

count0 = 0
count1 = 0
count2 = 0
result = []
for g in predictions:
    if(g==0):
        count0 = count0 + 1
        result.append('GRP')
    elif(g==1):
        count1 = count1 + 1
        result.append('IND')
    else:
        count2 = count2 + 1
        result.append('OTH')

print(result)

print('The number of  aggressive tweets targeting at individuals is: ',count1)
print('The number of  aggressive tweets targeting at groups is: ',count0)
print('The number of  aggressive tweets targeting at others is: ',count2)
report = classification_report(testlabels, result, output_dict=True)
print('The overall accuracy is: ',report['accuracy'])
f1_score = metrics.f1_score(testlabels,result, average='weighted', labels=np.unique(result))
print('The overall f1 score is: ',f1_score)
print('The macro avg summary is: ',report['macro avg'])
print('The weighted avg summary is: ',report['weighted avg'])














