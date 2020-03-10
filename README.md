# Identifying-and-Classifying-Offensive-Tweets
Offensive language is increasingly pervasive in social media, it frustrates people and leaves terrible influence on society. Therefore, we need to evaluate those language, expose the dangerous side and appeal people to avoid using offensive language.

For assignment 2, our group chose a relevant task from SemEval-2020, which is an international workshop on semantic evaluation. The task is about identifying and categorizing offensive language on Twitter. There are three tasks in total:
A.	Offensive language identification:
based on all the data we have, to classify whether a tweet is offensive or not 
B.	Offense types categorization
for the offensive tweets we found in sub-task A, to categorize whether they are targeted insult and threats
C.	Offense target identification
for the targeted offensive tweets we found in sub-task B, to decide who these language target at: individual, group, or other?

To find the best model, we tried using three models (SVM, Naïve Bayes and Random Forest) in two Python libraries (NLTK and Scikit-Learn) and compared their precision, recall and F1-score. 
