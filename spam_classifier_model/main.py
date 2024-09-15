# Importing essential libraries
import numpy as np
import pandas as pd
import sys

# Importing essential libraries for visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Importing essential libraries for performing NLP
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")



# Loading the dataset
df = pd.read_csv('./Spam_SMS_Collection.txt', sep='\t', names=['label', 'message'])

df['label'] = df['label'].map({'ham': 0, 'spam': 1})

df.drop_duplicates(inplace = True)

spam = df[df['label']==1]
ham =  df[df['label']==0]

count = int(ham.shape[0]-spam.shape[0])

for i in range(0, count):
    record = pd.DataFrame(spam.sample(n=1))
    df = pd.concat([df, record])

#Creating new feature 'word_count'
df['word_count'] = df['message'].apply(lambda x: len(x.split()))


# Creating feature 'contains_currency_symbol'
def currency(x):
    currency_symbols = ['€', '$', '¥', '£', '₹']
    for i in currency_symbols:
        if i in x:
            return 1
    return 0

df['contains_currency_symbol'] = df['message'].apply(currency)


# Creating feature 'contains_number'
def numbers(x):
    for i in x:
        if ord(i)>=48 and ord(i)<=57:
            return 1
    return 0

df['contains_number'] = df['message'].apply(numbers)


# Cleaning the messages
corpus = []
wnl = WordNetLemmatizer()

for sms_string in list(df.message):

    # Cleaning special character from the sms
    message = re.sub(pattern='[^a-zA-Z]', repl=' ', string=sms_string)

    # Converting the entire sms into lower case
    message = message.lower()

    # Tokenizing the sms by words
    words = message.split()

    # Removing the stop words
    filtered_words = [word for word in words if word not in set(stopwords.words('english'))]

    # Lemmatizing the words
    lemmatized_words = [wnl.lemmatize(word) for word in filtered_words]

    # Joining the lemmatized words
    message = ' '.join(lemmatized_words)

    # Building a corpus of messages
    corpus.append(message)


# Creating the Bag of Words model

tfidf = TfidfVectorizer(max_features=500)
vectors = tfidf.fit_transform(corpus).toarray()
feature_names = tfidf.get_feature_names_out()

# Extracting independent and dependent variables from the dataset
X = pd.DataFrame(vectors, columns=feature_names)
y = df['label']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


rf = RandomForestClassifier(n_estimators=20)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)


    
def predict_spam(sample_message):
    sample_message = re.sub(pattern='[^a-zA-Z]',repl=' ', string = sample_message)
    sample_message = sample_message.lower()
    sample_message_words = sample_message.split()
    sample_message_words = [word for word in sample_message_words if not word in set(stopwords.words('english'))]
    final_message = [wnl.lemmatize(word) for word in sample_message_words]
    final_message = ' '.join(final_message)

    temp = tfidf.transform([final_message]).toarray()
    return rf.predict(temp)

if __name__ == '__main__':

    # Prediction 1
    sample_message = input("Enter your meassage")

    if predict_spam(sample_message):
        print('Gotcha! This is a SPAM message.')
    else:
        print('This is a HAM (normal) message.')


