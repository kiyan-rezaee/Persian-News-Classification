import pandas as pd
from hazm import Stemmer, Lemmatizer, word_tokenize

data = pd.read_csv('per.csv')

# preprocessing

with open('stopwords.txt') as stopwords_file:
    stopwords = stopwords_file.readlines()
stopwords = [line.replace('\n', '') for line in stopwords]

stemmer = Stemmer()
lem = Lemmatizer()

df = pd.DataFrame(columns=('title_body', 'category'))
for index, row in data.iterrows():
    title_body = row['Title'] + ' ' + row['Body']
    title_body_tokenized = word_tokenize(title_body)
    title_body_tokenized_filtered = [w for w in title_body_tokenized if not w in stopwords]
    title_body_tokenized_filtered_stemmed = [stemmer.stem(w) for w in title_body_tokenized_filtered]
    title_body_tokenized_filtered_lem = [lem.lemmatize(w).replace('#', ' ') for w in title_body_tokenized_filtered]
    df.loc[index] = {
        'title_body': ' '.join(title_body_tokenized_filtered_lem) + ' ' + ' '.join(title_body_tokenized_filtered_stemmed),
        'category': row['Category2'].replace('\n', '')
    }

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder

vectorizer = TfidfVectorizer(ngram_range=(1, 2))
vectorizer.fit(df['title_body'])
X = vectorizer.transform(df['title_body'])
le = LabelEncoder()
Y = le.fit_transform(df['category'])

# Model 

from sklearn.model_selection import train_test_split
from sklearn import svm

X_train, X_test, y_train, y_test = train_test_split(X, Y)
svmc = svm.SVC()
svmc.fit(X_train, y_train)

#print score : accuracy
print(svmc.score(X_test, y_test))

# Evaluation of model

from sklearn.metrics import classification_report, confusion_matrix

y_pred = svmc.predict(X_test)
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))