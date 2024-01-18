import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

import nltk
nltk.download('punkt')
nltk.download('stopwords')

stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

def preprocess_text(text):
    if text:
        text = str(text)
        words = word_tokenize(text)
        words = [stemmer.stem(word.lower()) for word in words if word.isalnum() and word.lower() not in stop_words]
        return ' '.join(words)
    else:
        return ''

train_data = pd.read_csv("C:\\Users\\Lenovo\\Downloads\\twitter_training.csv")
validation_data = pd.read_csv("C:\\Users\\Lenovo\\Downloads\\twitter_validation.csv")

data = pd.concat([train_data, validation_data], axis=0, ignore_index=True)

print(data.columns)

text_column_name = 'text'
data['cleaned_text'] = data[text_column_name].apply(preprocess_text)
data = data[data['cleaned_text'] != '']
data['sentiment'].fillna(value='Unknown', inplace=True)

vectorizer = CountVectorizer(max_features=5000)
X = vectorizer.fit_transform(data['cleaned_text']).toarray()
y = data['sentiment']

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
X_train = pd.DataFrame(X_train).fillna(0).values

clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

test_data = pd.read_csv(r"C:\Users\Lenovo\Desktop\Prodigy Infotech\test_data.csv")

print(test_data.columns)

test_text_column_name = 'text'
test_data['cleaned_text'] = test_data[test_text_column_name].apply(preprocess_text)
test_data = test_data[test_data['cleaned_text'] != '']
test_data['sentiment'].fillna(value='Unknown', inplace=True)

X_test = vectorizer.transform(test_data['cleaned_text']).toarray()
X_test = pd.DataFrame(X_test).fillna(0).values

y_test = test_data['sentiment']
y_pred_test = clf.predict(X_test)

accuracy_test = accuracy_score(y_test, y_pred_test)
print(f"Test Accuracy: {accuracy_test}")
