import pandas as pd
import numpy as np

# read the dataset based on its format
test_data_file = 'test_data.csv'
train_data_file = 'train_data.csv'
# how to read the test and train data
test_data_df = pd.read_csv(test_data_file, header=None, delimiter='\t', quoting=3)
test_data_df.columns = ['Text']
train_data_df = pd.read_csv(train_data_file, header=None, delimiter='\t', quoting=3)
train_data_df.columns = ['Sentiment', 'Text']

print(train_data_df.shape)
print(test_data_df.shape)

# preparing the corpus
import re, nltk
from sklearn.feature_extraction.text import CountVectorizer
from nltk.stem.porter import PorterStemmer

stemmer = PorterStemmer()
def stem_tokens(tokens, stemmer):
    stemmed = []
    for item in tokens:
        stemmed.append(stemmer.stem(item))
    return stemmed

def tokenize(text):
    # remove non letters
    text = re.sub("[^a-zA-Z]", " ", text)
    # tokenize
    tokens = nltk.word_tokenize(text)
    # stem
    stems = stem_tokens(tokens, stemmer)
    return stems

vectorizer = CountVectorizer(
    analyzer='word',
    tokenizer=tokenize,
    lowercase=True,
    stop_words='english',
    max_features=85
)

corpus_data_features = vectorizer.fit_transform(train_data_df.Text.tolist() + test_data_df.Text.tolist())
corpus_data_features_nd = corpus_data_features.toarray()
print corpus_data_features_nd.shape

"""
# to get the vocab words
vocabs = vectorizer.get_feature_names()

# sum up the counts of each vocab words
dist = np.sum(corpus_data_features_nd, axis=0)

# for each, print vocab word and the number of times it appears in dataset
for tag, count in zip(vocabs, dist):
    print count, tag
"""

# corpus_data_features_nd contains all of our originak train and test data, we need to exclude the unlabeled test entries
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    corpus_data_features_nd[0:len(train_data_df)],
    train_data_df.Sentiment,
    train_size=0.85,
    random_state=1234
)

# train a classifier
from sklearn.linear_model import LogisticRegression
log_model = LogisticRegression()
log_model = log_model.fit(X=X_train, y=y_train)

y_pred = log_model.predict(X_test)

from sklearn.metrics import classification_report
print classification_report(y_test, y_pred)