import os
import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Adjust these paths as needed.
# The IMDb dataset can be obtained at: https://ai.stanford.edu/~amaas/data/sentiment/
# After downloading and extracting, you should have directories like:
# aclImdb/train/pos, aclImdb/train/neg, aclImdb/test/pos, aclImdb/test/neg
train_pos_dir = "aclImdb/train/pos"
train_neg_dir = "aclImdb/train/neg"
test_pos_dir = "aclImdb/test/pos"
test_neg_dir = "aclImdb/test/neg"

def load_reviews(directory):
    texts = []
    for filename in os.listdir(directory):
        if filename.endswith(".txt"):
            with open(os.path.join(directory, filename), "r", encoding="utf-8") as f:
                text = f.read().strip()
                texts.append(text)
    return texts

def preprocess_text(texts):
    cleaned = []
    for t in texts:
        t = t.lower()
        t = re.sub(r"[^a-z0-9\s]", "", t)
        cleaned.append(t)
    return cleaned

# Load data
train_pos = load_reviews(train_pos_dir)
train_neg = load_reviews(train_neg_dir)
test_pos = load_reviews(test_pos_dir)
test_neg = load_reviews(test_neg_dir)

train_texts = train_pos + train_neg
train_labels = np.array([1]*len(train_pos) + [0]*len(train_neg))

test_texts = test_pos + test_neg
test_labels = np.array([1]*len(test_pos) + [0]*len(test_neg))

# Preprocess text
train_texts = preprocess_text(train_texts)
test_texts = preprocess_text(test_texts)

# Vectorize
vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1,2))
X_train = vectorizer.fit_transform(train_texts)
X_test = vectorizer.transform(test_texts)

# Train classifier
clf = LogisticRegression(max_iter=1000)
clf.fit(X_train, train_labels)

# Evaluate
predictions = clf.predict(X_test)
accuracy = accuracy_score(test_labels, predictions)
print("Test accuracy:", accuracy)
