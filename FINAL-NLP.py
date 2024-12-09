#IMPORTING

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.preprocessing.sequence import pad_sequences
import nltk
nltk.download('opinion_lexicon')
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.util import ngrams
from nltk.probability import FreqDist
from nltk.corpus import stopwords, opinion_lexicon, product_reviews_1, product_reviews_2, wordnet, movie_reviews
from nltk.stem import WordNetLemmatizer
from nltk.stem.lancaster import LancasterStemmer
from nltk.classify import NaiveBayesClassifier
from nltk.metrics import ConfusionMatrix
from nltk.classify.util import accuracy
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import string
import random
from wordcloud import WordCloud
import spacy
from tqdm import tqdm

##

import csv
import os

# Function to process a TXT file and save it as a CSV
def process_txt_to_csv(input_txt, output_csv):
    if not os.path.exists(output_csv):  # Check if the output CSV already exists
        if os.path.exists(input_txt):  # Check if the input TXT file exists
            with open(output_csv, "w", newline="", encoding="utf-8") as csv_file:  # Open CSV for writing
                writer = csv.writer(csv_file)
                writer.writerow(["label", "title", "review"])  # Header row

                # Open TXT for reading with utf-8 encoding
                try:
                    with open(input_txt, "r", encoding="utf-8") as text_file:
                        for line in text_file:
                            review_data = line.split(sep=" ", maxsplit=1)
                            label = review_data[0] if len(review_data) > 1 else None
                            review_data = review_data[1].split(":", maxsplit=1) if len(review_data) > 1 else ["", ""]
                            title = review_data[0].strip() if len(review_data) > 1 else None
                            review = review_data[1].strip() if len(review_data) > 1 else None

                            writer.writerow([label, title, review])  # Write row to CSV

                    print(f"Processed data saved to {output_csv}")
                except UnicodeDecodeError as e:
                    print(f"Error decoding {input_txt}: {e}")
        else:
            print(f"File {input_txt} not found!")
    else:
        print(f"File {output_csv} already exists. Skipping processing.")

# Process train.txt into trainP.csv
process_txt_to_csv("train.txt", "trainP.csv")

# Process test.txt into testP.csv
process_txt_to_csv("test.txt", "testP.csv")

## train

train_1 = pd.read_csv("trainP.csv")
test_1 = pd.read_csv("testP.csv")
train_1 = train_1.sample(frac=1).reset_index(drop=True)
test_1 = test_1.sample(frac=1).reset_index(drop=True)

train_1 = train_1.sample(frac=1).reset_index(drop=True)

#head and info
print(train_1.head())
train_1.info()

# Load and shuffle data
train_1 = pd.read_csv("trainP.csv").sample(frac=1).reset_index(drop=True)
test_1 = pd.read_csv("testP.csv").sample(frac=1).reset_index(drop=True)

# Convert reviews to lowercase
train_1["review"] = train_1["review"].str.lower()
test_1["review"] = test_1["review"].str.lower()

# Separate reviews by labels
df_label1 = train_1[train_1["label"] == "__label__1"]
df_label2 = train_1[train_1["label"] == "__label__2"]

print(f"Label 1 reviews: {df_label1.shape[0]}")
print(f"Label 2 reviews: {df_label2.shape[0]}")

from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from tqdm import tqdm



def tokenize_reviews(reviews, batch_size=1000):
    lemmatizer = WordNetLemmatizer()
    all_tokens = []
    
    for start in tqdm(range(0, reviews.shape[0], batch_size), desc="Tokenizing reviews"):
        batch = reviews.iloc[start:start+batch_size]
        for review in batch:
            tokens = word_tokenize(review)
            lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens]
            all_tokens.extend(lemmatized_tokens)
    
    return all_tokens

# Tokenize reviews
label1_tokens = tokenize_reviews(df_label1["review"])
label2_tokens = tokenize_reviews(df_label2["review"])

# Remove stopwords and unwanted tokens
unwanted_tokens = {"'s", "'ll", "'d", "'re", "'m", "''", "``"}
stop_words = set(stopwords.words("english")).union(string.punctuation, unwanted_tokens)

def clean_tokens(tokens):
    return [token for token in tokens if token.lower() not in stop_words]

label1_tokens = clean_tokens(label1_tokens)
label2_tokens = clean_tokens(label2_tokens)

# Unique tokens and frequencies
unique_tokens_label1 = set(label1_tokens)
unique_tokens_label2 = set(label2_tokens)

print(f"Unique words: {len(unique_tokens_label1) + len(unique_tokens_label2)}")

freq_label1 = FreqDist(label1_tokens)
freq_label2 = FreqDist(label2_tokens)

# Display top-k frequent tokens
top_k = 30
print(f"Top {top_k} tokens in label 1: {freq_label1.most_common(top_k)}")
print(f"Top {top_k} tokens in label 2: {freq_label2.most_common(top_k)}")

# Sentiment analysis using Naive Bayes
opinion_words = opinion_lexicon.words()
documents = [(list(movie_reviews.words(fileid)), category)
             for category in movie_reviews.categories()
             for fileid in movie_reviews.fileids(category)]

def extract_features(document):
    words = set(document)
    return {f"contains({word})": (word in words) for word in opinion_words}

# Create feature sets
features = [(extract_features(doc), label) for doc, label in documents]
random.seed(42)
random.shuffle(features)

train_set, test_set = features[:1500], features[1500:]
classifier = NaiveBayesClassifier.train(train_set)

# Evaluate model
test_true_labels = [label for (_, label) in test_set]
test_pred_labels = [classifier.classify(fs) for (fs, _) in test_set]

cm = ConfusionMatrix(test_true_labels, test_pred_labels)
print(f"Confusion Matrix:\n{cm}")

# Metrics calculation
true_positives = sum(1 for x, y in zip(test_true_labels, test_pred_labels) if x == y == "pos")
false_positives = sum(1 for x, y in zip(test_true_labels, test_pred_labels) if x == "neg" and y == "pos")
false_negatives = sum(1 for x, y in zip(test_true_labels, test_pred_labels) if x == "pos" and y == "neg")
true_negatives = sum(1 for x, y in zip(test_true_labels, test_pred_labels) if x == y == "neg")

print(f"True Positive: {true_positives}")
print(f"False Positive: {false_positives}")
print(f"True Negative: {true_positives}")
print(f"False Negative: {false_negatives}")

precision = true_positives / (true_positives + false_positives)
recall = true_positives / (true_positives + false_negatives)
f1 = 2 * (precision * recall) / (precision + recall)
accuracy_score = accuracy(classifier, test_set)

print(f"Precision: {precision:.3f}")
print(f"Recall: {recall:.3f}")
print(f"F1-Score: {f1:.3f}")
print(f"Accuracy: {accuracy_score:.3f}")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
import tensorflow as tf

# Ensure that SentimentIntensityAnalyzer is imported only if needed
from nltk.sentiment import SentimentIntensityAnalyzer

# Create a copy of train and test dataframes, limiting data size for efficiency
train = train_1.copy().iloc[:300000]
test = test_1.copy().iloc[:300000]

# Convert labels to binary format
train["label"] = train["label"].str[-1].astype(int).apply(lambda x: 0 if x == 1 else 1)
test["label"] = test["label"].str[-1].astype(int).apply(lambda x: 0 if x == 1 else 1)

# Set tokenizer parameters
MAX_WORDS = 10000
MAX_SEQ_LEN = 150

tokenizer = Tokenizer(num_words=MAX_WORDS)
tokenizer.fit_on_texts(train["review"])

# Tokenize and pad sequences
X_train = pad_sequences(tokenizer.texts_to_sequences(train["review"]), maxlen=MAX_SEQ_LEN)
X_test = pad_sequences(tokenizer.texts_to_sequences(test["review"]), maxlen=MAX_SEQ_LEN)

# Convert labels to categorical format
Y_train = tf.keras.utils.to_categorical(train["label"], num_classes=2)
Y_test = tf.keras.utils.to_categorical(test["label"], num_classes=2)

# Define LSTM model
lstm_model = Sequential([
    Embedding(input_dim=MAX_WORDS, output_dim=128, input_length=MAX_SEQ_LEN),
    Dropout(0.3),
    LSTM(64, return_sequences=True),
    LSTM(32),
    Dense(64, activation="relu"),
    Dropout(0.2),
    Dense(2, activation='softmax')
])

# Compile the model
lstm_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Train the model
history = lstm_model.fit(
    X_train, Y_train,
    validation_split=0.2,
    epochs=10,
    batch_size=64
)

# Model predictions
Y_pred = lstm_model.predict(X_test)
Y_pred_class = np.argmax(Y_pred, axis=1)
Y_test_class = np.argmax(Y_test, axis=1)

# Classification report
print("Classification Report: ")
print(classification_report(Y_test_class, Y_pred_class))

# Accuracy plot
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy', linestyle='--')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

# Loss plot
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss', linestyle='--')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()






##


# %%
