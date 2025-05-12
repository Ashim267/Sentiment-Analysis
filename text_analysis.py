from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.probability import FreqDist
from nltk.corpus import stopwords
import string

def tokenize_reviews(reviews, batch_size=1000):
    lemmatizer = WordNetLemmatizer()
    all_tokens = []
    
    for start in range(0, reviews.shape[0], batch_size):
        batch = reviews.iloc[start:start+batch_size]
        for review in batch:
            tokens = word_tokenize(review)
            lemmatized = [lemmatizer.lemmatize(token) for token in tokens]
            all_tokens.extend(lemmatized)
    
    return all_tokens

def clean_tokens(tokens):
    unwanted = {"'s", "'ll", "'d", "'re", "'m", "''", "``"}
    stop_words = set(stopwords.words("english")).union(string.punctuation, unwanted)
    return [token for token in tokens if token.lower() not in stop_words]

def get_frequent_words(tokens, top_k=30):
    freq_dist = FreqDist(tokens)
    return freq_dist.most_common(top_k)
