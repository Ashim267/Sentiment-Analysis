
# Sentiment Analysis on Text Reviews (LSTM + Naive Bayes)

This project performs **binary sentiment classification** on labeled customer reviews using both classical and deep learning approaches. The pipeline includes data preprocessing, exploratory analysis, lexicon-based sentiment classification with **Naive Bayes**, and a **deep learning model using LSTM** with TensorFlow/Keras.

---

## 📁 Project Structure

├── main.py # Main script running preprocessing, analysis, and modeling
├── data_utils.py # Data preprocessing utilities
├── text_analysis.py # Tokenization and lexical analysis
├── model_utils.py # LSTM model creation and evaluation
└── README.md

````

---

## 💡 Features

- Converts raw `.txt` files to structured `.csv`
- Cleans and tokenizes reviews using **NLTK**
- Performs frequency and stopword analysis
- Sentiment classification with:
  - Naive Bayes (using `movie_reviews` corpus)
  - LSTM deep neural network (Keras)
- Visualizes model performance over epochs

---


### Key Libraries

* `tensorflow`
* `nltk`
* `pandas`
* `numpy`
* `matplotlib`
* `scikit-learn`
* `seaborn`
* `tqdm`
* `wordcloud`
* `spacy` *(optional)*

Also download required NLTK datasets:

```python
import nltk
nltk.download('punkt')
nltk.download('opinion_lexicon')
nltk.download('movie_reviews')
nltk.download('stopwords')
nltk.download('wordnet')
```

---

## 📊 Sample Output

* Top 30 frequent words in each label class
* Confusion Matrix and Accuracy of Naive Bayes
* LSTM performance:

  * Accuracy & Loss plots
  * Classification report on test set

---

## 🔍 Future Improvements

* Add more preprocessing like emoji stripping, contractions
* Add BERT or transformer-based models
* Support multiclass classification



