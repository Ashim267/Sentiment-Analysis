import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.preprocessing.sequence import pad_sequences
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
import numpy as np

def preprocess_text(train, test, tokenizer, max_seq_len=150):
    X_train = pad_sequences(tokenizer.texts_to_sequences(train["review"]), maxlen=max_seq_len)
    X_test = pad_sequences(tokenizer.texts_to_sequences(test["review"]), maxlen=max_seq_len)
    Y_train = tf.keras.utils.to_categorical(train["label"], num_classes=2)
    Y_test = tf.keras.utils.to_categorical(test["label"], num_classes=2)
    return X_train, X_test, Y_train, Y_test

def build_lstm_model(max_words=10000, max_seq_len=150):
    model = Sequential([
        Embedding(input_dim=max_words, output_dim=128, input_length=max_seq_len),
        Dropout(0.3),
        LSTM(64, return_sequences=True),
        LSTM(32),
        Dense(64, activation="relu"),
        Dropout(0.2),
        Dense(2, activation='softmax')
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

def plot_history(history):
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Acc')
    plt.plot(history.history['val_accuracy'], label='Val Acc', linestyle='--')
    plt.title('Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss', linestyle='--')
    plt.title('Loss')
    plt.legend()
    plt.tight_layout()
    plt.show()

def evaluate_model(model, X_test, Y_test):
    Y_pred = model.predict(X_test)
    Y_pred_class = np.argmax(Y_pred, axis=1)
    Y_test_class = np.argmax(Y_test, axis=1)
    print(classification_report(Y_test_class, Y_pred_class))
