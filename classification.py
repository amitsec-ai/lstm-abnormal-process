import argparse
import numpy as np
import pandas as pd
import tensorflow as tf
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences

def load_model_and_tokenizer(model_path, tokenizer_path):
    model = tf.keras.models.load_model(model_path)
    with open(tokenizer_path, 'rb') as f:
        tokenizer = pickle.load(f)
    return model, tokenizer

def preprocess_test_data(test_sequences, tokenizer, max_sequence_length):
    sequences = tokenizer.texts_to_sequences(test_sequences)
    padded_sequences = pad_sequences(sequences, maxlen=max_sequence_length, padding='post')
    return padded_sequences

def classify_test_data(model, tokenizer, test_data, max_sequence_length):
    X_test = preprocess_test_data(test_data, tokenizer, max_sequence_length)
    
    predictions = model.predict(X_test)
    
    binary_predictions = (predictions > 0.5).astype(int)
    
    return binary_predictions.flatten()

def main():
    parser = argparse.ArgumentParser(description="Classify test data using a trained LSTM model.")
    parser.add_argument('--test_data', type=str, required=True, help="Path to the CSV file containing test sequences.")
    parser.add_argument('--model_path', type=str, default='process_classifier_lstm.h5', help="Path to the trained model.")
    parser.add_argument('--tokenizer_path', type=str, default='tokenizer.pkl', help="Path to the tokenizer.")
    parser.add_argument('--max_sequence_length', type=int, default=20, help="Maximum length of sequences to pad/truncate.")
    args = parser.parse_args()

    df = pd.read_csv(args.test_data)
    test_sequences = df['ExeNameSequence'].apply(eval).tolist() 

    model, tokenizer = load_model_and_tokenizer(args.model_path, args.tokenizer_path)

    predictions = classify_test_data(model, tokenizer, test_sequences, args.max_sequence_length)

    df['Prediction'] = predictions
    df['Prediction'] = df['Prediction'].map({0: 'Normal', 1: 'Abnormal'}) 
    df.to_csv('test_predictions.csv', index=False)
    print("Predictions saved to 'test_predictions.csv'.")

if __name__ == "__main__":
    main()
