# lstm-abnormal-process (Linux)
LSTM model trained on normal and abnormal process sequences.
Data (process sequences):
Normal: 6000
Abnormal: 6000

Usage:
python3 classification.py --test_data test.csv --model_path process_classifier_lstm.h5 --tokenizer_path tokenizer.pkl --max_sequence_length 10
