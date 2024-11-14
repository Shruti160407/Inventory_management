import pandas as pd
import numpy as np
from scipy.signal import butter, lfilter
import wfdb  # If you're using WFDB format

# Load ECG data from CSV or WFDB
def load_data(filepath, file_type='csv'):
    if file_type == 'csv':
        # Load data from CSV (if your data is in CSV format)
        return pd.read_csv(filepath)
    elif file_type == 'wfdb':
        # Load data from WFDB format (using wfdb library)
        record = wfdb.rdrecord(filepath)  # Read the record
        return record.p_signal  # Return the ECG signal as a numpy array
    else:
        raise ValueError("Unsupported file type. Use 'csv' or 'wfdb'.")

# Low-pass filter to clean ECG signal
def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter(order, cutoff / (0.5 * fs), btype='low', analog=False)
    return lfilter(b, a, data)

# Normalize ECG data
def normalize_ecg(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))

# Segment ECG signals into P, QRS, and T waves (dummy function for now)
def segment_ecg(data):
    # Simulating segmentation (real implementation depends on the data)
    return np.array_split(data, 3)

# Process data: clean, normalize, and segment
def preprocess_data(filepath, file_type='csv'):
    # Load ECG data from file
    ecg_data = load_data(filepath, file_type)
    
    # Assuming 'ECG' column exists for CSV data or raw ECG signal from WFDB
    if file_type == 'csv':
        ecg_signal = ecg_data['ECG']  # Assuming the column is named 'ECG'
    else:
        ecg_signal = ecg_data  # If WFDB, it's already a numpy array

    # Apply low-pass filter to remove noise
    cleaned_data = butter_lowpass_filter(ecg_signal, cutoff=50, fs=1000)
    
    # Normalize the ECG signal
    normalized_data = normalize_ecg(cleaned_data)
    
    # Segment the ECG signal (dummy segmentation for now)
    segmented_data = segment_ecg(normalized_data)
    
    return np.array(segmented_data)

# Example usage
# Preprocess ECG data (replace 'data/ecg_data.csv' with your dataset path)
# ecg_data = preprocess_data('data/ecg_data.csv', file_type='csv')  # If using CSV
# ecg_data = preprocess_data('data/your_ecg_record', file_type='wfdb')  # If using WFDB
