import tensorflow as tf
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, LSTM, Input, concatenate # type: ignore
from tensorflow.keras.models import Model # type: ignore

# CNN model for ECG data
def cnn_module(input_shape):
    inputs = Input(shape=input_shape)
    x = Conv1D(32, kernel_size=3, activation='relu')(inputs)
    x = MaxPooling1D(pool_size=2)(x)
    x = Flatten()(x)
    return inputs, x

# LSTM model for temporal data
def lstm_module(input_shape):
    inputs = Input(shape=input_shape)
    x = LSTM(64, return_sequences=True)(inputs)
    x = LSTM(32)(x)
    return inputs, x

# MLP for structured data (e.g., demographics, medical history)
def mlp_module(input_shape):
    inputs = Input(shape=input_shape)
    x = Dense(64, activation='relu')(inputs)
    x = Dense(32, activation='relu')(x)
    return inputs, x

# Define the full multimodal model
def build_multimodal_model(ecg_shape, temporal_shape, structured_shape):
    ecg_input, ecg_output = cnn_module(ecg_shape)
    temporal_input, temporal_output = lstm_module(temporal_shape)
    structured_input, structured_output = mlp_module(structured_shape)

    combined = concatenate([ecg_output, temporal_output, structured_output])
    output = Dense(1, activation='sigmoid')(combined)

    model = Model(inputs=[ecg_input, temporal_input, structured_input], outputs=output)
    return model
