import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers import Adam # type: ignore
from model import build_multimodal_model
from data_processing import preprocess_data

# Load and preprocess the data
ecg_data = preprocess_data('data/ecg_data.csv')

# Simulating additional data like temporal and structured data
temporal_data = np.random.rand(len(ecg_data), 100, 1)  # Example temporal data
structured_data = np.random.rand(len(ecg_data), 10)    # Example structured data (age, medical history)

# Labels (1 for SCA risk, 0 for no risk)
labels = np.random.randint(0, 2, len(ecg_data))

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split([ecg_data, temporal_data, structured_data], labels, test_size=0.2)

# Define the model
model = build_multimodal_model((300, 1), (100, 1), (10,))

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2)

# Evaluate the model
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {test_accuracy}")

# Save the model for future use
model.save('sca_prediction_model.h5')
