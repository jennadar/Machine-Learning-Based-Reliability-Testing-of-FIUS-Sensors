# -*- coding: utf-8 -*-
"""
Created on Tue Mar 26 13:42:49 2024

@author: jenny
"""

import pandas as pd
import numpy as np
from scipy.fft import fft
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import layers, models
from scipy.signal import find_peaks
from sklearn.metrics import confusion_matrix, recall_score, f1_score
from tensorflow.keras import models, layers, regularizers,  callbacks
import os
import matplotlib.pyplot as plt

#***************** Variable defination *********************

X = []
y = []

# Directory containing the files
directory = "D:/jenny/Documents/FAUS_Study/Sem3/Machine-Learning/adc_plotting/Data_set"

# Define window size and overlap
window_size = 64  # microseconds
overlap = 0  # no overlap for now

# Initialize arrays to store loss and accuracy for plotting learning curve
train_loss = []
val_loss = []
train_acc = []
val_acc = []

# Initialize variables for hits and fails
hits = 0
fails = 0
false_positives = 0
true_negatives = 0
# Initialize confusion matrix
conf_matrix = [[0, 0], [0, 0]]  # [[True Positive, False Negative], [False Positive, True Negative]]

# Initialize arrays to store window data and labels
# X = [] 
# y = []
velocity_of_sound = 343
  # meters per second (for air at room temperature)
time_delay_microseconds = 2  # microseconds

# # Apply feature extraction to the data and create feature matrix X and labels y
# X = []
# y = []

# manual_distance_dML = 0

data= None


#********************************************

# ************************** Function Definations **********************

# Define a function to calculate distance based on frequency
def calculate_distance(position_of_max_peak,velocity_of_sound):
    # Calculate the distance of the maximum peak
    distance_of_max_peak = 0.5 * position_of_max_peak * velocity_of_sound
    return distance_of_max_peak
 
    
def apply_window_and_fft(row):
    # Apply your window function here
    
    windowed_row = row * np.ones_like(row)  # Example: Using rectangle window
    # windowed_row = row * np.hamming(len(row))
    fft_data = np.fft.fft(windowed_row)
    
    # Find the index of the maximum magnitude component in the FFT data
    max_magnitude_index = np.argmax(np.abs(fft_data))
    
    # Calculate the corresponding frequency
    sampling_rate = 125000000  # Example: Sampling rate (adjust according to your data)
    max_frequency = max_magnitude_index * sampling_rate / len(fft_data)
    
    # Construct a frequency domain signal with only the component corresponding to the maximum frequency
    max_freq_signal = np.zeros_like(fft_data)
    max_freq_signal[max_magnitude_index] = fft_data[max_magnitude_index]
    
    # Perform inverse FFT on the maximum frequency component
    ifft_result = np.fft.ifft(max_freq_signal)
    
    # Get only the real part of the inverse FFT result
    ifft_real_part = np.real(ifft_result)
    
    # Find peaks in the real part of the inverse FFT result
    peaks_indices, _ = find_peaks(ifft_real_part)
    
    # Get the peak values
    peaks_values = ifft_real_part[peaks_indices]
    
    # Check if peaks_values is empty
    if len(peaks_values) == 0:
        # Handle the case where no peaks are found
        return 0
    
    # Find the index of the maximum peak value
    max_peak_index = np.argmax(peaks_values)
    
    # Get the maximum peak value and its index
    max_peak_value = peaks_values[max_peak_index]
    max_peak_index_absolute = peaks_indices[max_peak_index]
    
    # Calculate the position of the maximum peak in terms of time delay
    position_of_max_peak = max_peak_index_absolute * (2 * time_delay_microseconds) / len(ifft_real_part)
    # position_of_max_peak = max_peak_index_absolute * (time_delay_microseconds)*velocity_of_sound
    
    # Calculate the distance corresponding to the maximum frequency
    max_distance = calculate_distance(position_of_max_peak, velocity_of_sound)
    
    return position_of_max_peak


# Define your feature extraction function
def extract_features(row,manual_distance_dML):
    # Apply windowing and FFT
    max_distance = apply_window_and_fft(row)
    max_distance = round(max_distance, 2)
    deviation = abs(max_distance - manual_distance_dML)
    return max_distance, deviation

# Define a function to plot the learning curve
def plot_learning_curve(history):
    # Plot training & validation accuracy values
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='lower right')
    plt.show()

    # Plot training & validation loss values
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper right')
    plt.show()


#**************************** FILE name extraction **********
def process_file(file_path):
    
    #Apply feature extraction to the data and create feature matrix X and labels y
    
    # Extract the filename from the file path
    file_name = os.path.basename(file_path)

    file_name_without_extension = os.path.splitext(file_name)[0]
    parts = file_name_without_extension.split("_")
    
    for part in parts:
        if "cm" in part:
            manual_distance_dML = float(part.replace("cm", "")) / 100  # Convert cm to meters
            break
        elif "m" in part:
            manual_distance_dML = float(part.replace("m", ""))  # Already in meters
            break
    
    #Print the extracted distance
    if manual_distance_dML is not None:
        print("Distance for", file_name, ":", manual_distance_dML, "meters")
    else:
        print("Distance information not found in file name for", file_name)
        
    data = pd.read_csv(file_path)
    data = data.iloc[:, 16:]
    for index, row in data.iterrows():
        # Apply windowing and FFT function to each row
        max_distance = apply_window_and_fft(row)
        max_distance = float(round(max_distance, 2))
        # print("Max distance for row", index, ":", max_distance)
        

    for index, row in data.iterrows():
        max_distance, deviation = extract_features(row,manual_distance_dML)
        X.append([max_distance, deviation])
        # Labeling based on deviation threshold
        if deviation < 0.01:
            y.append(1)  # Hit
        else:
            y.append(0)  # Fail
    
    return X, y
#**************************** FILE name extraction **********
def neural_network(X,y):
    # Convert to numpy arrays
    X = np.array(X)
    # X = np.array(X).reshape(-1, 1)
    y = np.array(y)

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Define and train a neural network model
    # model = models.Sequential([
    #     layers.Dense(64, activation='relu', input_shape=(2,)),
    #     layers.Dense(32, activation='relu'),
    #     layers.Dense(1, activation='sigmoid')
    # ])
    
    '+++Increased model complexity+++'
#     model = models.Sequential([
#         layers.Dense(128, input_shape=(2,), activation='relu'),
#         layers.Dropout(0.5),  # Dropout regularization
#         layers.BatchNormalization(),  # Batch normalization layer
#         layers.Dense(64, activation='relu'),
#         layers.Dropout(0.5),  # Dropout regularization
#         layers.BatchNormalization(),  # Batch normalization layer
#         layers.Dense(32, activation='relu'),
#         layers.Dropout(0.5),  # Dropout regularization
#         layers.BatchNormalization(),  # Batch normalization layer
#         layers.Dense(1, activation='sigmoid')
# ])
    model = models.Sequential([
    layers.Dense(128, activation='relu', input_shape=(2,)),  # Increased neurons in the first hidden layer
    layers.Dense(64, activation='relu'),                     # Additional hidden layer with increased neurons
    layers.Dense(32, activation='relu'),                     # Original hidden layer
    layers.Dense(1, activation='sigmoid')
])

    # Define a learning rate scheduler
    lr_scheduler = callbacks.ReduceLROnPlateau(factor=0.5, patience=3, verbose=1)

    # optimizer = tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9)
    # optimizer_adam = tf.keras.optimizers.Adam(learning_rate=0.001)
    
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    
    # Define callback for early stopping
    early_stopping = callbacks.EarlyStopping(monitor='val_loss', patience=5)


    history = model.fit(X_train, y_train, epochs=20, batch_size=32, verbose=1, callbacks=[lr_scheduler],validation_data=(X_test, y_test))
    #model.fit(X_train, y_train, epochs=20, batch_size=32, verbose=1)
    # Evaluate the model on the testing data
    plot_learning_curve(history)
    
    # Save the model
    model.save("ml_model.h5")
    
    test_loss, test_acc = model.evaluate(X_test, y_test)
    print('Test accuracy:', test_acc)

    # Make predictions on new data
    predictions = model.predict(X_test)

    # Convert predicted probabilities to binary predictions
    binary_predictions = (predictions > 0.5).astype(int)

    # Calculate recall
    recall = recall_score(y_test, binary_predictions, zero_division=0)

    # Calculate F1 score
    f1 = f1_score(y_test, binary_predictions, zero_division=0)

    print('Recall:', recall)
    print('F1 Score:', f1)

    # Calculate confusion matrix
    conf_matrix = confusion_matrix(y_test, binary_predictions)

    # Print confusion matrix
    print("Confusion Matrix:")
    print(conf_matrix)
# Iterate over each file in the directory
for filename in os.listdir(directory):
    if filename.endswith(".csv"):
        file_path = os.path.join(directory, filename)
        X, y=process_file(file_path)
    
neural_network(X,y)