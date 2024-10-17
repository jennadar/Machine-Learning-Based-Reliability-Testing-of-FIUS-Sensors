# -*- coding: utf-8 -*-
"""
Created on Fri Mar 15 11:10:03 2024

@author: jenny
"""

import pandas as pd
import numpy as np
from scipy.fft import fft

# Read ADC data from CSV file
data = pd.read_csv("D:/jenny/Documents/FAUS_Study/Sem3/Machine-Learning/spyder/adc_1m_hard_surface_test.csv")
data = data.iloc[:, 16:]

# Define window size and overlap
window_size = 100  # microseconds
overlap = 0  # no overlap for now
reading_index = 0

# Calculate the number of windows
num_windows = int(data.shape[0] / window_size)

# Initialize variables to store maximum echo information
max_echo_window = 0
max_echo_distance = 0

# Function to calculate distance from frequency
def freq_to_distance(freq, speed_of_sound):
    return speed_of_sound * (1 / (2 * freq))

# Speed of sound in air (m/s)
speed_of_sound = 343.0

# Function to apply rectangular window to a signal
def apply_rectangular_window(signal, window_size):
    window = np.ones(window_size)
    return signal * window

# Function to find peak of a signal
def find_peak(signal):
    fft_data = fft(signal)
    magnitude = np.abs(fft_data)
    peak_index = np.argmax(magnitude)
    return peak_index, magnitude[peak_index]

# Iterate over each row and process
for index, row in data.iterrows():
    # Get the signal (row)
    signal = row.values
    
    # Apply rectangular window
    windowed_signal = apply_rectangular_window(signal, window_size)
    
    # Calculate FFT
    fft_data = fft(windowed_signal)
    
    # Find peak
    peak_index, peak_magnitude = find_peak(fft_data)
    
    # Output results
    print(f"Row {index + 1}: Peak at index {peak_index}, Magnitude {peak_magnitude}")


# Iterate over each reading and find the first echo
for reading_index in range(data.shape[0] // window_size):
    # Initialize variables for each reading
    max_echo_distance_reading = 0
    max_echo_window_reading = 0
    
    for i in range(num_windows):
        start_index = reading_index * window_size + i * window_size
        end_index = start_index + window_size

        # Ensure end_index does not exceed the data length
        if end_index > data.shape[0]:
            end_index = data.shape[0]

        # Extract data for the current window
        window_data = data.iloc[start_index:end_index].values.squeeze()

        # Apply rectangle window
        windowed_data = window_data * np.ones_like(window_data)

        # Calculate FFT
        fft_data = fft(windowed_data)

        # Find frequency corresponding to maximum magnitude
        max_magnitude_index = np.argmax(np.abs(fft_data))
        max_frequency = max_magnitude_index / window_size

        # Convert frequency to distance
        distance = freq_to_distance(max_frequency, speed_of_sound)

        # Check if current window has maximum echo
        if distance > max_echo_distance_reading:
            max_echo_distance_reading = distance
            max_echo_window_reading = i + 1

    # Check if current reading has maximum echo
    if max_echo_distance_reading > max_echo_distance:
        max_echo_distance = max_echo_distance_reading
        max_echo_window = max_echo_window_reading + reading_index * num_windows

# Output results
print("Window with maximum echo:", max_echo_window)
print("Distance to maximum echo:", max_echo_distance, "meters")
