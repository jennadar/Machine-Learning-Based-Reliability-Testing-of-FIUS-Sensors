import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

# Load the CSV file into a DataFrame
df = pd.read_csv('adc_1m_hard_surface_2.csv')
#df = pd.read_csv('adc_soft_surface_standing.csv')

# Transpose the DataFrame so that each row represents a time step
df_transposed = df.iloc[:, 16:1600].transpose()

# Get the number of time steps
num_time_steps = len(df_transposed.columns)

# Define sampling frequency and time period
sampling_frequency = 2600  # Assuming data is sampled at unit intervals
time_period = 1 / sampling_frequency

# Apply Hann windowing function to the ADC readings before FFT
window = np.ones_like(len(df_transposed.columns))
df_transposed_windowed = df_transposed * window
print("total row:",df_transposed_windowed)
# Plot each row as a separate set of readings and compute FFT
for index, row in df_transposed_windowed.iterrows():
	# Compute FFT of windowed ADC values
	fft_values = np.fft.fft(row)
	frequencies = np.fft.fftfreq(len(row), time_period)
	magnitudes = np.abs(fft_values)
	fig, (ax1, ax2) = plt.subplots(2, 1)
	ax1.plot(range(num_time_steps), row)
	ax1.set_xlabel('Time')
	ax1.set_ylabel('ADC Value')
	ax1.set_title('ADC Values over Time')

	ax2.plot(frequencies, magnitudes)
	ax2.set_xlabel('Frequency')
	ax2.set_ylabel('Magnitude')
	ax2.set_title('FFT Spectrum')

	plt.tight_layout()
	plt.show()