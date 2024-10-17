import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import customtkinter
from MLP import manual_distance_dML, test_acc

# Set appearance mode and color theme for the GUI
customtkinter.set_appearance_mode("System")
customtkinter.set_default_color_theme("blue")

class App(customtkinter.CTk):
    def __init__(self):
        super().__init__()
        self.title("Peak Echo Measurement")

        # Configure grid layout (3x1)
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)
        self.grid_rowconfigure(2, weight=1)

        # Load ADC data from CSV and plot ADC and FFT
        self.plot_adc_fft()

        # Display results
        self.display_results()

    def plot_adc_fft(self):
        # Load ADC data from CSV
        df = pd.read_csv('D:/jenny/Documents/FAUS_Study/Sem3/Machine-Learning/spyder/adc_person_standing.csv')
        df_transposed = df.iloc[:, 16:1600].transpose()

        # Get the number of time steps
        num_time_steps = len(df_transposed.columns)

        # Select only the 10th row for plotting
        row_to_plot = df_transposed.iloc[1]  # 0-indexed, so 9 corresponds to the 10th row

        # Define sampling frequency and time period
        sampling_frequency = 2600
        time_period = 1 / sampling_frequency

        # Apply Hann windowing function to the ADC readings before FFT
        window = np.hanning(len(row_to_plot))
        row_to_plot_windowed = row_to_plot * window

        # Compute FFT of windowed ADC values
        fft_values = np.fft.fft(row_to_plot_windowed)
        frequencies = np.fft.fftfreq(len(row_to_plot), time_period)
        magnitudes = np.abs(fft_values)

        # Create a rectangular window for plotting
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))

        # Plot ADC values and FFT spectrum
        axes[0].plot(range(num_time_steps), row_to_plot)
        axes[0].set_xlabel('Time')
        axes[0].set_ylabel('ADC Value')
        axes[0].set_title('ADC Values over Time')

        axes[1].plot(frequencies, magnitudes)
        axes[1].set_xlabel('Frequency')
        axes[1].set_ylabel('Magnitude')
        axes[1].set_title('FFT Spectrum')

        # Embed the matplotlib plot into the tkinter GUI
        self.embed_plot(fig)

    def embed_plot(self, fig):
        # Embed the matplotlib plot into the tkinter GUI
        canvas = FigureCanvasTkAgg(fig, master=self)
        canvas.draw()
        canvas.get_tk_widget().grid(row=0, column=0, columnspan=2, sticky="nsew")

    def display_results(self):
        # Create a frame to display results
        results_frame = customtkinter.CTkFrame(self)
        results_frame.grid(row=2, column=0, columnspan=2, sticky="nsew")

        # Display RESULTS label
        results_label = customtkinter.CTkLabel(results_frame, text="RESULTS", font=customtkinter.CTkFont(size=12, weight="bold"),anchor="w")
        results_label.grid(row=0, column=0, pady=(10, 5))

        # Display dML (Distance measured by ML algorithm)
        dml_label = customtkinter.CTkLabel(results_frame, text="Distance measured by ML algorithm:", anchor="w")
        dml_label.grid(row=1, column=0, padx=10, sticky="w")
        dml_value = customtkinter.CTkLabel(results_frame, text=str(manual_distance_dML))

        dml_value.grid(row=1, column=1, padx=10, sticky="w")

        # Display accuracy
        accuracy_label = customtkinter.CTkLabel(results_frame, text="Accuracy:", anchor="w")
        accuracy_label.grid(row=2, column=0, padx=10, sticky="w")
        accuracy_value = customtkinter.CTkLabel(results_frame, text=str(test_acc))
        accuracy_value.grid(row=2, column=1, padx=10, sticky="w")

if __name__ == "__main__":
    app = App()
    app.geometry("800x600")  # Set the initial size of the GUI window
    app.mainloop()
