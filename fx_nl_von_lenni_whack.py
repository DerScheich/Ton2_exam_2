import numpy as np
import scipy.io.wavfile as wav
from pathlib import Path
import matplotlib.pyplot as plt



a, b = 1.1, 0
fs=48000
freq = 500

# linear steigendes Signal zur Erstellung der Kennlinien
dummy_input = np.arange(-1.1, 1.1, 0.1)

# nicht-linearer Effekt
def fx_nl(input_array, a, b):
    output_array = np.sinh(a*input_array+b)
    return output_array

# Übertragungs- und Verstärkungskennlinie plotten
def plot_transfer_functions(input_amplitude, output_amplitude, input_label, output_label):
    #Übertragung
    plt.plot(input_amplitude, output_amplitude)
    plt.plot(input_amplitude, 1.1 * input_amplitude) #Hilfsgerade, die überstiegen werden muss für Verstärkung >1.1
    plt.xlabel(input_label)
    plt.ylabel(output_label)
    plt.title("Übertragung")
    plt.grid(True)
    plt.show()

    #Verstärkung
    plt.plot(input_amplitude, output_amplitude / input_amplitude)
    plt. plot(input_amplitude, input_amplitude / input_amplitude * 1.1)
    plt.xlabel("Input")
    plt.ylabel("Verstärkung")
    plt.title("Verstärkung")
    plt.grid(True)
    plt.show()


#Sinus-Signal erstellen
def generate_sine(freq, duration=1, fs=48000, amplitude=1):
    t = np.arange(0, duration, 1 / fs)
    return amplitude * np.sin(2 * np.pi * freq * t)

#Klirrfaktor
def calculate_distortion_factor():
    sine_array = generate_sine(freq)

    non_linear_sine = fx_nl(sine_array,a,b)

    Y = np.fft.fft(non_linear_sine)
    N = len(non_linear_sine)
    freq_array = np.fft.fftfreq(N, 1/fs)

    # Nur positive Frequenzen
    mask = freq_array > 0
    freq_array = freq_array[mask]
    Y = np.abs(Y[mask])

    # Suche Index der Grundfrequenz und Harmonischen
    def find_freq(freq_array, target_array):
        return np.argmin(np.abs(freq_array - target_array))

    U1 = Y[find_freq(freq_array, freq)]
    U2 = Y[find_freq(freq_array, 2*freq)]
    U3 = Y[find_freq(freq_array, 3*freq)]

    # Klirrfaktor berechnen
    k = np.sqrt((U2**2 + U3**2) / U1**2)
    print(f"Klirrfaktor = {100*k:.4f} %")

if __name__ == "__main__":
    plot_transfer_functions(dummy_input, fx_nl(dummy_input, a, b), "Input", "Output")
    calculate_distortion_factor()
