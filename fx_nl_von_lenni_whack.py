import numpy as np
import scipy.io.wavfile as wav
from pathlib import Path
import matplotlib.pyplot as plt



#input = np.arange(-1.1, 1.1, 0.1)
#print(f"input= {input}")

a, b = 1.1, 0
fs=48000

def fx_nl(input_array, a, b):
    output_array = []
    output_array = np.sinh(a*input_array+b)
    print(f"output_array= {output_array}")
    return output_array


def plot_transfer_functions(input,output, input_label, output_label):
    #Übertragung
    plt.plot(input, output)
    plt.plot(input, 1.1*input) #Hilfsgerade, die überstiegen werden muss für Verstärkung >1.1
    plt.xlabel(input_label)
    plt.ylabel(output_label)
    plt.title("Übertragung")
    plt.grid(True)
    plt.show()

    #Verstärkung
    plt.plot(input, output/input)
    plt. plot(input, input/input*1.1)
    plt.xlabel("Input")
    plt.ylabel("Verstärkung")
    plt.title("Verstärkung")
    plt.grid(True)
    plt.show()

plot_transfer_functions(input, fx_nl(input, a, b), "Input", "Output")

#Sinus-Signal erstellen
def generate_sine(freq, duration=1, fs=48000, amplitude=1):
    t = np.arange(0, duration, 1 / fs)
    return amplitude * np.sin(2 * np.pi * freq * t)

#Klirrfaktor

non_linear_sine = fx_nl(generate_sine(500),a,b)

