import numpy as np
import scipy.io.wavfile as wav
from pathlib import Path
import matplotlib.pyplot as plt


#a = 1 # Eingangsvorverstärkung
#b = 1 # Arbeitspunkt


#input = input.astype(np.float32)
input = np.arange(-1.1, 1.1, 0.1)
print(f"input= {input}")


def make_non_linear(input_array, a, b):
    output_array = []
    output_array = np.sinh(a*input_array+b)
    print(f"output_array= {output_array}")
    return output_array



def make_plots(x,y, xlabel, ylabel):
    #Übertragung
    plt.plot(x, y)
    plt.plot(x, 1.1*x) #Hilfsgerade, die überstiegen werden muss für Verstärkung >1.1
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title("Übertragung")
    plt.grid(True)
    plt.show()

    #Verstärkung
    plt.plot(x, y/x)
    plt. plot(x, x/x*1.1)
    plt.xlabel("Input")
    plt.ylabel("Verstärkung")
    plt.title("Verstärkung")
    plt.grid(True)
    plt.show()

a, b = 2, 0

make_plots(input, make_non_linear(input, a, b), "Input", "Output")

#make_plots(input, make_non_linear(input, a, b), "Input", "Output")
