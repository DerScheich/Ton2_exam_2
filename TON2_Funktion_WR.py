"""
E. Wilk
1.6.2025
Test-Code für TON2 PP2 SoSe 25

Externe Funktion für WR

"""
import numpy as np

def weisses_rauschen(fs,laenge):  #Angabe von laenge in sek
    return np.random.normal(0, 1, fs * laenge)  #np.random.normal(mean, standard, anzahl samples)
