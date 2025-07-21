"""
------------------------
B.Sc. Medientechnik
Tontechnik 2 - Sommersemester 2025
Prof. Dr.-Ing. E. Wilk
------------------------

Black-Box-Effekt für Praxisproblem 2 der Fallstudie
BlackBox3, mit:
    Eingabe: x[n Ts]
    Ausgabe: y[n Ts]
    (c) E. Wilk, Mai 2025    

"""

import numpy as np
import scipy
import matplotlib.pyplot as plt
from scipy.io.wavfile import read
from scipy.io.wavfile import write

def blackbox(eingang, fs):
    delta_T = 1/fs
    x = eingang*pow(0.5,3)
    lang = len(x)
   
    w1 = 1
    w2 = 2
    p = 3
    d = 1200
    e = 600
    
    for i in range(w1,w2):
        #print(x[0])
        if i == 0:  
            y = np.zeros(lang)         
            alp_1 = (e*2*np.pi*delta_T)/(1+e*2*np.pi*delta_T)
            y[0] = alp_1*x[0]
            for i in range(1,lang):
                y[i] = alp_1*x[i] + (1-alp_1)*y[i-1] 
            if p == 2:
                y_1 = np.zeros(lang)
                y_1[0] = alp_1*x[0]
                for i in range(1,lang):
                   y_1[i] = alp_1*y[i] + (1-alp_1)*y_1[i-1]
                y = y_1   
            if w2 == 2: x = y
        if i == 1:  
            alp_2 = 1/(1+d*2*np.pi*delta_T)
            y1 = np.zeros_like(x)
            y1[0] = x[0]
            for i in range(1, lang):
                y1[i] = alp_2 * (y1[i-1] + x[i] - x[i-1])
            y = y1
            if p == 2:
                y2 = np.zeros_like(x)
                y2[0] = y1[0]
                for i in range(1, lang):
                    y2[i] = alp_2 * (y2[i-1] + y1[i] - y1[i-1])
                y = y2            
    return(y)
    


