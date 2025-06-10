"""
E. Wilk
1.6.2025
Test-Code für TON2 PP2 SoSe 25

Externe Funktion für RR

"""

import numpy as np

try:
    from pyfftw.interfaces.numpy_fft import irfft  # Performs much better than numpy's fftpack
except ImportError:  # Use monkey-patching np.fft perhaps instead?
    from numpy.fft import irfft  # pylint: disable=ungrouped-imports

def rosa_rauschen(N, state = None): #pink(N, state=None):
    """
    Pink noise.
    :param N: Amount of samples.
    :param state: State of PRNG.
    :type state: :class:`np.random.RandomState`
    Pink noise has equal power in bands that are proportionally wide.
    Power density decreases with 3 dB per octave.
    """
    state = np.random.RandomState() if state is None else state
    uneven = N % 2
    X = state.randn(N // 2 + 1 + uneven) + 1j * state.randn(N // 2 + 1 + uneven)
    S = np.sqrt(np.arange(len(X)) + 1.)  # +1 to avoid divide by zero
    y = (irfft(X / S)).real
    if uneven:
        y = y[:-1]
    return (y)