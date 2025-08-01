# PP2 Main
# Hart, Teich
# Gruppe 2
# 18.06.2025

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import chirp
from scipy.io.wavfile import read, write
from pathlib import Path
import librosa.display
from matplotlib.ticker import FixedLocator, FuncFormatter
import scipy.io.wavfile as wav

from PP2_BlackBox3 import blackbox


# Fs
fs = 44100

# paths
input_dir = Path('input')
output_dir = Path('output')

# Verzeichnisse erstellen, falls nicht vorhanden
input_dir.mkdir(parents=True, exist_ok=True)
output_dir.mkdir(parents=True, exist_ok=True)


def create_sweep():
    sweep_time = 5.0
    t = np.linspace(0, sweep_time, int(fs * sweep_time))
    sweep = chirp(t, f0=20, f1=20000, t1=sweep_time, method='linear')
    return sweep


def sweep_blackbox(sweep):
    bb_sweep = blackbox(sweep, fs)
    return bb_sweep


def write_output_files(sweep, bb_sweep):
    # write outs
    write(output_dir / 'sweep_in.wav', fs, np.float32(sweep))
    write(output_dir / 'sweep_bb.wav', fs, np.float32(bb_sweep))
    print(f"Sweep-Files geschrieben: {output_dir / 'sweep_in.wav'}, {output_dir / 'sweep_bb.wav'}")


def create_spectrogram(sweep, bb_sweep):
    n_fft = 4096
    hop_length = 4096  # abtastgröße freq.bereich

    S_in = np.abs(librosa.stft(sweep, n_fft=n_fft, hop_length=hop_length))
    S_out = np.abs(librosa.stft(bb_sweep, n_fft=n_fft, hop_length=hop_length))

    # Berechnung der Pegel (bezogen auf maximal vorkommenden Wert)
    ref_val = np.max(S_in)
    D_in = librosa.amplitude_to_db(S_in, ref=ref_val)
    D_out = librosa.amplitude_to_db(S_out, ref=ref_val)

    fig, ax = plt.subplots(nrows=2, ncols=1, sharex=True, figsize=(10, 6), constrained_layout=True)

    img_in = librosa.display.specshow(
        D_in,
        y_axis='log',
        x_axis='time',
        sr=fs,
        hop_length=hop_length,
        ax=ax[0],
        vmin=-80,
        vmax=0,
        cmap='magma',
    )
    ax[0].set(title='Spektogramm (Sweep vor Blackbox)')
    ax[0].label_outer()

    img_out = librosa.display.specshow(
        D_out,
        y_axis='log',
        x_axis='time',
        sr=fs,
        hop_length=hop_length,
        ax=ax[1],
        vmin=-80,
        vmax=0,
        cmap='magma',
    )
    ax[1].set(title='Spektogramm (Sweep nach Blackbox)')

    cbar_ax = fig.add_axes([0.9, 0.15, 0.02, 0.7])
    fig.colorbar(img_out, cax=cbar_ax, format='%+0.0f dB')
    # fig.tight_layout(rect=[0, 0, 0.9, 1])

    # Grafiken speichern
    plt.savefig(output_dir / 'sweep_spektrogramm.png', dpi=300)
    plt.close(fig)


def create_frequency_response(sweep, bb_sweep):
    # Amplitudenfrequenzgang
    N = len(sweep)
    freqs = np.fft.rfftfreq(N, 1 / fs)
    S = np.fft.rfft(sweep)
    Y = np.fft.rfft(bb_sweep)
    resp_S = 20 * np.log10(np.abs(S) / np.max(np.abs(S)))
    resp_Y = 20 * np.log10(np.abs(Y) / np.max(np.abs(S)))

    # Parameter für den Plot
    plt.rcParams["font.size"] = 10
    fig2, ax2 = plt.subplots(figsize=(8, 4))
    ax2.plot(freqs, resp_S, color="#8b0000", lw=1.2, label='Input')
    ax2.plot(freqs, resp_Y, color="#8b0000", lw=1.2, label='Output')
    ax2.set_xscale("log")
    ax2.set_xlim(20, 20000)
    xt = [20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000]
    ax2.xaxis.set_major_locator(FixedLocator(xt))
    ax2.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{int(x / 1000)} k" if x >= 1000 else f"{int(x)}"))
    ax2.set_xlabel("Frequenz [Hz]")
    ax2.set_ylim(-60, 5)
    ax2.set_ylabel("Pegel [dB relativ zur Maximalaussteuerung der Eingangsdatei]")
    ax2.grid(which="both", color="#666", alpha=0.3, ls="-")
    ax2.set_title("Amplituden-Frequenzgang", pad=10)
    fig2.tight_layout()
    plt.savefig(output_dir / 'amplitude_freq_response.png', dpi=300)
    plt.close(fig2)

    # wav
    wav_path = input_dir / 'input.wav'
    if not wav_path.exists():
        raise FileNotFoundError(f"Datei nicht gefunden: {wav_path}")

    fs2, audio = read(wav_path)

    # blackbox and normalize
    y = blackbox(audio.astype(float), fs2)
    y = y / np.max(np.abs(y))

    # Normalisieren und Speichern
    write(output_dir / 'output_blackbox.wav', fs2, np.int16(y * 32767))
    print(f"Output als {output_dir / 'output_blackbox.wav'} gespeichert")


def aufgabe_a():
    print("----------------------- AUFGABENTEIL A ----------------------------------")
    # Sweep
    sweep = create_sweep()
    bb_sweep = sweep_blackbox(create_sweep())
    write_output_files(sweep, bb_sweep)

    create_spectrogram(sweep, bb_sweep)
    create_frequency_response(sweep, bb_sweep)

# ----------------------- AUFGABENTEIL B ----------------------------------

def create_dummy_input():
    # linear steigendes Signal zur Erstellung der Kennlinien
    dummy_input = np.arange(-1, 1.1, 0.1)
    return dummy_input

# nicht-linearer Effekt
def fx_nl(input_array, a=1.1, b=0):
    output_array = np.sinh(a*input_array+b)
    return output_array

# Übertragungs- und Verstärkungskennlinie plotten
def plot_transfer_functions(input_amplitude, output_amplitude, input_label, output_label):
    #Übertragung
    plt.plot(input_amplitude, output_amplitude)
    plt.plot(input_amplitude, 1.1 * input_amplitude) #Hilfsgerade, die "überstiegen" werden muss für Verstärkung > 1.1
    plt.xlabel(input_label)
    plt.ylabel(output_label)
    plt.title("Übertragung")
    plt.grid(True)
    plt.show()

    #Verstärkung
    plt.plot(input_amplitude, output_amplitude / input_amplitude)
    plt. plot(input_amplitude, input_amplitude / input_amplitude * 1.1) #Hilfsgerade, die überstiegen werden muss für Verstärkung > 1.1
    plt.xlabel("Input")
    plt.ylabel("Verstärkung")
    plt.title("Verstärkung")
    plt.grid(True)
    plt.show()
    print("Übertragungs- und Verstärkungskennlinie erstellt")

#Sinus-Signal erstellen
def generate_sine(freq, duration=1, fs=44100, amplitude=1):
    t = np.arange(0, duration, 1 / fs)
    return amplitude * np.sin(2 * np.pi * freq * t)

#Klirrfaktor
def calculate_distortion_factor(freq=500):
    # Sinus-Signals erstellen und durch fx_nl schicken
    sine_array = generate_sine(freq)
    non_linear_sine = fx_nl(sine_array)

    # Wandlung Signal in Frequenzbereich, Berechnung Frequenzachse
    Y = np.fft.fft(non_linear_sine)
    N = len(non_linear_sine)
    freq_array = np.fft.fftfreq(N, 1/fs)

    # Nur positive Frequenzen
    mask = freq_array > 0
    freq_array = freq_array[mask]
    Y = np.abs(Y[mask])

    # Suche Index der Grundfrequenz und Harmonischen
    def find_freq(freq_array, target_freq):
        return np.argmin(np.abs(freq_array - target_freq))

    # Bestimme Amplituden der Harmonischen
    U1 = Y[find_freq(freq_array, freq)]
    U2 = Y[find_freq(freq_array, 2*freq)]
    U3 = Y[find_freq(freq_array, 3*freq)]

    # Klirrfaktor berechnen und ausgeben
    k = 100*np.sqrt((U2**2 + U3**2) / (U1**2+U2**2 + U3**2))
    print(f"Klirrfaktor = {k:.2f} %")

def apply_fx_nl_to_file():

    # Path Manager
    root_path = Path(__file__).parent.resolve()
    i_path = root_path / "input" / "spoken.wav"
    output_path = root_path / "output" / "output_file_wav.wav"

    _, input_file = wav.read(i_path)

    input_file = input_file/abs(max(input_file)) # auf 1 normalisieren
    output_file = fx_nl(input_file)

    # Ergebnisdatei speichern
    wav.write(output_path, fs, np.int16(output_file * 32767))


def aufgabe_b():
    print("----------------------- AUFGABENTEIL B ----------------------------------")



    plot_transfer_functions(create_dummy_input(), fx_nl(create_dummy_input()), "Input", "Output")
    calculate_distortion_factor()
    apply_fx_nl_to_file()


if __name__ == "__main__":
    aufgabe_a()
    aufgabe_b()












