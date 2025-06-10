import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import chirp
from scipy.io.wavfile import read, write
from pathlib import Path
import librosa.display
from matplotlib.ticker import FixedLocator, FuncFormatter

from PP2_BlackBox3 import blackbox

# Fs
fs = 44100

# paths
input_dir = Path('input')
output_dir = Path('output')

# Verzeichnisse erstellen, falls nicht vorhanden
input_dir.mkdir(parents=True, exist_ok=True)
output_dir.mkdir(parents=True, exist_ok=True)

# sweep
T = 5.0

t = np.linspace(0, T, int(fs*T))
sweep = chirp(t, f0=20, f1=20000, t1=T, method='logarithmic')
y_sweep = blackbox(sweep, fs)

# write outs
write(output_dir / 'sweep_in.wav', fs, np.float32(sweep))
write(output_dir / 'sweep_bb.wav', fs, np.float32(y_sweep))
print(f"Sweep-Files geschrieben: {output_dir / 'sweep_in.wav'}, {output_dir / 'sweep_bb.wav'}")

# spectrums
n_fft = 4096
hop_length = 4096  # abtastgröße freq.bereich

S_in = np.abs(librosa.stft(sweep, n_fft=n_fft, hop_length=hop_length))
S_out = np.abs(librosa.stft(y_sweep, n_fft=n_fft, hop_length=hop_length))

ref_val = np.max(S_in)
D_in = librosa.amplitude_to_db(S_in, ref=ref_val)
D_out = librosa.amplitude_to_db(S_out, ref=ref_val)

fig, ax = plt.subplots(nrows=2, ncols=1, sharex=True, figsize=(10, 6))

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
ax[0].set(title='Spektogram (Sweep vor Blackbox)')
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
ax[1].set(title='Spektogram (Sweep nach Blackbox)')

cbar_ax = fig.add_axes([0.9, 0.15, 0.02, 0.7])
fig.colorbar(img_out, cax=cbar_ax, format='%+0.0f dB')
fig.tight_layout(rect=[0, 0, 0.9, 1])

plt.savefig(output_dir / 'sweep_spektrogramm.png', dpi=300)
plt.close(fig)

# amplitude freq gang
N = len(sweep)
freqs = np.fft.rfftfreq(N, 1/fs)
S = np.fft.rfft(sweep)
Y = np.fft.rfft(y_sweep)
resp_S = 20 * np.log10(np.abs(S) / np.max(np.abs(S)))
resp_Y = 20 * np.log10(np.abs(Y) / np.max(np.abs(S)))

plt.rcParams["font.size"] = 10
fig2, ax2 = plt.subplots(figsize=(8, 4))
ax2.plot(freqs, resp_S, color="#8b0000", lw=1.2, label='Input')
ax2.plot(freqs, resp_Y, color="#8b0000", lw=1.2, label='Output')
ax2.set_xscale("log")
ax2.set_xlim(20, 20000)
xt = [20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000]
ax2.xaxis.set_major_locator(FixedLocator(xt))
ax2.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{int(x/1000)} k" if x >= 1000 else f"{int(x)}"))
ax2.set_xlabel("Frequenz [Hz]")
ax2.set_ylim(-60, 5)
ax2.set_ylabel("Pegel [dB relativ zur Maximalaussteuerung]")
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

# write out
write(output_dir / 'output_blackbox.wav', fs2, np.int16(y * 32767))
print(f"Output als {output_dir / 'output_blackbox.wav'} gespeichert")

