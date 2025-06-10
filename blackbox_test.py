import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import chirp
from scipy.io.wavfile import read, write
from pathlib import Path
import librosa.display

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
hop_length = 128 # abtastgröße freq.bereich

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

