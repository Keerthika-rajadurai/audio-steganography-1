import soundfile
import numpy as np
import matplotlib.pyplot as plt
from librosa import stft, amplitude_to_db
from librosa.display import specshow

initialAudio, initialAudioRate  = soundfile.read("audio.wav")
encodedAudio, encodedAudioRate = soundfile.read("encoded_audio.wav") 

initialAudioWindowLen = int(0.025 * initialAudioRate)
initialAudioHoplen = int(0.01 * initialAudioRate)

encodedAudioWindowLen = int(0.025 * encodedAudioRate)
encodedAudioHoplen = int(0.01 * encodedAudioRate)

initialAudioSpectrogram = np.abs(stft(initialAudio, hop_length=initialAudioHoplen, win_length=initialAudioWindowLen))
encodedAudioSpectrogram = np.abs(stft(encodedAudio, hop_length=encodedAudioHoplen, win_length=encodedAudioWindowLen))

specshow(amplitude_to_db(initialAudioSpectrogram, ref=np.max), sr=initialAudioRate, hop_length=initialAudioHoplen, y_axis="linear", x_axis="time")
plt.title("Spectrogram of Initial Audio")
plt.colorbar(format="%+2.0f dB")
plt.tight_layout()
plt.show()

specshow(amplitude_to_db(encodedAudioSpectrogram, ref=np.max), sr=encodedAudioRate, hop_length=encodedAudioHoplen, y_axis="linear", x_axis="time")
plt.title("Spectrogram of Encoded Audio")
plt.colorbar(format="%+2.0f dB")
plt.tight_layout()
plt.show()