import numpy as np
from scipy.io import wavfile
from scipy import interpolate

# SAMPLE AUDIO, WRITE TO NEW .WAV FILE, AND RETURN NEW AUDIO
def sampleWithSpecificRate(newRate, audioData, currentRate):
        duration = audioData.shape[0] / currentRate

        currentTime  = np.linspace(0, duration, audioData.shape[0])
        newTime  = np.linspace(0, duration, int(audioData.shape[0] * newRate / currentRate))

        interpolator = interpolate.interp1d(currentTime, audioData.T)
        newAudioData = interpolator(newTime).T

        wavfile.write("resampled_audio.wav", newRate, np.round(newAudioData).astype(audioData.dtype))

        return newAudioData