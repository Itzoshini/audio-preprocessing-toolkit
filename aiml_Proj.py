#!/usr/bin/env python
# coding: utf-8

# In[1]:


# AUDIO PREPROCESSING
#Load Audio: Load the audio file.
#Convert to Mono: Convert to a single audio channel.
#Reduce Noise: Remove background noise.
#Apply Bandpass Filter: Keep only relevant frequencies (e.g., 300 Hz - 3000 Hz).
#Extract Pitch: Identify the pitch (fundamental frequency).
#Compute Zero-Crossing Rate: Measure how often the audio signal crosses zero amplitude.
#Remove Silence: Focus on segments with sound (Voice Activity Detection).
#Normalize Audio: Adjust volume to a consistent level.
#Save Processed Audio 


# In[63]:


import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf
import noisereduce as nr
from scipy.signal import butter, lfilter
from IPython.display import Audio, display


# In[64]:


import librosa
import librosa.display
import matplotlib.pyplot as plt

# Load the audio file
file_path = "C:/Users/HP/Downloads/messi_vs_madrid_ucl.mp3"  # Replace with your file path
audio, sr = librosa.load(file_path, sr=22050)

# Plot the original audio waveform
plt.figure(figsize=(10, 4))
librosa.display.waveshow(audio, sr=sr, alpha=0.6)
plt.title("Original Audio Waveform")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.show()

# Display the audio player (for Jupyter Notebook)
from IPython.display import Audio
display(Audio(audio, rate=sr))


# In[65]:


import noisereduce as nr

# Convert to mono
audio_mono = librosa.to_mono(audio)

# Apply noise reduction
audio_denoised = nr.reduce_noise(y=audio_mono, sr=sr)

# Plot denoised audio waveform
plt.figure(figsize=(10, 4))
librosa.display.waveshow(audio_denoised, sr=sr, alpha=0.6)
plt.title("Denoised Audio Waveform")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.show()

# Display the denoised audio player
display(Audio(audio_denoised, rate=sr))


# In[66]:


from scipy.signal import butter, lfilter

# Bandpass filter function
def butter_bandpass(lowcut, highcut, fs, order=5):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return b, a

# Bandpass filtering function
def bandpass_filter(audio, lowcut, highcut, sr, order=5):
    b, a = butter_bandpass(lowcut, highcut, sr, order)
    return lfilter(b, a, audio)

# Apply bandpass filter (300 Hz - 3000 Hz)
lowcut = 300.0
highcut = 3000.0
audio_filtered = bandpass_filter(audio_denoised, lowcut, highcut, sr)

# Plot the filtered audio waveform
plt.figure(figsize=(10, 4))
librosa.display.waveshow(audio_filtered, sr=sr, alpha=0.6)
plt.title("Filtered Audio Waveform (Bandpass: 300 Hz - 3000 Hz)")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.show()



# In[67]:


# Pitch extraction
pitches, magnitudes = librosa.core.piptrack(y=audio_filtered, sr=sr)
pitch = []
for t in range(pitches.shape[1]):
    index = magnitudes[:, t].argmax()
    pitch.append(pitches[index, t])
pitch = np.array(pitch)

# Plot Pitch Extraction
plt.figure(figsize=(10, 6))
plt.plot(pitch, label="Extracted Pitch")
plt.title("Pitch Extraction")
plt.xlabel("Time (s)")
plt.ylabel("Pitch (Hz)")
plt.legend()
plt.show()

# Display the first few values of pitch
print(f"Pitch extracted: {pitch[:100]}...")  # Display first 100 values of pitch for brevity


# In[68]:


# Pitch extraction
pitches, magnitudes = librosa.piptrack(y=audio_filtered, sr=sr)
pitch = [pitches[:, t].argmax() for t in range(pitches.shape[1])]
pitch = np.array(pitch)
print(f"Pitch extracted: {pitch[:100]}...")  # Display first 100 values

# Plot Pitch
plt.figure(figsize=(10, 6))
plt.plot(pitch, label="Extracted Pitch")
plt.title("Pitch Extraction")
plt.xlabel("Time (s)")
plt.ylabel("Pitch (Hz)")
plt.legend()
plt.show()


# In[69]:


# Zero-Crossing Rate :Measure how often the audio signal crosses zero amplitude.

zcr = librosa.feature.zero_crossing_rate(audio_filtered)[0]

# Plot Zero-Crossing Rate
plt.figure(figsize=(10, 6))
plt.plot(zcr, label="Zero-Crossing Rate", color='orange')
plt.title("Zero-Crossing Rate")
plt.xlabel("Time (s)")
plt.ylabel("Zero-Crossing Rate")
plt.legend()
plt.show()

# Display the first few values of ZCR
print(f"Zero-Crossing Rate extracted: {zcr[:100]}...")  # Display first 100 values for brevity


# In[70]:


# Silence removal (Voice Activity Detection - VAD)
audio_vad, _ = librosa.effects.trim(audio_filtered)

# Plot original and VAD audio waveforms
plt.figure(figsize=(10, 4))
librosa.display.waveshow(audio_filtered, sr=sr, alpha=0.6)
plt.title("Audio Before Silence Removal (VAD)")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")

plt.figure(figsize=(10, 4))
librosa.display.waveshow(audio_vad, sr=sr, alpha=0.6)
plt.title("Audio After Silence Removal (VAD)")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.show()

# Display the VAD audio player
display(Audio(audio_vad, rate=sr))


# In[71]:


# Normalize the audio
audio_normalized = librosa.util.normalize(audio_vad)

# Plot the original and normalized audio waveforms for comparison
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
librosa.display.waveshow(audio_vad, sr=sr, alpha=0.6)
plt.title("Audio Before Normalization")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")

plt.subplot(1, 2, 2)
librosa.display.waveshow(audio_normalized, sr=sr, alpha=0.6)
plt.title("Audio After Normalization")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")

plt.tight_layout()
plt.figtext(0.15, 0.05, f'Original Audio Range: [{np.min(audio_vad):.4f}, {np.max(audio_vad):.4f}]', ha='left', fontsize=10, color='red')
plt.figtext(0.65, 0.05, f'Normalized Audio Range: [{np.min(audio_normalized):.4f}, {np.max(audio_normalized):.4f}]', ha='left', fontsize=10, color='blue')
plt.show()

# Plot the amplitude scaling before and after normalization
plt.figure(figsize=(10, 6))
plt.plot(audio_vad[:5000], label="Before Normalization", color='red')
plt.plot(audio_normalized[:5000], label="After Normalization", color='blue')
plt.title("Amplitude Scaling: Before and After Normalization")
plt.xlabel("Samples")
plt.ylabel("Amplitude")
plt.legend()
plt.tight_layout()


# In[72]:


import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf
from IPython.display import Audio, display

# Function to preprocess audio_files and extract MFCCs, Spectrogram, and RMS
def preprocess_audio(file_path, output_audio_path):
    try:
        # Load the audio file
        audio, sr = librosa.load(file_path, sr=22050)
        print("Audio loaded successfully.")

        # Extract MFCCs
        mfccs = np.mean(librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13), axis=1).tolist()
        print("MFCCs extracted.")

        # Plot Mel-spectrogram
        mel_spectrogram = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=128, fmax=8000)
        log_mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)

        plt.figure(figsize=(10, 6))
        librosa.display.specshow(log_mel_spectrogram, x_axis='time', y_axis='mel', sr=sr)
        plt.colorbar(format='%+2.0f dB')
        plt.title('Mel-spectrogram')
        plt.show()

        # Compute RMS Features
        rms = librosa.feature.rms(y=audio)

        # Plot the RMS Features for original audio
        plt.figure(figsize=(10, 6))
        plt.plot(rms[0], label='RMS', color='blue')
        plt.title('RMS Feature Visualization')
        plt.xlabel('Frames')
        plt.ylabel('RMS')
        plt.legend()
        plt.show()

        # Save the processed audio (optional step)
        sf.write(output_audio_path, audio, sr)
        print(f"Processed audio saved to {output_audio_path}")

        return {"mfccs": mfccs, "sample_rate": sr, "audio": audio, "rms": rms}

    except Exception as e:
        print(f"Error during preprocessing: {e}")
        return None

# Main script
if __name__ == "__main__":
    # Input and output audio file paths
    audio_file = "C:/Users/HP/Downloads/messi_vs_madrid_ucl.mp3"
    output_audio_path = "C:/Users/HP/Downloads/audio/output_audio.wav"

    # Preprocess the audio file
    preprocessed = preprocess_audio(audio_file, output_audio_path)
    if preprocessed:
        mfccs = preprocessed["mfccs"]
        sr = preprocessed["sample_rate"]
        audio = preprocessed["audio"]
        rms = preprocessed["rms"]

        # Print extracted MFCCs and RMS
        print(f"MFCCs: {mfccs}")
        print(f"RMS: {rms[0][:100]}...")  # Display first 100 RMS values for brevity

        # Play the processed audio in Jupyter Notebook
        print("Playing processed audio...")
        display(Audio(audio_vad, rate=sr))

       
    else:
        print("Failed to preprocess the audio file.")


# In[ ]:




