# 🎼 Audio Preprocessing and Feature Extraction with Python

## 📘 Project Description

This project performs comprehensive **audio preprocessing** and **feature extraction** using Python. It is designed to clean and analyze audio files by applying techniques like noise reduction, silence removal, pitch detection, MFCC extraction, and more. The extracted features and cleaned audio are useful for speech recognition, acoustic analysis, and machine learning tasks.

---

## 🔧 Features

- 🎧 **Audio Preprocessing**
  - Load audio file (`.mp3`)
  - Convert stereo to mono
  - Noise reduction
  - Bandpass filtering (300 Hz to 3000 Hz)
  - Silence removal using Voice Activity Detection (VAD)
  - Audio normalization

- 📊 **Feature Extraction**
  - Pitch detection using `piptrack`
  - Zero-Crossing Rate (ZCR)
  - MFCCs (Mel-Frequency Cepstral Coefficients)
  - RMS (Root Mean Square) energy
  - Mel-spectrogram and waveform visualization

- 💾 **Output**
  - Save processed audio as `.wav`
  - Display plots and extracted features

---

## 🛠 Libraries Used

- `librosa`
- `matplotlib`
- `noisereduce`
- `soundfile`
- `scipy`
- `IPython.display`

---

## 📁 Directory Structure

├── aiml_Proj.py # Main Python script
├── input_audio.mp3 # Replace with your input file
├── output_audio.wav # Processed audio output

## ▶️ How to Run

1. Install dependencies:
   ```bash
   pip install librosa matplotlib soundfile noisereduce
2. Edit the file path in the script:
   ```bash
   file_path = "your audio path"
3. Run the script:
   ```bash
   python aiml_Proj.py
## Results
1. Processed audio saved to output_audio.wav
2. MFCCs and RMS printed
3. Visualizations shown (waveform, pitch, ZCR, spectrogram)



