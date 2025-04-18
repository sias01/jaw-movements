import numpy as np
import librosa
from scipy.signal import find_peaks
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
import os

def extract_14_features(y, sr):
    # Normalize audio
    y = y / np.max(np.abs(y))
    
    # Envelope
    envelope = np.abs(y)
    
    # V1: Max amplitude
    v1 = np.max(envelope)
    # V2: Min amplitude
    v2 = np.min(envelope)
    # V3: Range
    v3 = v1 - v2
    # V4: Duration (in seconds)
    v4 = len(y) / sr
    # V5: Area under the envelope
    v5 = np.sum(envelope) / sr
    # V6: Symmetry (difference between attack and decay)
    half = len(envelope) // 2
    v6 = np.abs(np.sum(envelope[:half]) - np.sum(envelope[half:])) / sr
    # V7: Number of local maxima
    peaks, _ = find_peaks(envelope)
    v7 = len(peaks)

    # Spectral features
    S = np.abs(librosa.stft(y, n_fft=1024, hop_length=256))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=1024)[:S.shape[0]]  # Trim to match S

    spectral_centroid = np.mean(librosa.feature.spectral_centroid(S=S, sr=sr))
    spectral_entropy = -np.sum((S / np.sum(S, axis=0, keepdims=True)) * np.log2(S + 1e-10)) / S.shape[1]
    spectral_flux = np.mean(np.sqrt(np.sum(np.diff(S, axis=1)**2, axis=0)))
    spectral_bandwidth = np.mean(librosa.feature.spectral_bandwidth(S=S, sr=sr))
    spectral_rolloff = np.mean(librosa.feature.spectral_rolloff(S=S, sr=sr, roll_percent=0.85))
    spectral_flatness = np.mean(librosa.feature.spectral_flatness(S=S))

    # Energy in 1–2 kHz band
    band_mask = (freqs >= 1000) & (freqs <= 2000)
    energy_band = np.mean(np.sum(S[band_mask, :], axis=0))

    return np.array([
        v1, v2, v3, v4, v5, v6, v7,
        spectral_centroid, spectral_entropy, spectral_flux,
        spectral_bandwidth, spectral_rolloff, spectral_flatness,
        energy_band
    ])

X = []
y = []

for i in os.listdir("./segmented_audios"):
    try:
        print(i)
        # y, sr = librosa.load("./segmented_audios/recording_01_bite_0.wav", sr=None)
        # print(f"Sample rate: {sr}")
        # features = extract_14_features(y, sr)
        # print(features)
        print(os.path.join("segmented_audios",i))
        t, sr = librosa.load(os.path.join("segmented_audios",i), sr=None)
        # print(f"Sample rate: {sr}")
        features = extract_14_features(t, sr)
        # print(features)
        X.append(features)
        label = i.split("_")[2]
        if label == "chew":
            y.append(0)
        elif label == "bite":
            y.append(1)
        elif label == "chew-bite":
            y.append(2)
        # print(label)
        # print(X)
        # print(y)
        # break
    except Exception as e:
        error_message = f"{i} - {str(e)}\n"
        with open("failed_files.txt", "a") as f:
            f.write(error_message)
        print(f"Failed to process {i}: {e}")



# # X = list of 14-d feature vectors
# # y = corresponding labels (e.g., 0 = chew, 1 = bite, 2 = chew-bite)

# X = []
# y = []

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)
print(classification_report(y_test, y_pred, target_names=["Chew", "Bite", "Chew-Bite"]))
