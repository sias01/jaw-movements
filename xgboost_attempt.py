import librosa
import numpy as np
import os
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
import joblib  # for saving and loading the model
import xgboost.onnx as xgb_onnx  # for ONNX conversion

def extract_mfcc(y, sr, n_mfcc=13, hop_length=256, n_fft=1024):
    mfcc_feat = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc,
                                     hop_length=hop_length, n_fft=n_fft)
    return mfcc_feat.astype(np.float32)  # shape: [n_mfcc, time]

def load_dataset(file_paths, labels, sr=22050):
    features = []
    for path in file_paths:
        y, _ = librosa.load(path, sr=sr)
        mfcc = extract_mfcc(y, sr)

        if mfcc.shape[1] < 128:
            pad_width = 128 - mfcc.shape[1]
            mfcc = np.pad(mfcc, ((0, 0), (0, pad_width)), mode='constant')
        else:
            mfcc = mfcc[:, :128]

        features.append(mfcc.flatten())

    return np.array(features), np.array(labels)

# Prepare dataset
files = os.listdir("./segmented_audios")
file_paths = [os.path.join("segmented_audios", file) for file in files if file.endswith(".wav")]
labels_text = [file.split("_")[3] for file in file_paths]

label_mapping = {"chew": 0, "bite": 1, "chew-bite": 2}
labels = [label_mapping[label] for label in labels_text]

train_files, val_files, train_labels, val_labels = train_test_split(file_paths, labels, test_size=0.2)

X_train, y_train = load_dataset(train_files, train_labels)
X_val, y_val = load_dataset(val_files, val_labels)

# # Train model
# model = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
# model.fit(X_train, y_train)

# # Evaluate
# y_pred = model.predict(X_val)
# acc = accuracy_score(y_val, y_pred)
# print(f"Validation Accuracy: {acc * 100:.2f}%")

# # Save the model (Pickle format using joblib)
# joblib.dump(model, "xgb_model.pkl")
# print("Saved XGB model to 'xgb_model.pkl'")

# ------------------------
# Convert the .pkl to ONNX
# ------------------------

# Load the saved model
loaded_model = joblib.load("xgb_model.pkl")

# Define the input shape
input_shape = (None, X_train.shape[1])  # (None, 1664) in your case

# Convert to ONNX
onnx_model = xgb_onnx.to_onnx(loaded_model, input_shape=input_shape, target_opset=17)

# Save the ONNX model
with open("xgb_model.onnx", "wb") as f:
    f.write(onnx_model.SerializeToString())

print("Saved ONNX model to 'xgb_model.onnx'")
