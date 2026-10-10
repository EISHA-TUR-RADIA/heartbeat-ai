# ECG Arrhythmia AI — Google Colab starter
# Research/demo prototype only. NOT for medical diagnosis.
# In a separate Google Colab cell, first run:
# !pip -q install wfdb numpy scikit-learn matplotlib

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import wfdb
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, accuracy_score, recall_score

# A small subset of MIT-BIH records to keep the demo manageable.
RECORDS = [
    "100", "101", "103", "105", "106", "108", "109", "111", "112", "113",
    "117", "118", "119", "121", "122", "123", "200", "202", "203", "205"
]

# Beat labels used for this simplified binary demo.
NORMAL_SYMBOLS = {"N", "L", "R", "e", "j"}
ABNORMAL_SYMBOLS = {"A", "a", "J", "S", "V", "E", "F", "f"}

HALF_WINDOW = 90  # 90 samples before and after the annotated beat
SAMPLES_PER_BEAT = HALF_WINDOW * 2

def load_beats(record_id):
    """Download one MIT-BIH record and return selected labeled ECG beat windows."""
    record = wfdb.rdrecord(record_id, pn_dir="mitdb")
    annotation = wfdb.rdann(record_id, "atr", pn_dir="mitdb")

    # Use first ECG channel for a simple baseline.
    signal = record.p_signal[:, 0].astype(np.float32)
    x_rows, y_rows = [], []

    for sample, symbol in zip(annotation.sample, annotation.symbol):
        if symbol not in NORMAL_SYMBOLS and symbol not in ABNORMAL_SYMBOLS:
            continue
        start = sample - HALF_WINDOW
        end = sample + HALF_WINDOW
        if start < 0 or end > len(signal):
            continue

        segment = signal[start:end]
        # Normalize each segment to reduce sensitivity to signal scale.
        segment = segment - np.mean(segment)
        std = np.std(segment)
        if std > 1e-8:
            segment = segment / std

        x_rows.append(segment)
        y_rows.append(1 if symbol in ABNORMAL_SYMBOLS else 0)

    return np.asarray(x_rows, dtype=np.float32), np.asarray(y_rows, dtype=np.int64)

print("Downloading and preparing ECG recordings. This may take a few minutes...")
record_data = {}
for rid in RECORDS:
    try:
        X_record, y_record = load_beats(rid)
        if len(y_record):
            record_data[rid] = (X_record, y_record)
        print(f"Record {rid}: {len(y_record)} selected beats")
    except Exception as exc:
        print(f"Skipping record {rid} because of a download/read issue: {exc}")

if len(record_data) < 4:
    raise RuntimeError(
        "Not enough records loaded. Check your internet connection and run the cell again."
    )

# Split by recording, not by individual heartbeat, to reduce data leakage.
record_ids = np.array(list(record_data.keys()))
train_ids, test_ids = train_test_split(
    record_ids, test_size=0.25, random_state=42
)

X_train = np.concatenate([record_data[r][0] for r in train_ids], axis=0)
y_train = np.concatenate([record_data[r][1] for r in train_ids], axis=0)
X_test = np.concatenate([record_data[r][0] for r in test_ids], axis=0)
y_test = np.concatenate([record_data[r][1] for r in test_ids], axis=0)

print(f"\nTraining beats: {len(y_train)} | Test beats: {len(y_test)}")
print("Training class counts (0=selected normal, 1=selected abnormal):",
      np.bincount(y_train, minlength=2))
print("Test class counts (0=selected normal, 1=selected abnormal):",
      np.bincount(y_test, minlength=2))

model = RandomForestClassifier(
    n_estimators=100,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

print("\nTraining model...")
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("\n=== Evaluation on held-out recordings ===")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.3f}")
print("\nClassification report (0=selected normal, 1=selected abnormal):")
print(classification_report(
    y_test, y_pred, labels=[0, 1],
    target_names=["Selected normal beats", "Selected abnormal beats"],
    zero_division=0
))

abnormal_recall = recall_score(y_test, y_pred, pos_label=1, zero_division=0)
print(f"Abnormal-beat recall/sensitivity on this test split: {abnormal_recall:.3f}")
print("Note: this is a limited beat-classification experiment, not patient-level diagnosis.")

cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Selected normal", "Selected abnormal"]
).plot()
plt.title("ECG Beat Classification — Test Recordings")
plt.tight_layout()
plt.show()

# Show one held-out ECG segment and the model's prediction.
example_index = 0
plt.figure(figsize=(10, 3))
plt.plot(X_test[example_index])
plt.title("Example ECG beat segment (normalized)")
plt.xlabel("Sample within segment")
plt.ylabel("Normalized signal")
plt.tight_layout()
plt.show()

example_prediction = model.predict(X_test[example_index].reshape(1, -1))[0]
print("Example model output:",
      "Selected ABNORMAL beat" if example_prediction == 1 else "Selected NORMAL beat")
print("\nReminder: do not use this output to make medical decisions.")
