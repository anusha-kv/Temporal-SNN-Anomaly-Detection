"""
Stage 8.1 - Temporal SNN
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
from sklearn.linear_model import Ridge
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    balanced_accuracy_score,
    confusion_matrix
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

SEED = 42
TIME_BINS = 20
HIDDEN_NEURONS = 32
DECAY = 0.8
FIRING_THRESHOLD = 1.0
INPUT_SCALE = 1.0

ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = ROOT / "data" / "spike_encoded_trials.csv"
RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)

print("Stage 8.1 Temporal SNN started")
print("Project folder:", ROOT)
print("Dataset:", DATA_FILE)
# Load dataset
if not DATA_FILE.exists():
    raise FileNotFoundError(f"Dataset not found: {DATA_FILE}")

df = pd.read_csv(DATA_FILE)

required_columns = [
    "trial_id",
    "condition",
    "split",
    "time",
    "voltage_spike",
    "current_spike",
    "label"
]

for col in required_columns:
    if col not in df.columns:
        raise ValueError(f"Missing column: {col}")

print("Dataset loaded successfully!")
print("Total rows:", len(df))
print("Total trials:", df["trial_id"].nunique())
print(df.head())
# Temporal SNN feature extraction

FS = 1000
DURATION = 2.0
INPUT_COUNT_NORMALIZER = 5.0

rng = np.random.default_rng(SEED)

W_input = rng.uniform(
    0.35, 0.90,
    size=(2, HIDDEN_NEURONS)
)


def extract_features(trial):

    trial = trial.sort_values("time")

    time = trial["time"].to_numpy()
    voltage = trial["voltage_spike"].to_numpy()
    current = trial["current_spike"].to_numpy()

    if np.any(time < 0) or np.any(time > DURATION):
        raise ValueError("Trial time is outside the expected duration.")

    bin_edges = np.linspace(
        0, DURATION, TIME_BINS + 1
    )

    input_features = np.zeros((TIME_BINS, 2))

    for i in range(TIME_BINS):
        mask = (
            (time >= bin_edges[i]) &
            (time < bin_edges[i + 1])
        )

        if i == TIME_BINS - 1:
            mask |= (time == bin_edges[i + 1])

        input_features[i, 0] = voltage[mask].sum()
        input_features[i, 1] = current[mask].sum()

    input_features /= INPUT_COUNT_NORMALIZER
    input_features *= INPUT_SCALE

    membrane = np.zeros(HIDDEN_NEURONS)
    hidden_spikes = np.zeros(
        (TIME_BINS, HIDDEN_NEURONS)
    )

    for t in range(TIME_BINS):

        input_current = input_features[t] @ W_input

        membrane = DECAY * membrane + input_current

        fired = membrane >= FIRING_THRESHOLD

        hidden_spikes[t, fired] = 1
        membrane[fired] = 0

    return np.concatenate([
        hidden_spikes.flatten(),
        input_features.flatten()
    ])


print("Feature extraction function created.")
# Prepare trial features and labels

X = []
y = []
trial_splits = []
conditions = []

for trial_id, trial in df.groupby("trial_id"):

    trial = trial.sort_values("time")

    features = extract_features(trial)

    labels = trial["label"].unique()
    splits = trial["split"].unique()
    names = trial["condition"].unique()

    if len(labels) != 1 or len(splits) != 1 or len(names) != 1:
        raise ValueError(f"Inconsistent data in trial {trial_id}")

    X.append(features)
    y.append(int(labels[0]))
    trial_splits.append(str(splits[0]).lower())
    conditions.append(str(names[0]))

X = np.asarray(X)
y = np.asarray(y)
trial_splits = np.asarray(trial_splits)
conditions = np.asarray(conditions)

print("Feature preparation completed.")
print("Feature shape:", X.shape)
print("Labels:", np.unique(y))
print("Splits:", np.unique(trial_splits))
# Split training and testing data

train_indices = np.where(trial_splits == "train")[0]
test_indices = np.where(trial_splits == "test")[0]

if len(train_indices) == 0 or len(test_indices) == 0:
    raise ValueError("Training or testing data is missing.")

fit_indices, val_indices = train_test_split(
    train_indices,
    test_size=0.20,
    random_state=SEED,
    stratify=y[train_indices]
)

# Standardize features
scaler = StandardScaler()

X_fit = scaler.fit_transform(X[fit_indices])
X_val = scaler.transform(X[val_indices])
X_test = scaler.transform(X[test_indices])

y_fit = y[fit_indices]
y_val = y[val_indices]
y_test = y[test_indices]

print("Training samples:", len(fit_indices))
print("Validation samples:", len(val_indices))
print("Testing samples:", len(test_indices))
# Train ridge classifier

best_model = None
best_alpha = None
best_threshold = None
best_score = -1

for alpha in [0.01, 0.1, 1.0, 10.0, 100.0]:

    model = Ridge(alpha=alpha)
    model.fit(X_fit, y_fit)

    val_scores = model.predict(X_val)

    for threshold in np.unique(val_scores):
        predictions = (val_scores >= threshold).astype(int)

        score = balanced_accuracy_score(
            y_val, predictions
        )

        if score > best_score:
            best_score = score
            best_model = model
            best_alpha = alpha
            best_threshold = threshold

print("Training completed!")
print("Best alpha:", best_alpha)
print("Best threshold:", best_threshold)
print("Validation balanced accuracy:", best_score)
# Evaluate on independent test data

test_scores = best_model.predict(X_test)

test_predictions = (
    test_scores >= best_threshold
).astype(int)

cm = confusion_matrix(
    y_test, test_predictions, labels=[0, 1]
)

accuracy = accuracy_score(y_test, test_predictions)
precision = precision_score(y_test, test_predictions, zero_division=0)
recall = recall_score(y_test, test_predictions, zero_division=0)
f1 = f1_score(y_test, test_predictions, zero_division=0)
balanced_acc = balanced_accuracy_score(y_test, test_predictions)

print("\nTEST RESULTS")
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)
print("Balanced Accuracy:", balanced_acc)

print("\nConfusion Matrix:")
print(cm)
# Save model and results

np.savez(
    RESULTS / "temporal_snn_v2_model.npz",
    W_input=W_input,
    W_output=best_model.coef_,
    intercept=best_model.intercept_,
    X_mean=scaler.mean_,
    X_std=scaler.scale_,
    threshold=best_threshold,
    alpha=best_alpha,
    decay=DECAY,
    time_bins=TIME_BINS,
    hidden_neurons=HIDDEN_NEURONS,
    input_scale=INPUT_SCALE,
    duration=DURATION,
    fs=FS
)

pd.DataFrame([{
    "accuracy": accuracy,
    "precision": precision,
    "recall": recall,
    "f1": f1,
    "balanced_accuracy": balanced_acc
}]).to_csv(
    RESULTS / "temporal_v2_metrics.csv",
    index=False
)

print("\nModel and metrics saved successfully!")
# Save confusion matrix plot

fig, ax = plt.subplots(figsize=(6, 5))
ax.imshow(cm)

ax.set_title("Temporal SNN v2 - Confusion Matrix")
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
ax.set_xticks([0, 1], labels=["Normal", "Abnormal"])
ax.set_yticks([0, 1], labels=["Normal", "Abnormal"])

for i in range(2):
    for j in range(2):
        ax.text(j, i, str(cm[i, j]), ha="center", va="center")

plt.tight_layout()
plt.savefig(RESULTS / "temporal_v2_confusion_matrix.png")
plt.close()

# Save individual predictions
pd.DataFrame({
    "condition": conditions[test_indices],
    "actual": y_test,
    "predicted": test_predictions
}).to_csv(
    RESULTS / "temporal_v2_test_results.csv",
    index=False
)

# Save condition-wise results
condition_results = []

for condition in np.unique(conditions[test_indices]):
    mask = conditions[test_indices] == condition

    condition_results.append({
        "condition": condition,
        "total": int(mask.sum()),
        "correct": int(
            (y_test[mask] == test_predictions[mask]).sum()
        ),
        "accuracy": accuracy_score(
            y_test[mask], test_predictions[mask]
        )
    })

pd.DataFrame(condition_results).to_csv(
    RESULTS / "temporal_v2_condition_results.csv",
    index=False
)

print("\nAll results saved in:", RESULTS)
print("Stage 8.1 completed!")