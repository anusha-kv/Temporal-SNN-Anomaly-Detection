
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path

# ==================================================
# 1. PROJECT SETTINGS
# ==================================================

SEED = 42
rng = np.random.default_rng(SEED)

FS = 1000
DURATION = 2.0

TIME_BINS = 20
HIDDEN_NEURONS = 32

DECAY = 0.80
FIRING_THRESHOLD = 1.0
INPUT_SCALE = 1.0

project_folder = Path(__file__).resolve().parents[1]

data_file = (
    project_folder
    / "data"
    / "spike_encoded_trials.csv"
)

results_folder = project_folder / "results"
results_folder.mkdir(exist_ok=True)

if not data_file.exists():
    raise FileNotFoundError(
        "Run encode_spikes.py first!"
    )

df = pd.read_csv(data_file)

required = [
    "trial_id",
    "condition",
    "split",
    "time",
    "voltage_spike",
    "current_spike",
    "label"
]

for col in required:
    if col not in df.columns:
        raise ValueError(f"Missing column: {col}")

print("\nTEMPORAL SNN")
print("==============================")
print("Dataset loaded:", len(df), "samples")


# ==================================================
# 2. INITIALIZE LIF INPUT WEIGHTS
# ==================================================

# Two input channels:
# 0 = Voltage
# 1 = Current

W_input = rng.uniform(
    0.35,
    0.90,
    size=(2, HIDDEN_NEURONS)
)

print("Input neurons: 2")
print("Hidden LIF neurons:", HIDDEN_NEURONS)


# ==================================================
# 3. TEMPORAL SPIKE ENCODING
# ==================================================

def create_temporal_features(trial):

    time = trial["time"].to_numpy()

    voltage = trial[
        "voltage_spike"
    ].to_numpy()

    current = trial[
        "current_spike"
    ].to_numpy()

    edges = np.linspace(
        0,
        DURATION,
        TIME_BINS + 1
    )

    # Count spikes in each time bin

    voltage_counts, _ = np.histogram(
        time,
        bins=edges,
        weights=voltage
    )

    current_counts, _ = np.histogram(
        time,
        bins=edges,
        weights=current
    )

    # Two-channel temporal input

    temporal_input = np.column_stack([
        voltage_counts,
        current_counts
    ]).astype(float)

    # Normalize input spike counts

    temporal_input /= 5.0

    # ==============================================
    # 4. LIF HIDDEN NEURON SIMULATION
    # ==============================================

    membrane = np.zeros(HIDDEN_NEURONS)

    hidden_spikes = np.zeros(
        (TIME_BINS, HIDDEN_NEURONS)
    )

    for t in range(TIME_BINS):

        input_current = (
            temporal_input[t] @ W_input
        ) * INPUT_SCALE

        # LIF membrane potential update

        membrane = (
            DECAY * membrane
            + input_current
        )

        # Firing condition

        fired = membrane >= FIRING_THRESHOLD

        hidden_spikes[t] = fired.astype(float)

        # Reset after firing

        membrane[fired] = 0.0

    # Preserve temporal information by flattening
    # the complete sequence, not just total spikes.

    features = np.concatenate([
        hidden_spikes.flatten(),
        temporal_input.flatten()
    ])

    return features


# ==================================================
# 5. EXTRACT FEATURES FROM ALL TRIALS
# ==================================================

print("\nExtracting temporal spike features...")

trial_features = []

for trial_id, trial in df.groupby(
    "trial_id",
    sort=False
):

    trial = trial.reset_index(drop=True)

    feature_vector = create_temporal_features(
        trial
    )

    trial_features.append({

        "trial_id": trial_id,

        "condition": trial[
            "condition"
        ].iloc[0],

        "split": trial[
            "split"
        ].iloc[0],

        "label": int(
            trial["label"].iloc[0]
        ),

        "features": feature_vector

    })

feature_df = pd.DataFrame(trial_features)

X = np.vstack(
    feature_df["features"].to_numpy()
)

y = feature_df["label"].to_numpy()

print("Feature matrix:", X.shape)
print("Trials processed:", len(X))


# ==================================================
# 6. SPLIT TRAINING DATA INTO TRAIN / VALIDATION
# ==================================================

train_indices = np.where(
    feature_df["split"].to_numpy() == "train"
)[0]

test_indices = np.where(
    feature_df["split"].to_numpy() == "test"
)[0]

if len(train_indices) == 0 or len(test_indices) == 0:
    raise ValueError("Training or testing trials missing.")

# Split by trial, not by individual time samples.

rng_split = np.random.default_rng(SEED)

fit_indices = []
validation_indices = []

for label in [0, 1]:

    class_indices = train_indices[
        y[train_indices] == label
    ].copy()

    rng_split.shuffle(class_indices)

    validation_size = max(
        1,
        int(0.20 * len(class_indices))
    )

    validation_indices.extend(
        class_indices[:validation_size]
    )

    fit_indices.extend(
        class_indices[validation_size:]
    )

fit_indices = np.array(fit_indices)
validation_indices = np.array(validation_indices)

print("\nDATASET SPLIT")
print("------------------------------")
print("Training:", len(fit_indices))
print("Validation:", len(validation_indices))
print("Testing:", len(test_indices))


# ==================================================
# 7. STANDARDIZE FEATURES
# ==================================================

# Fit scaling parameters using training data only.

X_mean = X[fit_indices].mean(axis=0)

X_std = X[fit_indices].std(axis=0)

X_std[X_std < 1e-8] = 1.0

X_scaled = (X - X_mean) / X_std

# Add bias term

X_scaled = np.column_stack([
    X_scaled,
    np.ones(len(X_scaled))
])


# ==================================================
# 8. TRAIN SYNAPTIC READOUT
# ==================================================

# Supervised ridge regression.
# Hidden LIF input weights remain fixed.
# Output synaptic weights are learned.

target = np.where(y == 1, 1.0, -1.0)

X_fit = X_scaled[fit_indices]
y_fit = target[fit_indices]

X_val = X_scaled[validation_indices]
y_val = y[validation_indices]

alphas = [
    0.01,
    0.1,
    1.0,
    10.0,
    100.0
]

best_alpha = None
best_threshold = None
best_weights = None
best_validation_score = -1

print("\nTRAINING SYNAPTIC READOUT...")
print("------------------------------")

for alpha in alphas:

    # Dual ridge regression solution

    kernel = X_fit @ X_fit.T

    regularized = (
        kernel
        + alpha * np.eye(len(X_fit))
    )

    coefficients = np.linalg.solve(
        regularized,
        y_fit
    )

    weights = X_fit.T @ coefficients

    validation_scores = X_val @ weights

    candidates = np.unique(
        np.concatenate([
            validation_scores,
            [
                validation_scores.min() - 1e-8,
                validation_scores.max() + 1e-8
            ]
        ])
    )

    for threshold in candidates:

        predictions = (
            validation_scores > threshold
        ).astype(int)

        actual = y_val

        tp = np.sum(
            (actual == 1) & (predictions == 1)
        )

        tn = np.sum(
            (actual == 0) & (predictions == 0)
        )

        fp = np.sum(
            (actual == 0) & (predictions == 1)
        )

        fn = np.sum(
            (actual == 1) & (predictions == 0)
        )

        sensitivity = (
            tp / (tp + fn)
            if tp + fn else 0
        )

        specificity = (
            tn / (tn + fp)
            if tn + fp else 0
        )

        balanced_accuracy = (
            sensitivity + specificity
        ) / 2

        if balanced_accuracy > best_validation_score:

            best_validation_score = balanced_accuracy

            best_alpha = alpha

            best_threshold = threshold

            best_weights = weights.copy()


print("Best regularization:", best_alpha)

print(
    "Validation balanced accuracy:",
    round(best_validation_score * 100, 2),
    "%"
)

print("Selected threshold:", best_threshold)


# ==================================================
# 9. FINAL TEST ON UNSEEN TRIALS
# ==================================================

X_test = X_scaled[test_indices]

y_test = y[test_indices]

test_scores = X_test @ best_weights

predictions = (
    test_scores > best_threshold
).astype(int)

# ==================================================
# 10. CALCULATE PERFORMANCE
# ==================================================

TP = np.sum(
    (y_test == 1) & (predictions == 1)
)

TN = np.sum(
    (y_test == 0) & (predictions == 0)
)

FP = np.sum(
    (y_test == 0) & (predictions == 1)
)

FN = np.sum(
    (y_test == 1) & (predictions == 0)
)

accuracy = (TP + TN) / len(y_test)

precision = (
    TP / (TP + FP)
    if TP + FP else 0
)

recall = (
    TP / (TP + FN)
    if TP + FN else 0
)

f1 = (
    2 * precision * recall / (precision + recall)
    if precision + recall else 0
)

specificity = (
    TN / (TN + FP)
    if TN + FP else 0
)

print("\nTEMPORAL SNN TEST RESULTS")
print("==============================")

print("True positives:", TP)
print("True negatives:", TN)
print("False positives:", FP)
print("False negatives:", FN)

print("\nAccuracy:", round(accuracy * 100, 2), "%")
print("Precision:", round(precision * 100, 2), "%")
print("Recall:", round(recall * 100, 2), "%")
print("Specificity:", round(specificity * 100, 2), "%")
print("F1-score:", round(f1 * 100, 2), "%")


# ==================================================
# 11. CONDITION-WISE CLASSIFICATION
# ==================================================

test_results = feature_df.iloc[
    test_indices
][[
    "trial_id",
    "condition",
    "label"
]].copy()

test_results["prediction"] = predictions

test_results["score"] = test_scores

test_results["actual_class"] = (
    test_results["label"].map({
        0: "NORMAL",
        1: "ABNORMAL"
    })
)

test_results["predicted_class"] = (
    test_results["prediction"].map({
        0: "NORMAL",
        1: "ABNORMAL"
    })
)

print("\nCONDITION-WISE RESULTS")
print("------------------------------")

for condition, group in test_results.groupby("condition"):

    correct = np.sum(
        group["label"] == group["prediction"]
    )

    print(
        f"{condition}: {correct}/{len(group)} correct"
    )


# ==================================================
# 12. CONFUSION MATRIX
# ==================================================

cm = np.array([
    [TN, FP],
    [FN, TP]
])

fig, ax = plt.subplots(figsize=(7, 6))

im = ax.imshow(cm)

ax.set_title(
    "Temporal SNN Confusion Matrix"
)

ax.set_xlabel("Predicted Class")
ax.set_ylabel("Actual Class")

ax.set_xticks([0, 1])
ax.set_yticks([0, 1])

ax.set_xticklabels([
    "Normal",
    "Abnormal"
])

ax.set_yticklabels([
    "Normal",
    "Abnormal"
])

for i in range(2):
    for j in range(2):

        ax.text(
            j,
            i,
            str(cm[i, j]),
            ha="center",
            va="center",
            fontsize=16
        )

plt.colorbar(im, ax=ax)

plt.tight_layout()

plt.savefig(
    results_folder / "temporal_confusion_matrix.png",
    dpi=300
)

plt.show()


# ==================================================
# 13. TEMPORAL FEATURE VISUALIZATION
# ==================================================

plt.figure(figsize=(10, 5))

hidden_feature_count = (
    TIME_BINS * HIDDEN_NEURONS
)

sample = X[0][:hidden_feature_count].reshape(
    TIME_BINS,
    HIDDEN_NEURONS
)

plt.imshow(
    sample.T,
    aspect="auto",
    interpolation="nearest"
)

plt.title("Temporal LIF Hidden Neuron Activity")

plt.xlabel("Time Bin")
plt.ylabel("Hidden Neuron")

plt.colorbar(
    label="Spike activity"
)

plt.tight_layout()

plt.savefig(
    results_folder / "temporal_lif_activity.png",
    dpi=300
)

plt.show()


# ==================================================
# 14. SAVE TRAINED MODEL
# ==================================================

np.savez(
    results_folder / "temporal_snn_model.npz",

    W_input=W_input,

    W_output=best_weights,

    X_mean=X_mean,

    X_std=X_std,

    threshold=best_threshold,

    alpha=best_alpha,

    decay=DECAY,

    firing_threshold=FIRING_THRESHOLD,

    time_bins=TIME_BINS,

    hidden_neurons=HIDDEN_NEURONS
)


# ==================================================
# 15. SAVE TEST RESULTS AND METRICS
# ==================================================

test_results.to_csv(
    results_folder / "temporal_test_results.csv",
    index=False
)

metrics = pd.DataFrame([{

    "accuracy": accuracy,

    "precision": precision,

    "recall": recall,

    "specificity": specificity,

    "f1_score": f1,

    "validation_balanced_accuracy": best_validation_score,

    "regularization": best_alpha,

    "threshold": best_threshold

}])

metrics.to_csv(
    results_folder / "temporal_snn_metrics.csv",
    index=False
)


# ==================================================
# 16. FINAL SUMMARY
# ==================================================

print("\n===================================")
print(" TEMPORAL SNN COMPLETED")
print("===================================")

print("Training trials:", len(fit_indices))
print("Validation trials:", len(validation_indices))
print("Testing trials:", len(test_indices))

print("Accuracy:", round(accuracy * 100, 2), "%")

print("\nSaved files:")
print("- temporal_snn_model.npz")
print("- temporal_snn_metrics.csv")
print("- temporal_test_results.csv")
print("- temporal_confusion_matrix.png")
print("- temporal_lif_activity.png")

print("\nStage 8 completed!")
