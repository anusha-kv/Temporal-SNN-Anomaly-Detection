
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from brian2 import *
from pathlib import Path

# ==========================================
# 1. INITIAL SETTINGS
# ==========================================

prefs.codegen.target = "numpy"

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
        "spike_encoded_trials.csv not found. "
        "Run encode_spikes.py first."
    )

df = pd.read_csv(data_file)

required_columns = [
    "trial_id",
    "condition",
    "split",
    "time",
    "voltage_spike",
    "current_spike",
    "label"
]

for column in required_columns:
    if column not in df.columns:
        raise ValueError(
            f"Missing required column: {column}"
        )

print("Encoded electrical dataset loaded!")
print("Total samples:", len(df))
print("Total trials:", df["trial_id"].nunique())


# ==========================================
# 2. DEFINE LIF NETWORK
# ==========================================

def run_lif_trial(trial_data):

    start_scope()

    defaultclock.dt = 0.1 * ms

    # Extract spike information

    voltage_spikes = trial_data[
        "voltage_spike"
    ].to_numpy(dtype=int)

    current_spikes = trial_data[
        "current_spike"
    ].to_numpy(dtype=int)

    time_values = trial_data[
        "time"
    ].to_numpy(dtype=float)

    # Create input neuron indices

    indices = np.concatenate([
        np.zeros(
            np.sum(voltage_spikes),
            dtype=int
        ),
        np.ones(
            np.sum(current_spikes),
            dtype=int
        )
    ])

    # Get spike event times

    times = np.concatenate([
        time_values[voltage_spikes == 1],
        time_values[current_spikes == 1]
    ])

    # Sort events chronologically

    order = np.argsort(times, kind="stable")

    indices = indices[order]
    times = times[order]

    # Remove duplicate events from the same neuron
    # at the same time, if present.

    if len(times) > 0:

        unique_events = np.concatenate([
            [True],
            (
                (indices[1:] != indices[:-1])
                |
                (times[1:] != times[:-1])
            )
        ])

        indices = indices[unique_events]
        times = times[unique_events]

    # Create two input spike neurons

    input_neurons = SpikeGeneratorGroup(
        2,
        indices=indices,
        times=times * second
    )

    # Define LIF neuron

    tau = 20 * ms

    equations = '''
    dv/dt = -v/tau : 1
    '''

    lif_neuron = NeuronGroup(
        1,
        equations,
        threshold="v >= 1",
        reset="v = 0",
        refractory=5*ms,
        method="exact"
    )

    lif_neuron.v = 0

    # Connect input neurons

    synapses = Synapses(
        input_neurons,
        lif_neuron,
        on_pre="v_post += 0.6"
    )

    synapses.connect()

    # Record output spikes

    spike_monitor = SpikeMonitor(lif_neuron)

    # Run the simulation

    simulation_duration = (
        time_values.max() + 0.001
    )

    run(simulation_duration * second)

    return int(spike_monitor.num_spikes)


# ==========================================
# 3. PROCESS ALL TRIALS
# ==========================================

trial_results = []

print("\nRunning LIF simulations...")
print("Please wait while all trials are processed.")

for trial_id, trial_data in df.groupby(
    "trial_id",
    sort=False
):

    trial_data = trial_data.reset_index(drop=True)

    output_spikes = run_lif_trial(trial_data)

    trial_results.append({

        "trial_id": trial_id,

        "condition": trial_data[
            "condition"
        ].iloc[0],

        "split": trial_data[
            "split"
        ].iloc[0],

        "label": int(
            trial_data["label"].iloc[0]
        ),

        "output_spikes": output_spikes

    })

    print(
        f"{trial_id}: {output_spikes} output spikes"
    )


results = pd.DataFrame(trial_results)


# ==========================================
# 4. SEPARATE TRAINING AND TESTING DATA
# ==========================================

train = results[
    results["split"] == "train"
].copy()

test = results[
    results["split"] == "test"
].copy()

if train.empty or test.empty:
    raise ValueError(
        "Training or testing dataset is empty."
    )

if set(train["trial_id"]) & set(test["trial_id"]):
    raise ValueError(
        "Data leakage detected: trial IDs overlap."
    )

print("\nTRAINING TRIALS:", len(train))
print("TESTING TRIALS:", len(test))


# ==========================================
# 5. CALIBRATE THRESHOLD USING TRAINING DATA
# ==========================================

train_counts = train[
    "output_spikes"
].to_numpy()

train_labels = train[
    "label"
].to_numpy()

unique_counts = np.unique(train_counts)

candidates = np.concatenate([
    [unique_counts[0] - 0.5],
    (unique_counts[:-1] + unique_counts[1:]) / 2,
    [unique_counts[-1] + 0.5]
])

best_threshold = None
best_balanced_accuracy = -1

for threshold in candidates:

    predictions = (
        train_counts > threshold
    ).astype(int)

    true_positive = np.sum(
        (train_labels == 1)
        & (predictions == 1)
    )

    true_negative = np.sum(
        (train_labels == 0)
        & (predictions == 0)
    )

    false_positive = np.sum(
        (train_labels == 0)
        & (predictions == 1)
    )

    false_negative = np.sum(
        (train_labels == 1)
        & (predictions == 0)
    )

    sensitivity = (
        true_positive
        / (true_positive + false_negative)
        if true_positive + false_negative > 0
        else 0
    )

    specificity = (
        true_negative
        / (true_negative + false_positive)
        if true_negative + false_positive > 0
        else 0
    )

    balanced_accuracy = (
        sensitivity + specificity
    ) / 2

    if balanced_accuracy > best_balanced_accuracy:

        best_balanced_accuracy = balanced_accuracy
        best_threshold = threshold

print("\nTRAINING CALIBRATION")
print("----------------------------")

print(
    "Selected spike threshold:",
    best_threshold
)

print(
    "Training balanced accuracy:",
    round(best_balanced_accuracy * 100, 2),
    "%"
)


# ==========================================
# 6. TEST ON UNSEEN TRIALS
# ==========================================

test["prediction"] = (
    test["output_spikes"] > best_threshold
).astype(int)

test["prediction_name"] = test[
    "prediction"
].map({
    0: "NORMAL",
    1: "ABNORMAL"
})

test["actual_name"] = test[
    "label"
].map({
    0: "NORMAL",
    1: "ABNORMAL"
})


# ==========================================
# 7. CALCULATE PERFORMANCE METRICS
# ==========================================

y_true = test["label"].to_numpy()
y_pred = test["prediction"].to_numpy()

TP = np.sum((y_true == 1) & (y_pred == 1))
TN = np.sum((y_true == 0) & (y_pred == 0))
FP = np.sum((y_true == 0) & (y_pred == 1))
FN = np.sum((y_true == 1) & (y_pred == 0))

accuracy = (
    (TP + TN) / len(y_true)
    if len(y_true) else 0
)

precision = (
    TP / (TP + FP)
    if TP + FP else 0
)

recall = (
    TP / (TP + FN)
    if TP + FN else 0
)

f1_score = (
    2 * precision * recall / (precision + recall)
    if precision + recall else 0
)

print("\nTEST PERFORMANCE")
print("============================")

print("True Positives:", TP)
print("True Negatives:", TN)
print("False Positives:", FP)
print("False Negatives:", FN)

print(
    "Accuracy:",
    round(accuracy * 100, 2), "%"
)

print(
    "Precision:",
    round(precision * 100, 2), "%"
)

print(
    "Recall:",
    round(recall * 100, 2), "%"
)

print(
    "F1-score:",
    round(f1_score * 100, 2), "%"
)


# ==========================================
# 8. CONDITION-WISE TEST RESULTS
# ==========================================

print("\nCONDITION-WISE TEST RESULTS")
print("============================")

for condition, group in test.groupby("condition"):

    correct = np.sum(
        group["label"] == group["prediction"]
    )

    total = len(group)

    print(
        f"{condition}: "
        f"{correct}/{total} correct"
    )


# ==========================================
# 9. CREATE CONFUSION MATRIX
# ==========================================

confusion_matrix = np.array([
    [TN, FP],
    [FN, TP]
])

fig, ax = plt.subplots(figsize=(7, 6))

image = ax.imshow(confusion_matrix)

ax.set_title("SNN Anomaly Detection Confusion Matrix")

ax.set_xlabel("Predicted Class")
ax.set_ylabel("Actual Class")

ax.set_xticks([0, 1])
ax.set_yticks([0, 1])

ax.set_xticklabels(["Normal", "Abnormal"])
ax.set_yticklabels(["Normal", "Abnormal"])

for i in range(2):
    for j in range(2):

        ax.text(
            j,
            i,
            str(confusion_matrix[i, j]),
            ha="center",
            va="center",
            fontsize=16
        )

plt.colorbar(image, ax=ax)

plt.tight_layout()

plt.savefig(
    results_folder / "confusion_matrix.png",
    dpi=300
)

plt.show()


# ==========================================
# 10. PLOT TEST SPIKE COUNTS
# ==========================================

plt.figure(figsize=(11, 6))

for condition, group in test.groupby("condition"):

    plt.scatter(
        [condition] * len(group),
        group["output_spikes"],
        label=condition
    )

plt.axhline(
    best_threshold,
    linestyle="--",
    color="black",
    label="Training threshold"
)

plt.title("SNN Output Spike Counts on Test Trials")
plt.xlabel("Electrical Condition")
plt.ylabel("Output Spike Count")

plt.legend()
plt.grid(True, axis="y")

plt.tight_layout()

plt.savefig(
    results_folder / "test_spike_counts.png",
    dpi=300
)

plt.show()


# ==========================================
# 11. SAVE TEST RESULTS
# ==========================================

test.to_csv(
    results_folder / "test_classification_results.csv",
    index=False
)

results.to_csv(
    results_folder / "all_trial_spike_counts.csv",
    index=False
)

metrics = pd.DataFrame([{
    "threshold": best_threshold,
    "training_balanced_accuracy": best_balanced_accuracy,
    "accuracy": accuracy,
    "precision": precision,
    "recall": recall,
    "f1_score": f1_score
}])

metrics.to_csv(
    results_folder / "performance_metrics.csv",
    index=False
)


# ==========================================
# 12. FINAL SUMMARY
# ==========================================

print("\n===================================")
print(" SNN EVALUATION COMPLETED")
print("===================================")

print("Training trials:", len(train))
print("Testing trials:", len(test))

print("Selected threshold:", best_threshold)

print("Accuracy:", round(accuracy * 100, 2), "%")
print("Precision:", round(precision * 100, 2), "%")
print("Recall:", round(recall * 100, 2), "%")
print("F1-score:", round(f1_score * 100, 2), "%")

print("\nFiles saved in results folder:")
print("- confusion_matrix.png")
print("- test_spike_counts.png")
print("- test_classification_results.csv")
print("- all_trial_spike_counts.csv")
print("- performance_metrics.csv")

print("\nSNN evaluation finished successfully!")
