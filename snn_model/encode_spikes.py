
import numpy as np
import pandas as pd
from pathlib import Path

# ==========================================
# 1. LOAD ELECTRICAL DATASET
# ==========================================

project_folder = Path(__file__).resolve().parents[1]

input_file = project_folder / "data" / "electrical_trials.csv"

df = pd.read_csv(input_file)

print("Electrical trial dataset loaded!")
print("Total samples:", len(df))

# ==========================================
# 2. SPIKE ENCODING PARAMETERS
# ==========================================

sampling_rate = 1000
dt = 1 / sampling_rate

max_spike_rate = 100

rng = np.random.default_rng(42)

# ==========================================
# 3. ENCODE EACH TRIAL
# ==========================================

encoded_trials = []

for trial_id, trial_data in df.groupby("trial_id", sort=False):

    trial_data = trial_data.reset_index(drop=True)

    voltage = trial_data["voltage"].to_numpy()
    current = trial_data["current"].to_numpy()

    # Calculate deviations from normal values

    voltage_deviation = np.clip(
        np.abs(voltage - 48) / 12,
        0,
        1
    )

    current_deviation = np.clip(
        np.abs(current - 2) / 3,
        0,
        1
    )

    # Convert deviations into firing probabilities

    voltage_probability = (
        voltage_deviation * max_spike_rate * dt
    )

    current_probability = (
        current_deviation * max_spike_rate * dt
    )

    # Generate stochastic spikes

    voltage_spikes = (
        rng.random(len(voltage))
        < voltage_probability
    ).astype(int)

    current_spikes = (
        rng.random(len(current))
        < current_probability
    ).astype(int)

    # Create encoded trial

    encoded = trial_data[
        [
            "trial_id",
            "condition",
            "split",
            "time",
            "label"
        ]
    ].copy()

    encoded["voltage_deviation"] = voltage_deviation
    encoded["current_deviation"] = current_deviation

    encoded["voltage_spike"] = voltage_spikes
    encoded["current_spike"] = current_spikes

    encoded_trials.append(encoded)

# ==========================================
# 4. COMBINE ENCODED TRIALS
# ==========================================

spike_dataset = pd.concat(
    encoded_trials,
    ignore_index=True
)

# ==========================================
# 5. DISPLAY SPIKE STATISTICS
# ==========================================

print("\nSPIKE ENCODING SUMMARY")
print("--------------------------------")

summary = spike_dataset.groupby(
    ["split", "condition"]
).agg(
    trials=("trial_id", "nunique"),
    voltage_spikes=("voltage_spike", "sum"),
    current_spikes=("current_spike", "sum")
)

print(summary)

# ==========================================
# 6. SAVE ENCODED DATASET
# ==========================================

output_file = (
    project_folder
    / "data"
    / "spike_encoded_trials.csv"
)

spike_dataset.to_csv(
    output_file,
    index=False
)

print("\nSpike encoding completed!")
print("Total encoded samples:", len(spike_dataset))
print("Saved to:", output_file)

# ==========================================
# 7. VERIFY TRAIN/TEST SEPARATION
# ==========================================

train_ids = set(
    spike_dataset.loc[
        spike_dataset["split"] == "train",
        "trial_id"
    ]
)

test_ids = set(
    spike_dataset.loc[
        spike_dataset["split"] == "test",
        "trial_id"
    ]
)

assert train_ids.isdisjoint(test_ids)

print("\nTraining trials:", len(train_ids))
print("Testing trials:", len(test_ids))
print("Training/testing separation verified!")
