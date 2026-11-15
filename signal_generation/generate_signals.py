
import numpy as np
import pandas as pd
from pathlib import Path

# ==========================================
# 1. SIMULATION SETTINGS
# ==========================================

FS = 1000
DURATION = 2.0

TIME = np.arange(0, DURATION, 1 / FS)

TRAIN_TRIALS = 50
TEST_TRIALS = 20

rng = np.random.default_rng(42)

# ==========================================
# 2. GENERATE ONE ELECTRICAL TRIAL
# ==========================================

def generate_trial(condition, trial_number, split):

    # Small variations in electrical measurements
    voltage_noise = rng.normal(0, 0.15, len(TIME))
    current_noise = rng.normal(0, 0.025, len(TIME))

    voltage = (
        48
        + 0.2 * np.sin(2 * np.pi * 2 * TIME)
        + voltage_noise
    )

    current = (
        2
        + 0.05 * np.sin(2 * np.pi * 2 * TIME)
        + current_noise
    )

    # Random abnormal event timing
    start = rng.uniform(0.5, 0.9)
    duration = rng.uniform(0.3, 0.7)
    end = min(start + duration, 1.8)

    event = (TIME >= start) & (TIME <= end)

    # ======================================
    # NORMAL CONDITION
    # ======================================

    if condition == "Normal":
        pass

    # ======================================
    # VOLTAGE SAG
    # ======================================

    elif condition == "Voltage_Sag":

        sag_depth = rng.uniform(8, 14)

        voltage[event] -= sag_depth

    # ======================================
    # CURRENT SURGE
    # ======================================

    elif condition == "Current_Surge":

        surge_amount = rng.uniform(2, 4)

        current[event] += surge_amount

    # ======================================
    # VOLTAGE OSCILLATION
    # ======================================

    elif condition == "Oscillation":

        amplitude = rng.uniform(2.5, 5)
        frequency = rng.uniform(15, 30)

        voltage[event] += (
            amplitude
            * np.sin(2 * np.pi * frequency * TIME[event])
        )

    return pd.DataFrame({
        "trial_id": f"{condition}_{trial_number}",
        "condition": condition,
        "split": split,
        "time": TIME,
        "voltage": voltage,
        "current": current,
        "label": int(condition != "Normal")
    })


# ==========================================
# 3. GENERATE ALL TRIALS
# ==========================================

conditions = [
    "Normal",
    "Voltage_Sag",
    "Current_Surge",
    "Oscillation"
]

all_trials = []

for condition in conditions:

    for trial in range(TRAIN_TRIALS + TEST_TRIALS):

        if trial < TRAIN_TRIALS:
            split = "train"
        else:
            split = "test"

        trial_data = generate_trial(
            condition,
            trial + 1,
            split
        )

        all_trials.append(trial_data)

# ==========================================
# 4. COMBINE DATASETS
# ==========================================

dataset = pd.concat(
    all_trials,
    ignore_index=True
)

# Shuffle trial order without mixing samples
trial_order = dataset[
    ["trial_id", "split"]
].drop_duplicates().sample(
    frac=1,
    random_state=42
)

dataset = dataset.merge(
    trial_order[["trial_id"]],
    on="trial_id",
    how="inner"
)

# ==========================================
# 5. SAVE DATASET
# ==========================================

project_folder = Path(__file__).resolve().parents[1]

data_folder = project_folder / "data"
data_folder.mkdir(exist_ok=True)

output_file = data_folder / "electrical_trials.csv"

dataset.to_csv(output_file, index=False)

# ==========================================
# 6. DISPLAY DATASET INFORMATION
# ==========================================

print("\nELECTRICAL DATASET GENERATED")
print("--------------------------------")

print("Total samples:", len(dataset))
print("Total trials:", dataset["trial_id"].nunique())

print("\nTrials by condition:")
print(
    dataset.groupby(
        ["split", "condition"]
    )["trial_id"].nunique()
)

print("\nDataset saved to:", output_file)
