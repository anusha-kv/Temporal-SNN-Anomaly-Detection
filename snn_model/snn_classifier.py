
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from brian2 import *
from pathlib import Path

# ==========================================
# 1. INITIAL SETTINGS
# ==========================================

prefs.codegen.target = 'numpy'

project_folder = Path(__file__).resolve().parents[1]

data_file = project_folder / "data" / "spike_encoded_signals.csv"

results_folder = project_folder / "results"
results_folder.mkdir(exist_ok=True)

df = pd.read_csv(data_file)

print("Electrical spike dataset loaded!")
print("Total samples:", len(df))

# ==========================================
# 2. DEFINE LIF NETWORK
# ==========================================

def run_snn(condition_data):

    start_scope()

    defaultclock.dt = 0.1 * ms

    # Extract input spike events
    voltage_spikes = condition_data[
        "voltage_spike"
    ].to_numpy()

    current_spikes = condition_data[
        "current_spike"
    ].to_numpy()

    time_values = condition_data[
        "time"
    ].to_numpy()

    voltage_times = time_values[
        voltage_spikes == 1
    ]

    current_times = time_values[
        current_spikes == 1
    ]

    # Combine voltage and current spike events
    indices = np.concatenate([
        np.zeros(len(voltage_times), dtype=int),
        np.ones(len(current_times), dtype=int)
    ])

    times = np.concatenate([
        voltage_times,
        current_times
    ])

    # Sort spike events by time
    order = np.argsort(times, kind="stable")

    indices = indices[order]
    times = times[order]

    # Create two input spike neurons
    input_neurons = SpikeGeneratorGroup(
        2,
        indices=indices,
        times=times * second
    )

    # LIF neuron model
    tau = 20 * ms

    equations = '''
    dv/dt = -v/tau : 1
    '''

    lif_neuron = NeuronGroup(
        1,
        equations,
        threshold='v >= 1',
        reset='v = 0',
        refractory=5*ms,
        method='exact'
    )

    lif_neuron.v = 0

    # Connect input neurons to output neuron
    synapses = Synapses(
        input_neurons,
        lif_neuron,
        on_pre='v_post += 0.6'
    )

    synapses.connect()

    # Record output activity
    spike_monitor = SpikeMonitor(lif_neuron)

    voltage_monitor = StateMonitor(
        lif_neuron,
        'v',
        record=True
    )

    # Run for 2 seconds
    run(2 * second)

    output_spikes = spike_monitor.num_spikes

    return (
        output_spikes,
        voltage_monitor.t / ms,
        voltage_monitor.v[0]
    )

# ==========================================
# 3. PROCESS ALL ELECTRICAL CONDITIONS
# ==========================================

conditions = [
    "Normal",
    "Voltage_Sag",
    "Current_Surge",
    "Oscillation"
]

results = {}

for condition in conditions:

    condition_data = df[
        df["condition"] == condition
    ].reset_index(drop=True)

    if condition_data.empty:
        raise ValueError(
            f"No data found for {condition}"
        )

    spikes, voltage_time, membrane = run_snn(
        condition_data
    )

    results[condition] = {
        "spikes": spikes,
        "time": voltage_time,
        "membrane": membrane
    }

    print(
        condition,
        "-> Output spikes:",
        spikes
    )

# ==========================================
# 4. NORMAL BASELINE CALIBRATION
# ==========================================

normal_spikes = results["Normal"]["spikes"]

# Initial experimental threshold
anomaly_threshold = normal_spikes + 2

print("\nAnomaly detection threshold:", anomaly_threshold)

# ==========================================
# 5. CLASSIFY ELECTRICAL CONDITIONS
# ==========================================

classification_results = []

for condition in conditions:

    output_spikes = results[condition]["spikes"]

    if output_spikes > anomaly_threshold:
        prediction = "ABNORMAL"
    else:
        prediction = "NORMAL"

    classification_results.append({
        "Condition": condition,
        "Output_Spikes": output_spikes,
        "Prediction": prediction
    })

    print(
        condition,
        "=>",
        prediction
    )

# ==========================================
# 6. PLOT SNN OUTPUT ACTIVITY
# ==========================================

fig, axes = plt.subplots(
    4, 1,
    figsize=(11, 10)
)

for ax, condition in zip(axes, conditions):

    result = results[condition]

    ax.plot(
        result["time"],
        result["membrane"]
    )

    ax.axhline(
        1,
        linestyle="--",
        label="Firing threshold"
    )

    ax.set_title(condition)
    ax.set_ylabel("Membrane potential")
    ax.set_xlabel("Time (ms)")
    ax.grid(True)
    ax.legend()

plt.tight_layout()

plt.savefig(
    results_folder / "snn_membrane_activity.png"
)

plt.show()

# ==========================================
# 7. PLOT OUTPUT SPIKE COMPARISON
# ==========================================

names = list(results.keys())

spike_counts = [
    results[name]["spikes"]
    for name in names
]

plt.figure(figsize=(9, 5))

plt.bar(names, spike_counts)

plt.axhline(
    anomaly_threshold,
    linestyle="--",
    label="Anomaly threshold"
)

plt.title("SNN Output Spike Comparison")
plt.xlabel("Electrical Condition")
plt.ylabel("Output Spike Count")
plt.legend()
plt.grid(axis="y")

plt.tight_layout()

plt.savefig(
    results_folder / "snn_spike_comparison.png"
)

plt.show()

# ==========================================
# 8. SAVE CLASSIFICATION RESULTS
# ==========================================

result_df = pd.DataFrame(
    classification_results
)

result_df.to_csv(
    results_folder / "classification_results.csv",
    index=False
)

print("\n================================")
print(" SNN ANOMALY DETECTION COMPLETED")
print("================================")

print(result_df.to_string(index=False))

print("\nResults saved successfully!")
