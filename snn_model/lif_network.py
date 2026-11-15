
from brian2 import *
import matplotlib.pyplot as plt
from pathlib import Path

# ==========================================
# 1. INITIALISE SIMULATION
# ==========================================

start_scope()

# Use NumPy instead of Cython
prefs.codegen.target = 'numpy'

defaultclock.dt = 0.1 * ms

# ==========================================
# 2. CREATE INPUT SPIKE GENERATORS
# ==========================================

input_neurons = PoissonGroup(
    2,
    rates=[20, 80] * Hz
)

# ==========================================
# 3. DEFINE LIF NEURON
# ==========================================

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

# ==========================================
# 4. CONNECT INPUT AND LIF NEURON
# ==========================================

connections = Synapses(
    input_neurons,
    lif_neuron,
    on_pre='v_post += 0.6'
)

connections.connect()

# ==========================================
# 5. RECORD NEURON ACTIVITY
# ==========================================

spike_monitor = SpikeMonitor(lif_neuron)

voltage_monitor = StateMonitor(
    lif_neuron,
    'v',
    record=True
)

# ==========================================
# 6. RUN SIMULATION
# ==========================================

run(1 * second)

# ==========================================
# 7. DISPLAY RESULTS
# ==========================================

fig, axes = plt.subplots(2, 1, figsize=(10, 7))

# Membrane potential graph

axes[0].plot(
    voltage_monitor.t / ms,
    voltage_monitor.v[0]
)

axes[0].axhline(
    1,
    linestyle='--',
    label='Firing threshold'
)

axes[0].set_title('LIF Neuron Membrane Potential')
axes[0].set_xlabel('Time (ms)')
axes[0].set_ylabel('Membrane Potential')
axes[0].legend()
axes[0].grid(True)

# Output spike graph

axes[1].scatter(
    spike_monitor.t / ms,
    spike_monitor.i,
    marker='|',
    s=150
)

axes[1].set_title('LIF Neuron Output Spikes')
axes[1].set_xlabel('Time (ms)')
axes[1].set_ylabel('Neuron')
axes[1].grid(True)

plt.tight_layout()

# ==========================================
# 8. SAVE RESULTS
# ==========================================

project_folder = Path(__file__).resolve().parents[1]

results_folder = project_folder / 'results'
results_folder.mkdir(exist_ok=True)

plt.savefig(
    results_folder / 'lif_neuron_results.png'
)

plt.show()

# ==========================================
# 9. DISPLAY SIMULATION INFORMATION
# ==========================================

print("\n================================")
print(" LIF NEURON SIMULATION RESULTS")
print("================================")

print(
    "Total output spikes:",
    spike_monitor.num_spikes
)

print(
    "Average firing rate:",
    spike_monitor.num_spikes / second
)

print("\nSimulation completed successfully!")
