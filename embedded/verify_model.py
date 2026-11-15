
import numpy as np

data = np.load("results/temporal_snn_v2_model.npz")

TIME_BINS = int(data["time_bins"])
HIDDEN_NEURONS = int(data["hidden_neurons"])

voltageBins = np.array([
    0.2, 0.3, 0.2, 0.4, 0.3,
    0.2, 0.3, 0.4, 0.2, 0.3,
    0.2, 0.3, 0.2, 0.4, 0.3,
    0.2, 0.3, 0.2, 0.3, 0.4
])

currentBins = np.array([
    0.3, 0.2, 0.3, 0.2, 0.4,
    0.3, 0.2, 0.3, 0.2, 0.3,
    0.4, 0.2, 0.3, 0.2, 0.3,
    0.2, 0.4, 0.3, 0.2, 0.3
])

W_INPUT = data["W_input"]
W_OUTPUT = data["W_output"]
X_MEAN = data["X_mean"]
X_STD = data["X_std"]

DECAY = float(data["decay"])
INPUT_SCALE = float(data["input_scale"])
THRESHOLD = float(data["threshold"])
INTERCEPT = float(data["intercept"])

features = np.zeros(680)
membrane = np.zeros(HIDDEN_NEURONS)

for t in range(TIME_BINS):
    voltage = voltageBins[t] / 5.0 * INPUT_SCALE
    current = currentBins[t] / 5.0 * INPUT_SCALE

    for n in range(HIDDEN_NEURONS):
        input_current = (
            voltage * W_INPUT[0, n] +
            current * W_INPUT[1, n]
        )

        membrane[n] = DECAY * membrane[n] + input_current

        if membrane[n] >= 1.0:
            features[t * HIDDEN_NEURONS + n] = 1.0
            membrane[n] = 0.0

    features[640 + 2 * t] = voltage
    features[640 + 2 * t + 1] = current

normalized = (features - X_MEAN) / X_STD
score = INTERCEPT + np.dot(normalized, W_OUTPUT)

print("Python SNN Verification")
print("Anomaly score:", score)
print("Model threshold:", THRESHOLD)

if score >= THRESHOLD:
    print("Result: ABNORMAL SIGNAL")
else:
    print("Result: NORMAL SIGNAL")
