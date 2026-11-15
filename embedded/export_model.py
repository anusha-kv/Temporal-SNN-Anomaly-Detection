
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

model_path = ROOT / "results" / "temporal_snn_v2_model.npz"
output_path = ROOT / "embedded" / "snn_model.h"

data = np.load(model_path)


def cpp_float(value):
    s = f"{float(value):.9g}"

    if "." not in s and "e" not in s and "E" not in s:
        s += ".0"

    return s + "f"


def format_array(name, values):
    values = np.asarray(values).flatten()

    formatted = [cpp_float(v) for v in values]

    return (
        f"const float {name}[{len(formatted)}] = {{\n"
        + ",\n".join(formatted)
        + "\n};\n"
    )


with open(output_path, "w") as f:

    f.write("#pragma once\n\n")

    f.write("const int HIDDEN_NEURONS = 32;\n")
    f.write("const int TIME_BINS = 20;\n")
    f.write("const int FEATURE_COUNT = 680;\n\n")

    f.write(format_array("W_INPUT", data["W_input"]))
    f.write(format_array("W_OUTPUT", data["W_output"]))
    f.write(format_array("X_MEAN", data["X_mean"]))
    f.write(format_array("X_STD", data["X_std"]))

    for name in [
        "intercept",
        "threshold",
        "decay",
        "input_scale",
        "duration",
        "fs"
    ]:

        value = cpp_float(data[name])

        f.write(
            f"const float {name.upper()} = {value};\n"
        )


print("Model conversion successful!")
print("Saved to:", output_path)
