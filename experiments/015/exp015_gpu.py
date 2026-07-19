"""GPU lane: d15 random-tt definitive + two d8 noise-curve points."""
import sys

sys.path.insert(0, r"D:\sphertt-0.1.0\prototype\015")
from exp015_common import run_config                      # noqa: E402

run_config(15, "random-tt", 0.99, 8192, "cupy", "float32", "gpu")
run_config(8, "random-tt", 0.92, 2048, "cupy", "float32", "gpu")
run_config(8, "random-tt", 0.98, 2048, "cupy", "float32", "gpu")
print("gpu lane done.")
