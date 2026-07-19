"""CPU lane: d8 anchor (kron) + two noise-curve points, fp64."""
import sys

sys.path.insert(0, r"D:\sphertt-0.1.0\prototype\015")
from exp015_common import run_config                      # noqa: E402

run_config(8, "kron-sum", 0.99, 2048, "numpy", "float64", "cpu")
run_config(8, "random-tt", 0.95, 2048, "numpy", "float64", "cpu")
run_config(8, "random-tt", 0.90, 2048, "numpy", "float64", "cpu")
print("cpu lane done.")
