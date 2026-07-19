"""CPU lane: light (low-rank) configs at float64, run concurrently with
the GPU lane."""
import sys

sys.path.insert(0, r"D:\sphertt-0.1.0\prototype\012")
from exp012_common import evaluate                        # noqa: E402

for w_kind, chi_w in [("kron-sum", 2), ("kron-sum", 4),
                      ("kron-orth-sum", 2), ("kron-orth-sum", 4)]:
    evaluate(w_kind, chi_w, "numpy", "float64", runner="cpu")
print("cpu lane done.")
