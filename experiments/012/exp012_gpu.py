"""GPU lane: heavy (high-rank) configs at float32."""
import sys

sys.path.insert(0, r"D:\sphertt-0.1.0\prototype\012")
from exp012_common import evaluate                        # noqa: E402

for w_kind, chi_w in [("random-tt", 8), ("kron-sum", 8),
                      ("kron-orth-sum", 8)]:
    evaluate(w_kind, chi_w, "cupy", "float32", runner="gpu")
print("gpu lane done.")
