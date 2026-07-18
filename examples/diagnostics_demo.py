"""Demonstrate the truncation-safety diagnostics: stable band vs runaway."""
import numpy as np

from sphertt import SphereTTReservoir, amplification_report
from sphertt.tasks import narma

u, _ = narma(1500, rng=np.random.default_rng(0))

for beta in [0.8, 0.5]:
    res = SphereTTReservoir(n_dims=10, mode_size=2, beta=beta, chi_w=8,
                            seed=0)
    rep = amplification_report(res, u, chi_x=16)
    print(f"beta={beta}: g_bar={rep['g_bar']:.4f} "
          f"A_pred={rep['amp_predicted']:.1f} "
          f"A_meas={rep['amp_measured']:.1f} runaway={rep['runaway']}")
