import sys
import numpy as np
from pathlib import Path

# Add project root to path
project_root = Path("/home/martinpv/DIPC/haldane")
sys.path.append(str(project_root))

from tools.model import HaldaneSystem

t1 = 1.0
t2 = 0.1
phi = np.pi / 4
M = 0.2

system = HaldaneSystem(t1, t2, phi, M)
kx, ky = system.generate_k_mesh(n_pts=100)
energies, _ = system.solve_at_k(system.k_mesh)

E_v_min = np.min(energies[..., 0])
E_v_max = np.max(energies[..., 0])
E_c_min = np.min(energies[..., 1])
E_c_max = np.max(energies[..., 1])

print(f"Valence band: [{E_v_min:.4f}, {E_v_max:.4f}]")
print(f"Conduction band: [{E_c_min:.4f}, {E_c_max:.4f}]")
print(f"Gap: {E_c_min - E_v_max:.4f}")
