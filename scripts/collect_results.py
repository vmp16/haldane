import sys
import numpy as np
from pathlib import Path

# Add project root to path
project_root = Path("/home/martinpv/DIPC/haldane")
sys.path.append(str(project_root))

from tools.model import HaldaneSystem
from tools.analysis import calculate_conductivity, calculate_ahe

# Tight-binding parameters
t1 = 1.0
t2 = 0.1
phi = np.pi / 4
M = 0.2

# Conduction parameters
tau_real = 100
h = 4.1357 # eV*fs
tau_eff = (tau_real / h) * t1

kB = 8.617e-5
T_real = 20
T_eff = (kB * T_real) / t1

mu_eff_cond = -0.4 # Near the valence band edge (E_v_max ~ -0.3832)

def collect_data():
    n_pts_list = [20, 40, 60, 80, 100, 120, 150]
    
    print(f"Collecting longitudinal conductivity data (mu={mu_eff_cond})...")
    print(f"{'N_PTS':>6} | {'sigma_xx':>10} | {'sigma_yy':>10} | {'Drude':>10}")
    print("-" * 50)
    for n_pts in n_pts_list:
        system = HaldaneSystem(t1, t2, phi, M)
        sigma_tensor, _ = calculate_conductivity(system, T_eff, mu_eff_cond, tau_eff, n_pts)
        
        # Calculate Drude again just to be sure we have the latest printed value
        # In analysis.py, drude_conductivity is called inside calculate_conductivity
        # But it's printed to stdout. Let's just catch the values.
        print(f"{n_pts:6d} | {sigma_tensor[0,0]:10.4f} | {sigma_tensor[1,1]:10.4f}")

    print("\nChecking AHE quantization for different M (mu in the gap)...")
    for M_val in [0.0, 0.1, 0.2, 0.5, 0.7]:
        system = HaldaneSystem(t1, t2, phi, M_val)
        kx, ky = system.generate_k_mesh(n_pts=80)
        energies, _ = system.solve_at_k(system.k_mesh)
        E_v_max = np.max(energies[..., 0])
        E_c_min = np.min(energies[..., 1])
        mu_gap = (E_v_max + E_c_min) / 2
        s_xy, _ = calculate_ahe(system, T_eff, mu_gap, N_PTS=80)
        print(f"M={M_val:.1f}: sigma_xy={s_xy:.4f} e^2/h")

if __name__ == "__main__":
    collect_data()
