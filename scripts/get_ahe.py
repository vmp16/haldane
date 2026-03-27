import sys
import numpy as np
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from tools.model import HaldaneSystem
from tools.analysis import calculate_ahe

# -------------------- CONFIGURATION ------------------------

# Tight-binding parameters
t1 = 1.0            # Used as energy scale
t2 = 0.1            # Defined with respect to t1
phi = np.pi / 4
M = 0.2             # Defined with respect to t1

# General constants
kB = 8.617e-5       # Boltzmann constant in eV/K
h = 4.1357          # Planck constant [eV·fs]

# Conduction parameters
tau_real = 100           # relaxation time [fs=1e-15s]
tau_eff = (tau_real / h) * t1

kB = 8.617e-5       # Boltzmann constant in eV/K
T_real = 20         # temperature [K]
T_eff = (kB * T_real) / t1

mu_eff = -0.2 / t1       # Fermi level [t1 units]

# Number of points for discretization
N_PTS = 60

# Output options
FOLDER_DATA = "data"
FOLDER_FIGS = "figures"

# -----------------------------------------------------------

def main():
    print("--- Building The System ---")
    # Build the simulation
    system = HaldaneSystem(t1, t2, phi, M)

    # Calculate conductivity
    print(f"Calculating the AHE for mu = {mu_eff:.2f} ...")
    sigma_xy, sigma_xy_per_band = calculate_ahe(system, T_eff, mu_eff, N_PTS=N_PTS)
    
    print("End of calculation")

    print(f"sigma_xy per band: {sigma_xy_per_band}")
    print(f"sigma_xy = {sigma_xy:.4f} e^2/h")

if __name__ == "__main__":
    main()