import sys
import numpy as np
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from tools.model import HaldaneSystem
from tools.analysis import calculate_berry_curv

# -------------------- CONFIGURATION ------------------------

# Tight-binding parameters
t1 = 1.0            # Used as energy scale
t2 = 0.1            # Defined with respect to t1
phi = np.pi / 4
M = 0.2             # Defined with respect to t1

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

    kx, ky = system.generate_k_mesh(n_pts=N_PTS)
    energies, eigenstates = system.solve_at_k(system.k_mesh)

    print("Jmat shape =", system.Jmat.shape)
    print("Eigenvector shape =", eigenstates[:, :, 0].T.shape)

    # print("Im[eigenvector] = ", np.imag(np.min(eigenstates[:, :, 0])))

    Omega_list = calculate_berry_curv(system, N_PTS)

    print("Curvatures shape =", Omega_list.shape)

    print("Max curvature: ", np.max(Omega_list))
    print("Min curvature: ", np.min(Omega_list))

    # PLOT THE CURVATURE IN THE BRILLOUIN ZONE ?

if __name__ == "__main__":
    main()