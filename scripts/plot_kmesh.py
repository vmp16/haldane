import sys
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from tools.model import HaldaneSystem

# --------- CONFIGURATION -----------------------------------

# Tight-binding parameters
t1 = 1.0
t2 = 0.1
phi = np.pi / 4
M = 0.5

# Number of points for discretisation
N_PTS = 10

# -----------------------------------------------------------

def main():
    # Build the simulation
    system = HaldaneSystem(t1, t2, phi, M)

    # Generate the grid over the Brillouin Zone
    kx, ky = system.generate_k_mesh(n_pts=N_PTS)

    plt.figure(figsize=(6, 8))

    plt.scatter(kx, ky, color='blue', s=20)
    plt.scatter(kx[0, 0], ky[0, 0], color='red', s=20, zorder=5)
    plt.scatter(kx[0, 1], ky[0, 1], color='red', s=20, zorder=5)

    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)

    plt.title("Brillouin zone", fontsize=16)
    plt.xlabel("kx", fontsize=16)
    plt.ylabel("ky", fontsize=16)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.show()

if __name__ == "__main__":
    main()