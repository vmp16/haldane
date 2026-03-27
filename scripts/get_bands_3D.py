import sys
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from tools.model import HaldaneSystem
from tools.tools import save_figure, save_simulation

# --------- CONFIGURATION -----------------------------------

# Tight-binding parameters
t1 = 1.0            # Used as energy scale
t2 = 0.1            # Defined with respect to t1
phi = np.pi / 4
M = 0.2             # Defined with respect to t1

# Number of points for discretisation
N_PTS = 50

FERMI_LEVEL = -0.2 / t1     # Fermi level [t1 units]

# DEFINING K SPACE
# # Brillouin Zone
# kx, ky = np.meshgrid(np.linspace(-2.5, 2.5, N_PTS), np.linspace(-2.5, 2.5, N_PTS))
# k = np.stack([kx, ky], axis=-1).reshape(-1, 2)

# Output options
FOLDER_DATA = "data/eigen"
FOLDER_FIGS = "figures"

# -----------------------------------------------------------

def main():
    print("--- Initializing Haldane System ---")
    print(f"t1 = {t1}, t2 = {t2}, phi = {phi:.3f}, M = {M}")

    # Build the simulation
    system = HaldaneSystem(t1, t2, phi, M)

    # Generate the grid over the Brillouin Zone
    kx, ky = system.generate_k_mesh(n_pts=N_PTS)

    # Diagonalize at all the points in k-path
    energies, eigenstates = system.solve_at_k(system.k_mesh)

    print("kx shape = ", kx.shape)
    print("ky shape = ", ky.shape)
    print("energies shape = ", energies.shape)
    print("eigenstates shape = ", eigenstates[:,:,0].shape)

    # Plot the band structure
    fig = plt.figure()

    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(kx, ky, energies[..., 0], alpha=0.7, label='Band 1')
    ax.plot_surface(kx, ky, energies[..., 1], alpha=0.7, label='Band 2')

    # Add a horizontal plane at the Fermi level
    ax.plot_surface(
        kx, ky, np.full_like(kx, FERMI_LEVEL),
        alpha=0.3, color='gray', linewidth=0, antialiased=False
    )

    ax.set_xlabel('kx')
    ax.set_ylabel('ky')
    ax.set_zlabel('Energy')
    ax.set_title(f'Band Structure\nt1 = {t1}, t2 = {t2}, phi = {phi:.3f}, M = {M}')

    plt.show()

if __name__ == "__main__":
    main()