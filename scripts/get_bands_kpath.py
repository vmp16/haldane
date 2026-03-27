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

# DEFINING K-PATH
# High-symetry points
Gamma = np.array((0.00000, 0.00000))
K1 = (2*np.pi / 3) * np.array((1, 1/np.sqrt(3)))
K2 = (2*np.pi / 3) * np.array((1, -1/np.sqrt(3)))
M2 = (2*np.pi / 3) * np.array((1, 0))

N_PTS = 60          # Number of points per path segment

k_path = np.concatenate([
    np.linspace(Gamma, K2, N_PTS)[:-1],           # Gamma - K2
    np.linspace(K2, K1, N_PTS)[:-1],              # K2- K1
    np.linspace(K1, Gamma, N_PTS)                 # K1 - Gamma
])

# k_path = np.array([[k, 0] for k in np.linspace(0, 2*np.pi, 50)])
dk = np.diff(k_path, axis=0)
distances = np.linalg.norm(dk, axis=1)
k_mod = np.cumsum(np.concatenate([[0], distances]))

# Output options
FOLDER_DATA = "data/eigen"
FOLDER_FIGS = "figures"

# -----------------------------------------------------------

def main():
    print("--- Initializing Haldane System ---")
    print(f"t1 = {t1}, t2 = {t2}, phi = {phi:.3f}, M = {M}")

    # Build the simulation
    system = HaldaneSystem(t1, t2, phi, M)

    # Diagonalize at all the points in k-path
    energies, eigenstates = system.solve_at_k(k_path)

    # Save the data
    save_simulation(system, k_path, k_mod, folder=FOLDER_DATA)

    # print("Saving the data ...")
    print(f"Energies shape: {energies.shape}")
    print(f"Eigenstates shape: {eigenstates[:, :, 0].shape}")

    # Plot the band structure
    fig = plt.figure(figsize=(6,5))
    plt.plot(k_mod, energies, color='navy')

    # Add symmetry point labels
    sym_points_plot = [
        (k_mod[0], r"$\Gamma$"),
        (k_mod[N_PTS - 1], "K'"),
        (k_mod[3*N_PTS//2 - 2], "M"),
        (k_mod[2*N_PTS - 2], "K"),
        (k_mod[-1], r"$\Gamma$")
    ]

    plt.yticks(fontsize=14)
    plt.xticks([pos for pos, label in sym_points_plot], [label for pos, label in sym_points_plot], fontsize=15)
    for pos, label in sym_points_plot:
        plt.axvline(x=pos, color='gray', linestyle=':', alpha=0.7)

    plt.xlabel("k-path", fontsize=15)
    plt.ylabel("Energy", fontsize=15)
    plt.title(f"t1 = {t1}, t2 = {t2}, phi = {phi:.3f}, M = {M}", fontsize=15)

    plt.tight_layout()

    # Save the figure
    fig_name = f"band_structure_t2{system.t2}_phi{system.phi:.3f}_M{system.M}.png"
    save_figure(fig, fig_name, FOLDER_FIGS)

    plt.show()
    
if __name__ == "__main__":
    main()