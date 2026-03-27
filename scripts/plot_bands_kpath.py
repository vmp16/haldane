import sys
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from tools.tools import save_figure, load_simulation

# --------- CONFIGURATION -----------------------------------

filename = "band_structure_t20.1_phi0.785_M0.2.npz"
folder_data = "data/eigen"
input_dir = project_root / folder_data / filename

FERMI_LEVEL = -0.2

SAVE_PLOT = False
FOLDER_FIGS = "figures"

# -----------------------------------------------------------

def main():
    # Load the data
    data = load_simulation(input_dir)
    energies = data['energies']
    k_mod = data['k_mod']
    t1 = data['t1']
    t2 = data['t2']
    phi = data['phi']
    M = data['M']

    # print(f"k_mod shape = {k_mod.shape}")
    N_PTS = k_mod.shape[0] // 3 + 1

    # Print the energies around the gap
    print(f"Max of conduction band = {np.max(energies[..., 0])}")
    print(f"Min of valence band = {np.min(energies[..., 1])}")

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

    # Add the Fermi level if there is one
    if FERMI_LEVEL is not None:
        plt.axhline(y=FERMI_LEVEL, color='red', linestyle='--', label='Fermi level')

    plt.yticks(fontsize=14)
    plt.xticks([pos for pos, label in sym_points_plot], [label for pos, label in sym_points_plot], fontsize=15)
    for pos, label in sym_points_plot:
        plt.axvline(x=pos, color='gray', linestyle=':', alpha=0.7)
    plt.axhline(y=0, color='gray', linestyle=':', alpha=0.5)

    plt.xlabel("k-path", fontsize=15)
    plt.ylabel("Energy", fontsize=15)
    plt.title(f"t1 = {t1}, t2 = {t2}, phi = {phi:.3f}, M = {M}", fontsize=15)

    plt.tight_layout()

    if SAVE_PLOT:
        # Save the figure
        fig_name = f"band_structure_t2{t2}_phi{phi:.3f}_M{M}.svg"
        save_figure(fig, fig_name, FOLDER_FIGS)

    plt.show()

if __name__ == "__main__":
    main()