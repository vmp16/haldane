import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from tools.model import HaldaneSystem
from tools.analysis import calculate_berry_curv
from tools.tools import save_figure

# -------------------- CONFIGURATION ------------------------

# Tight-binding parameters
t1 = 1.0            # Used as energy scale
t2 = 0.1            # Defined with respect to t1
phi = np.pi / 4
M = 0.2             # Defined with respect to t1

# Number of points for discretization
N_PTS = 100

# Output options
FOLDER_FIGS = "figures"
SAVE_PLOT = False

# -----------------------------------------------------------

def main():
    print("--- Building The System ---")
    system = HaldaneSystem(t1, t2, phi, M)

    # Generate the grid over the Brillouin Zone
    kx, ky = system.generate_k_mesh(n_pts=N_PTS)
    
    print("Calculating Berry Curvature...")
    # Omega shape: (N_PTS, N_PTS, n_bands)
    Omega = calculate_berry_curv(system, N_PTS)

    # Plotting
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), subplot_kw={'projection': '3d'})
    
    titles = ['Valence Band (n=0)', 'Conduction Band (n=1)']
    cmaps = ['viridis', 'magma']

    for n in range(2):
        ax = axes[n]
        surf = ax.plot_surface(kx, ky, Omega[..., n], cmap=cmaps[n], 
                               linewidth=0, antialiased=True)
        
        ax.set_title(f'Berry Curvature - {titles[n]}', fontsize=14)
        ax.set_xlabel('$k_x$', fontsize=12)
        ax.set_ylabel('$k_y$', fontsize=12)
        ax.set_zlabel(r'$\Omega_z(k)$', fontsize=12)
        fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10, pad=0.1)

    plt.tight_layout()

    if SAVE_PLOT:
        fig_name = f"berry_curvature_t2{t2}_phi{phi:.3f}_M{M}.png"
        save_figure(fig, fig_name, FOLDER_FIGS)

    plt.show()

if __name__ == "__main__":
    main()
