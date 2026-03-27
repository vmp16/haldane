import numpy as np
from pathlib import Path
from datetime import datetime
from .model import HaldaneSystem

def save_simulation(system, k_path, k_mod, folder="data", prefix="band_structure"):
    """
    Saves band structure data including eigenvectors to a compressed npz file.
    """
    # 1. Find the Project Root (Assuming tools.py is in haldane/tools/)
    # .parent gives 'tools', .parent.parent gives 'haldane'
    project_root = Path(__file__).resolve().parent.parent
    target_dir = project_root / folder
    
    target_dir.mkdir(parents=True, exist_ok=True)
    
    params = system.get_metadata()
    
    filename = f"{prefix}_t2{system.t2}_phi{system.phi:.3f}_M{system.M}.npz"
    
    filepath = target_dir / filename
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Prepare data
    data_to_save = {
        'k_path': k_path,
        'k_mod': k_mod,
        'energies': system.energies,
        'eigenstates': system.eigenstates,
        'timestamp': timestamp
    }
    data_to_save.update(params)  # scalar parameters

    print(f"Saving band structure data to: {filepath} ...")
    np.savez_compressed(filepath, **data_to_save)
    print("Save successful.")
    return str(filepath)

def load_simulation(filepath):
    """Loads a simulation file."""
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"File {filepath} not found.")
        
    with np.load(filepath, allow_pickle=True) as data:
        results = {key: data[key] for key in data.files}
        for key in results:
            if results[key].ndim == 0:
                results[key] = results[key].item()
    return results

def load_system(filepath):
    """
    Loads a .npz file and reconstructs the HofstadterSystem object
    with the energies and eigenstates populated.
    """
    data = load_simulation(filepath)
    
    # Reconstruct the system object using the saved parameters
    # We use .get() with defaults for backward compatibility
    sim = HaldaneSystem(
        t1=data['t1'],
        t2=data['t2'],
        phi=data['phi'],
        M=data['M']
    )
    
    # Inject the computed results
    sim.energies = data['energies']
    sim.eigenstates = data['eigenstates']
        
    return sim

def save_figure(fig, filename, folder="figures"):
    """
    Saves a matplotlib figure relative to the Project Root.
    """
    # Find Project Root
    project_root = Path(__file__).resolve().parent.parent
    target_dir = project_root / folder
    
    # Create folder if needed
    target_dir.mkdir(parents=True, exist_ok=True)
    
    filepath = target_dir / filename
    
    print(f"Saving plot to: {filepath} ...")
    fig.savefig(filepath, bbox_inches='tight') #  format='svg'
    print("Plot saved.")
    return str(filepath)