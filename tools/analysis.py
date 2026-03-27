import numpy as np
from .model import HaldaneSystem

def fermi_distrib(E, mu_eff, T_eff):
    """
    Returns the Fermi distribution value.

    Parameters
    ----------
    E : float
        Energy in t1 units
    mu_eff : float
        Fermi level in t1 units
    T_eff : float
        Scalated temperature (kB * T_real / t1)

    Returns
    ----------
    f(E, T, mu) : float
        Corresponding value of the Fermi distribution
    """

    x = (E - mu_eff) / T_eff

    # Avoid overflow in exp by clipping x
    x_clipped = np.clip(x, -700, 700)
    return 1 / (1 + np.exp(x_clipped))

def grad_hexa(f, system):
    """
    Calculates the gradient of f in hexagonal coordinates.
    """
    dy_du = (np.roll(f, -1, axis=0) - np.roll(f, 1, axis=0)) / (2 * system.du)
    dy_dv = (np.roll(f, -1, axis=1) - np.roll(f, 1, axis=1)) / (2 * system.dv)

    return np.stack([dy_du, dy_dv], axis=0)

def grad_cart(f, system):
    """
    Calculates the gradient of f in cartesian coordinates.
    """
    deriv_hexa = grad_hexa(f, system)

    # Use tensordot to contract the 2nd axis of the 2x2 Jacobian matrix
    # with the 1st axis of the gradient tensor.
    # This safely handles f having any arbitrary number of dimensions.
    return np.tensordot(system.Jmat, deriv_hexa, axes=([1], [0]))


def drude_conductivity(tau_eff, mu_eff):
    """
    Calculates the conductivity following Drude's formula (per spin).
    
    Parameters
    ----------
    tau_eff : float
        Scaled relaxation time (tau_real / h * t1)
    mu_eff : float
        Scaled chemical potential (mu / t1) from the band extremum

    Returns
    ----------
    sigma : float
        Drude's conductivity in e^2/h units
    """

    sigma = 2 * np.pi * tau_eff * mu_eff

    print(f"Drude's conductivity = {sigma:.4f} e^2/h")

    return sigma


def calculate_conductivity(system, T_eff, mu_eff, tau_eff, N_PTS=60):
    """
    Calculate the conductivity tensor in the relaxation time approximation
    using the semiclassical description.
    
    Parameters
    ----------
    system : HaldaneSystem
        The system with tight-binding parameters
    T_eff : float
        Scaled temperature (kB * T_real / t1)
    mu_eff : float
        Scaled chemical potential (mu / t1)
    tau_eff : float
        Scaled relaxation time (tau_real / h * t1)
    N_PTS : int
        Number of k-points per direction in the Brillouin zone mesh
        
    Returns
    -------
    sigma : ndarray
        Conductivity tensor (2x2) in units of e^2/h
    """
    # Generate the grid over the Brillouin Zone if not already done
    if not hasattr(system, 'k_mesh'):
        kx, ky = system.generate_k_mesh(n_pts=N_PTS)
    
    area_bz = system.area_bz

    # Calculate the energy bands in the k-grid if not already done
    if system.energies is None:
        energies, eigenstates = system.solve_at_k(system.k_mesh)
    else:
        energies = system.energies

    # Calculate Drude's conductivity (scalar approximation)
    E_v = np.max(energies[..., 0])
    E_c = np.min(energies[..., 1])
    
    if mu_eff < E_v:
        mu_kf = E_v - mu_eff
    elif mu_eff > E_c:
        mu_kf = mu_eff - E_c
    else:
        mu_kf = 0.0
    
    drude = drude_conductivity(tau_eff, mu_kf)

    # Define factor before the sum
    prefactor = (tau_eff * area_bz) / (N_PTS * N_PTS)

    # Initialize for storing
    sigma_xx_list = []
    sigma_xy_list = []
    sigma_yx_list = []
    sigma_yy_list = []

    # Iterate over each band
    for n in range(energies.shape[-1]):
        band_E = energies[..., n]

        # Compute second derivatives in hexagonal coordinates
        d2E_du2 = (np.roll(band_E, -1, axis=0) - 2 * band_E + np.roll(band_E, 1, axis=0)) / (system.du**2)
        d2E_dv2 = (np.roll(band_E, -1, axis=1) - 2 * band_E + np.roll(band_E, 1, axis=1)) / (system.dv**2)
        d2E_dudv = (
            np.roll(np.roll(band_E, -1, axis=0), -1, axis=1)
            - np.roll(np.roll(band_E, -1, axis=0), 1, axis=1)
            - np.roll(np.roll(band_E, 1, axis=0), -1, axis=1)
            + np.roll(np.roll(band_E, 1, axis=0), 1, axis=1)
        ) / (4 * system.du * system.dv)

        # Build Hessian matrix in hexagonal coordinates
        H_uv = np.array([
            [d2E_du2, d2E_dudv],
            [d2E_dudv, d2E_dv2]
        ])

        # Transpose H_uv to (Nkx, Nky, 2, 2)
        H_uv_trans = H_uv.transpose(2, 3, 0, 1)

        # Calculate the Hessian matrix in cartesian coordinates
        H_xy = system.Jmat @ H_uv_trans @ system.Jmat.T

        # Extract components
        d2E_dkx2 = H_xy[..., 0, 0]
        d2E_dkx_dky = H_xy[..., 0, 1]
        d2E_dky_dkx = H_xy[..., 1, 0]
        d2E_dky2 = H_xy[..., 1, 1]

        # Calculate the integrals for each component
        f = fermi_distrib(band_E, mu_eff, T_eff)
        integral_xx = np.sum(f * d2E_dkx2)
        integral_xy = np.sum(f * d2E_dkx_dky)
        integral_yx = np.sum(f * d2E_dky_dkx)
        integral_yy = np.sum(f * d2E_dky2)

        sigma_xx_list.append(integral_xx * prefactor)
        sigma_xy_list.append(integral_xy * prefactor)
        sigma_yx_list.append(integral_yx * prefactor)
        sigma_yy_list.append(integral_yy * prefactor)

    sigma_xx = np.sum(sigma_xx_list)
    sigma_xy = np.sum(sigma_xy_list)
    sigma_yx = np.sum(sigma_yx_list)
    sigma_yy = np.sum(sigma_yy_list)

    sigma_tensor = np.array([[sigma_xx, sigma_xy], [sigma_yx, sigma_yy]])

    # Create per-band tensors
    sigma_per_band = []
    for i in range(len(sigma_xx_list)):
        band_tensor = np.array([[sigma_xx_list[i], sigma_xy_list[i]], 
                               [sigma_yx_list[i], sigma_yy_list[i]]])
        sigma_per_band.append(band_tensor)

    return sigma_tensor, sigma_per_band

# ----- DOES NOT WORK, DERIVATES NUMERICALLY THE WAVE FUNCTIONS -----
# def calculate_berry_curv(system, N_PTS=50):
    """
    Calculate the Berry's curvature in cartesian coordinates.

    Parameters
    ----------
    system : HaldaneSystem
        The system with tight-binding parameters
    N_PTS : int
        Number of points for the k-mesh

    Returns
    -------
    A_list : list
        Berry's connexion for every eigenstate
    Omega_list : list
        Berry's curvature in the perpendicular direction for every eigenstate
    """

    # Generate the grid over the Brillouin Zone if not already done
    if not hasattr(system, 'k_mesh'):
        kx, ky = system.generate_k_mesh(n_pts=N_PTS)

    # Calculate the energy bands in the k-grid if not already done
    if system.energies is None:
        energies, eigenstates = system.solve_at_k(system.k_mesh)
    else:
        energies = system.energies
        eigenstates = system.eigenstates

    A_list = []         # stocking connexions per eigenstate
    Omega_list = []     # stocking curvatures per eigenstate

    # Number of bands
    n_bands = eigenstates.shape[-1]

    # Iterate over every eigenstate
    for n in range(n_bands):
        u_n = eigenstates[..., :, n]

        # Vectorized gradient calculation for the spinor
        # eigenvect shape: (Nkx, Nky, 2)
        # dvect_cart shape: (2, Nkx, Nky, 2) representing (dx/dy, kx, ky, spinor_component)
        dvect_cart = grad_cart(u_n, system)

        # Calculate Berry connections using einsum to sum over the spinor components
        # 'xys' represents kx, ky, spinor_comp
        # 'dxys' represents dx/dy, kx, ky, spinor_comp
        # Result 'xyd' represents kx, ky, dx/dy -> shape: (Nkx, Nky, 2)
        Avect = 1j * np.einsum('xys,dxys->xyd', u_n.conj(), dvect_cart)
        A_list.append(Avect)

        # Vectorized calculation of Berry curvature
        # Avect shape: (Nkx, Nky, 2). grad_Avect shape: (2, Nkx, Nky, 2) -> (dx/dy, kx, ky, Ax/Ay)
        grad_Avect = grad_cart(Avect, system)
        
        # Berry curvature: Omega_z = dAy/dkx - dAx/dky
        # grad_Avect[0, ..., 1] is the x-derivative of Ay
        # grad_Avect[1, ..., 0] is the y-derivative of Ax
        Omega_z = grad_Avect[0, ..., 1] - grad_Avect[1, ..., 0]

        Omega_list.append(Omega_z)

    # Stack along the last axis to match the band dimension in energies and eigenstates
    return np.stack(A_list, axis=-1), np.stack(Omega_list, axis=-1)

def calculate_berry_curv(system, N_PTS=50):
    """
    Calculate Berry's curvature using Kubo-like expression to ensure gauge invariance.

    Parameters
    ----------
    system : HaldaneSystem
        The system with tight-binding parameters
    N_PTS : int
        Number of points per direction for the discretization of the Brillouin zone.

    Returns
    -------
    Omega_list : array
        Value of the Berry curvature for each wave function at every point of the k-mesh.
    """

    # Generate the grid over the Brillouin Zone if not already done
    if not hasattr(system, 'k_mesh'):
        kx, ky = system.generate_k_mesh(n_pts=N_PTS)

    # Calculate the energy bands in the k-grid if not already done
    if system.energies is None:
        energies, eigenstates = system.solve_at_k(system.k_mesh)
    else:
        energies = system.energies
        eigenstates = system.eigenstates

    Omega_list = []     # stocking curvatures per eigenstate

    # Calculate analytical derivatives of the Hamiltonian
    dH_dkx = system.derivate_hamiltonian(system.k_mesh, axis=0)
    dH_dky = system.derivate_hamiltonian(system.k_mesh, axis=1)

    # Vectorized calculation over all bands
    # U shape: (..., spinor, band) => eigenvectors as columns
    U = eigenstates
    # U_dag shape: (..., band, spinor)
    U_dag = np.swapaxes(U.conj(), -1, -2) # => eigenvectors as rows + conjugate

    # Velocity matrices: V_i = U^\dagger * dH_dk_i * U
    # Resulting shapes: (..., band, band)
    Vx = U_dag @ dH_dkx @ U
    Vy = U_dag @ dH_dky @ U

    # Cross terms: (Vx)_{nm} * (Vy)_{nm}^* - (Vy)_{nm} * (Vx)_{nm}^*
    cross_term = Vx * Vy.conj() - Vy * Vx.conj()

    # Energy differences squared: (E_n - E_m)^2
    # Broadcast energies to compute pairwise differences
    # energies shape: (..., band) -> dE shape: (..., band, band)
    dE = energies[..., :, np.newaxis] - energies[..., np.newaxis, :]
    dE_sq = dE ** 2

    # Calculate curvature
    with np.errstate(divide='ignore', invalid='ignore'):
        term = 1j * cross_term / dE_sq
        # Avoid DivisionByZero on the diagonal (n == m) or degeneracies
        term[dE_sq == 0] = 0.0

    # Sum over the 'm' index (axis=-1) to get the curvature for each 'n' band
    Omega = np.real(np.sum(term, axis=-1))

    return Omega


def calculate_ahe(system, T_eff, mu_eff, N_PTS=60):
    """
    Calculate the Anomalous Hall Effect using the Berry curvature.
    
    system : HaldaneSystem
        The system with tight-binding parameters
    T_eff : float
        Scaled temperature (kB * T_real / t1)
    mu_eff : float
        Scaled chemical potential (mu / t1)
    N_PTS : int
        Number of k-points per direction in the Brillouin zone mesh

    Returns
    -------

    """
    curvatures = calculate_berry_curv(system, N_PTS=N_PTS)

    energies = system.energies

    prefactor = system.area_bz / (2 * np.pi * N_PTS * N_PTS)
    
    sigma_xy_list = []

    # Number of bands
    n_bands = energies.shape[-1]

    for n in range(n_bands):
        band_E = energies[..., n]
        band_curvature = curvatures[..., n]

        # Calculate the integral
        f = fermi_distrib(band_E, mu_eff, T_eff)
        integral = np.sum(f * band_curvature)

        sigma_xy_list.append(integral * prefactor)

    sigma_xy = np.sum(sigma_xy_list)

    return sigma_xy, np.array(sigma_xy_list)