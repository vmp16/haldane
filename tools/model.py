import numpy as np
from numpy.linalg import eigh

class HaldaneSystem:
    def __init__(self, t1, t2, phi, M):
        """
        Initialize the Haldane system

        Parameters:
        -----------
        t1 : float
            1st neihgbor hopping
        t2 : float
            2nd neighbor hooping
        phi : float
            Magnetic flux associated to t2
        M : float
            on-site (mass) energy
        """

        self.t1 = t1
        self.t2 = t2
        self.phi = phi
        self.M = M

        # Storage for results
        self.hamiltonian = None
        self.energies = None
        self.eigenstates = None

        # ----- LATTICE GEOMETRY -----
        # Assuming a distance between 1st neighbors is a=1
        # Real space lattice vectors
        e1 = np.array((3, np.sqrt(3))) / 2
        e2 = np.array((3, -np.sqrt(3))) / 2
        self.e = [e1, e2]

        # Connecting 1st neighbors
        a1 = np.array((1, 0))
        a2 = np.array((-1, np.sqrt(3))) / 2
        a3 = np.array((-1, -np.sqrt(3))) / 2
        self.a = [a1, a2, a3]

        # Connecting 2nd neighbors
        b1 = a2 - a3
        b2 = a3 - a1
        b3 = a1 - a2
        self.b = [b1, b2, b3]

        # Reciprocal lattice vectors
        f1 = (2*np.pi/3) * np.array((1, np.sqrt(3)))
        f2 = (2*np.pi/3) * np.array((1, -np.sqrt(3)))
        self.f = [f1, f2]
    
    def generate_k_mesh(self, n_pts=30):
        """
        Returns the grid over the polygon formed by the reciprocal lattice vectors.
        Equivalent to the Brillouin Zone.
        """
        vals = np.linspace(0, 1, n_pts, endpoint=False)
        u, v = np.meshgrid(vals, vals)
        self.du = u[0, 1] - u[0, 0]
        self.dv = v[1, 0] - v[0, 0]

        f1, f2 = self.f

        kx = u * f1[0] + v * f2[0]
        ky = u * f1[1] + v * f2[1]
        self.k_mesh = np.stack([kx, ky], axis=-1)

        # Calculate the area of the Brillouin Zone
        self.area_bz = np.linalg.norm(f1[0] * f2[1] - f1[1] * f2[0])
        self.dk_area = self.area_bz / (n_pts ** 2)

        # Compute the Jacobian of the k-mesh with cartesians coordinates
        Fmat = np.column_stack(self.f)
        self.Jmat = np.linalg.inv(Fmat.T)

        return kx, ky

    def solve_at_k(self, k):

        # Extract kx and ky as arrays
        kx = k[..., 0]
        ky = k[..., 1]

        # Calculate h coefficients
        # 1st neighbors
        f_k = sum(np.exp(1j * (kx * a[0] + ky * a[1])) for a in self.a)
        h1 = self.t1 * np.real(f_k)
        h2 = self.t1 * np.imag(f_k)

        # 2nd neighbors
        g_k = sum(np.exp(1j * (kx * b[0] + ky * b[1])) for b in self.b)
        h0 = 2 * self.t2 * np.cos(self.phi) * np.real(g_k)
        h3 = self.M - 2 * self.t2 * np.sin(self.phi) * np.imag(g_k)

        h_coeffs = [h0, h1, h2, h3]

        # Defining Pauli's matrices
        sigmas = [np.eye(2),
                  np.array(((0, 1), (1, 0))),
                  np.array(((0, -1j), (1j, 0))),
                  np.array(((1, 0), (0, -1)))]
        
        # Build the Hamiltonian as a tensor: a 2x2 matrix at every point in a k-grid
        H = sum(h[..., np.newaxis, np.newaxis] * s for h, s in zip(h_coeffs, sigmas))
        self.hamiltonian = H

        # Vectorized diagonalization
        energies, eigenstates = eigh(H)

        self.energies = energies
        self.eigenstates = eigenstates

        return energies, eigenstates
    
    def derivate_hamiltonian(self, k, axis=0):
        """
        Calculate the derivative of the hamiltonian with respect to the direction given by axis (0 for x, 1 for y).
        """
        # Extract kx and ky as arrays
        kx = k[..., 0]
        ky = k[..., 1]

        # Check that axis lies within the dimensions of the 2D system
        if axis > 1 :
            print("Error: axis must be 0 or 1, for x or y direction.")
            return

        # Calculate dh/dk_i
        # 1st neighbors
        f_k = sum(a[axis] * np.exp(1j * (kx * a[0] + ky * a[1])) for a in self.a)
        dh1 = - self.t1 * np.imag(f_k)
        dh2 = self.t1 * np.real(f_k)
        
        # 2nd neighbors
        g_k = sum(b[axis] * np.exp(1j * (kx * b[0] + ky * b[1])) for b in self.b)
        dh0 = - 2 * self.t2 * np.cos(self.phi) * np.imag(g_k)
        dh3 = - 2 * self.t2 * np.sin(self.phi) * np.real(g_k)

        dh_coeffs = [dh0, dh1, dh2, dh3]

        # Defining Pauli's matrices
        sigmas = [np.eye(2),
                  np.array(((0, 1), (1, 0))),
                  np.array(((0, -1j), (1j, 0))),
                  np.array(((1, 0), (0, -1)))]
        
        # Build the derivated Hamiltonian as a tensor: a 2x2 matrix at every point in a k-grid
        dH = sum(h[..., np.newaxis, np.newaxis] * s for h, s in zip(dh_coeffs, sigmas))

        return dH
    
    def get_metadata(self):
        """Returns metadata for saving."""
        return {
            "t1": self.t1,
            "t2": self.t2,
            "phi": self.phi,
            "M": self.M,
        }