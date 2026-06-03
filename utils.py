import numpy as np
import scipy
from scipy.sparse.linalg import spsolve

def wrap(y: np.ndarray) -> np.ndarray: 
	return ((y + np.pi) % (2*np.pi)) - np.pi

def unwrap(Psi: np.ndarray, psi0_row_ind: int, psi0_col_ind: int) -> np.ndarray:
	
    Ny, Nx = Psi.shape

    Nm = Nx*Ny

    Ix = scipy.sparse.identity(Nx, dtype=np.float32)
    Iy = scipy.sparse.identity(Ny, dtype=np.float32)

    ones_x = np.ones((Nx), dtype=np.float32)
    ones_y = np.ones((Ny), dtype=np.float32)

    diags_x = np.vstack((-ones_x, ones_x))
    Dx = scipy.sparse.spdiags(diags_x, [0, 1], Nx-1, Nx, format="csr")
    Gx = scipy.sparse.kron(Iy, Dx)

    diags_y = np.vstack((-ones_y, ones_y))
    Dy = scipy.sparse.spdiags(diags_y, [0, 1], Ny-1, Ny, format="csr")
    Gy = scipy.sparse.kron(Dy, Ix)

    # set dirichilet condition
    last_row_bol_ind = np.ravel_multi_index((psi0_row_ind, psi0_col_ind), (Nx, Ny))
    last_row = scipy.sparse.csr_matrix(([1.0], ([0], [last_row_bol_ind])), shape=(1, Nm), dtype=np.float32)

    G = scipy.sparse.bmat(
        [
            [Gx],
            [Gy],
            [last_row]
        ], 
        format="csr")    

    # make data vector
    d = wrap(G @ Psi.flatten())

    # solve least squares unwrapping problem 
    LHS = G.T@G
    RHS = G.T@d
    Phi_star = spsolve(LHS, RHS)

    return Phi_star.reshape((Ny,Nx))

def make_diff_opperator(m, Nx, Nz, h, omega):

	Nm = Nx*Nz

	# central node  # z-1 node  # z+1 node  # x-1 node  # x+1 node
	diags = [0, Nx, -Nx, 1, -1]

	J, I = np.meshgrid(np.arange(Nx), np.arange(Nz))
	I = I.flatten()
	J = J.flatten()
	c2_flat = m.flatten()**2

	bol_top = I == 0
	bol_btm = I == Nz-1
	bol_lft = J == 0
	bol_rht = J == Nx-1
	bol_int = (bol_top + bol_btm + bol_lft + bol_rht) == 0

	# initialize diagonals
	D = np.zeros((5, Nm), dtype=complex)

	# set central stencils
	D[0,:][bol_int] = -4 * c2_flat[bol_int] / (h**2) + 1.0*(omega**2)       
	D[1,:][bol_int] = 1.0 * c2_flat[bol_int] / (h**2)                       
	D[2,:][bol_int] = 1.0 * c2_flat[bol_int] / (h**2)                      
	D[3,:][bol_int] = 1.0 * c2_flat[bol_int] / (h**2)                      
	D[4,:][bol_int] = 1.0 * c2_flat[bol_int] / (h**2)                      

	# set right boundary condition
	D[0,:][bol_rht] += -1.0/h - 1j*omega / np.sqrt(c2_flat[bol_rht])
	D[4,:][bol_rht] += +1.0/h
	D[4,:] = np.roll(D[4,:],-1)

	# set bottom boundary condition
	D[0,:][bol_btm] += -1.0/h - 1j*omega / np.sqrt(c2_flat[bol_btm])
	D[2,:][bol_btm] += +1.0/h
	D[2,:] = np.roll(D[2,:], -(Nx))

	# set left boundary condition
	D[0,:][bol_lft] += -1.0/h - 1j*omega / np.sqrt(c2_flat[bol_lft])
	D[3,:][bol_lft] += +1.0/h
	D[3,:] = np.roll(D[3,:],1)

	# set top boundary condition
	D[0,:][bol_top] += -1.0/h - 1j*omega / np.sqrt(c2_flat[bol_top])
	D[1,:][bol_top] += +1.0/h
	D[1,:] = np.roll(D[1,:], Nx)

	return scipy.sparse.spdiags(D, diags, Nm, Nm, format="csr")
    

