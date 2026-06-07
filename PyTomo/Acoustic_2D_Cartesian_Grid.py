import numpy as np
import numba
from numba import jit
import PT_data_classes


@jit(nopython=True, fastmath=True, cache=True)
def Time_Step(u: np.ndarray, 
			u_prev: np.ndarray, 
			u_next: np.ndarray, 
			c_sqrd_over_h_sqrd: np.ndarray, 
			f: np.ndarray, 
			delta_t_sqrd: float,
			Nx: int, 
			Nz: int) -> None: 
	
	"""
	Performs a single time step of the wave equation using the finite difference method.
	inputs: 
		u: 2d numpy array current wavefield
		u_prev: 2d numpy array previous wavefield
		u_next: 2d numpy array of the next wavefield
		c_sqrd_over_h_sqrd: 2d numpy array of the wave speed sqaured devided by the grid spacing squared
		f: 2d numpy array of the source field at the current time step
		delta_t_sqrd: float, square of the time step 
		Nx: int, number of grid points in the x-direction
		Nz: int, number of grid points in the z-direction
	"""

	for i in range(1, Nz-1): 
		for j in range(1, Nx-1): 
			u_next[i, j] = delta_t_sqrd * (
				c_sqrd_over_h_sqrd[i, j] * (
					u[i+1, j] + u[i-1, j] + u[i, j+1] + u[i, j-1] - 4.0 * u[i, j]) 
					+ f[i, j] ) + 2.0 * u[i, j] - u_prev[i, j] 
			
			
@jit(nopython=True, fastmath=True, cache=True)
def apply_staceys_ABCs(u: np.ndarray, 
					  u_next: np.ndarray, 
					  c: np.ndarray,
					  delta_t: float, 
					  h: float, 
					  ABC_sides: list[int]) -> None:
	
	"""
	Applies Stacey's Absorbing Boundary Conditions (ABCs) to the wavefield.
	inputs: 
		u: 2d numpy array current wavefield
		u_next: 2d numpy array of the next wavefield, this will updated to account for ABCs
		c: 2d numpy array of the wave speed
		delta_t: float, time step
		h: float, grid spacing
		ABC_sides: list of integers indicating which sides to apply ABCs to
			1 - left, 2 - bottom, 3 - right, 4 - top
	"""
			
	# left boundary
	if 1 in ABC_sides: 
		u_next[:, 0] = ((c[:, 0] * delta_t) * (u[:, 1] - u[:, 0]) / h + u[:, 0])
	# right boundary
	if 3 in ABC_sides:
		u_next[:, -1] = (-(c[:, -1] * delta_t) * (u[:, -1] - u[:, -2]) / h + u[:, -1])
	# top boundary
	if 4 in ABC_sides:
		u_next[0, :] = ((c[0, :] * delta_t) * (u[1, :] - u[0, :]) / h + u[0, :])
	# bottom boundary
	if 2 in ABC_sides: 
		u_next[-1, :] = (-(c[-1, :] * delta_t) * (u[-1, :] - u[-2, :]) / h + u[-1, :])


@jit(nopython=True, fastmath=True, cache=True)
def apply_sponge_ABCs(u: np.ndarray, 
					  u_next: np.ndarray, 
					  u_prev: np.ndarray,
					  delta_t: float, 
					  ABC_sides: list[int], 
					  sponge_pnts: int,
					  delta_t_sqrd: float,
					  gamma1: np.ndarray,
					  gamma1_T: np.ndarray,
					  gamma1_sqrd: np.ndarray,
					  gamma1_T_sqrd: np.ndarray,
					  gamma2: np.ndarray,
					  gamma2_T: np.ndarray,
					  gamma2_sqrd: np.ndarray,
					  gamma2_T_sqrd: np.ndarray) -> None:
	
	""""
	Applies sponge absorbing boundary conditions to the wavefield.
	inputs:
		u: 2d numpy array current wavefield
		u_next: 2d numpy array of the next wavefield, this will updated to account for ABCs
		u_prev: 2d numpy array of the previous wavefield
		delta_t: float, time step
		ABC_sides: list of integers indicating which sides to apply ABCs to
			1 - left, 2 - bottom, 3 - right, 4 - top
		sponge_pnts: int, number of points in the sponge layer
		delta_t_sqrd: float, square of the time step
		gamma1: 1 x sponge_pnts numpy array of the sponge coefficients for the left boundary
		gamma1_T: sponge_pnts x 1 numpy array of the sponge coefficients for the top boundary
		gamma1_sqrd: 1 x sponge_pnts numpy array of the square of the sponge coefficients for the left boundary
		gamma1_T_sqrd: sponge_pnts x 1 numpy array of the square of the sponge coefficients for the top boundary
		gamma2: 1 x sponge_pnts numpy array of the sponge coefficients for the right boundary
		gamma2_T: sponge_pnts x 1 numpy array of the sponge coefficients for the bottom boundary
		gamma2_sqrd: 1 x sponge_pnts numpy array of the square of the sponge coefficients for the right boundary
		gamma2_T_sqrd: sponge_pnts x 1 numpy array of the square of the sponge coefficients for the bottom boundary
	Note that gamma2 is a reflected version of gamma1
	"""

	for i in range(sponge_pnts):
		# left boundary
		if 1 in ABC_sides:
			u_next[:, i] -= ((gamma1_sqrd[0, i]) * delta_t_sqrd * u[:, i] + 
							 gamma1[0, i] * 2 * delta_t * (u[:, i] - u_prev[:, i]))
		# right boundary
		if 3 in ABC_sides:
			u_next[:, -1-i] -= ((gamma2_sqrd[0, -1-i]) * delta_t_sqrd * u[:, -1-i] + 
							 gamma2[0, -1-i] * 2 * delta_t * (u[:, -1-i] - u_prev[:, -1-i]))
		# bottom boundary
		if 4 in ABC_sides:
			u_next[i, :] -= ((gamma1_T_sqrd[i, 0]) * delta_t_sqrd * u[i, :] + 
							 gamma1_T[i, 0] * 2 * delta_t * (u[i, :] - u_prev[i, :]))
		# top boundary
		if 2 in ABC_sides:
			u_next[-1-i, :] -= ((gamma2_T_sqrd[-1-i, 0]) * delta_t_sqrd * u[-1-i, :] + 
							 gamma2_T[-1-i, 0] * 2 * delta_t * (u[-1-i, :] - u_prev[-1-i, :])) 
			
						
def wave_solver(model_params: PT_data_classes.Acoustic_2D_Grid_model_params, 
				aquisition_params: PT_data_classes.Aquisition_params,
				mode: str,
				save_inds: list[int] = [],
				u_prev: np.ndarray = None,
				u: np.ndarray = None,
				wave_fields: list[np.ndarray] = None,
				compute_hessian_kernel: bool = False,
				save_adjoint_wavefield: bool = False) -> tuple[list[np.ndarray], np.ndarray, np.ndarray | None, np.ndarray | None]:
	
	"""
	Performs a forward/adjoint simulation of the wave equation using finite difference method.
	inputs:
		model_params: model parameters dataclass
		aquisiton_parmas: aquisition parameters datacalss
		mode: string equal to either "forward" or "adjoint" 
		save_inds: list of integers indicating which time steps to save the wavefield
		u_prev: 2d numpy array previous wavefield (optional) for t=0 initial condition
		u: 2d numpy array current wavefield (optional) for t=0 initial condition
		wave_fields: list of 2d numpy arrays of the forward wavefield at the saved time indices (only used if mode == "adjoint")
		compute_hessian_kernel: weather or not to compute hessian kernel (only used if mode == "adjoint")
		save_adj_wavefield: bool, whether to save the adjoint wavefield as the same tiem steps as the (only used if mode == "adjoint")
	outputs:
		wave_fields: list of 2d numpy arrays of the wavefield at the saved time steps
		SG: 2d numpy array of the shot gather, shape (Nt, len(rec_inds)), where each collumn is a 
			receiver at the corresponding index in rec_inds
		SK: 2d numpy array of the sensitivity kernel, shape (Nz, Nx) or None is mode == forward
		HK: 2d numpy array of the Hessian kernel, shape (Nz, Nx) if compute_hessian_kernel is True and model == adjoint, else None
	"""

	# check if model is either forward or adjoint
	if mode != "forward": 
		if mode != "adjoint":
			raise ValueError("mode must be equal to either forward or adjoint")

	# check to that time step is not too big
	if aquisition_params.delta_t > model_params.h / (2.0 * np.max(model_params.c)):
		print("The maximum allowable time step is {:.10f} seconds.".format(model_params.h / (2.0 * np.max(model_params.c))))
		raise ValueError("Time step is too large for stability. Reduce delta_t.")

	# initialize wave field arrays
	if u_prev is None:
		u_prev = np.zeros((model_params.Nz, model_params.Nx), dtype=np.float32, order='C')
	if u is None:
		u = np.zeros((model_params.Nz, model_params.Nx), dtype=np.float32, order='C')
	u_next = np.zeros((model_params.Nz, model_params.Nx), dtype=np.float32, order='C')

	# initialzie Fource array 
	F = np.zeros((model_params.Nz, model_params.Nx), dtype=np.float32, order='C')

	# initialize wavespeed array
	c_sqrd_over_h_sqrd = model_params.c**2 /model_params.h**2

	# initalize the sensitivity kernels
	if mode == "adjoint":

		inv_c_cubd = model_params.c**-3
		
		# initialize sensitivity kernel 
		SK = np.zeros((model_params.Nz, model_params.Nx), dtype=np.float32, order='C')

		# initialize Hessian kernel if requested
		if compute_hessian_kernel:
			HK = np.zeros((model_params.Nz, model_params.Nx), dtype=np.float32, order='C')
		else:
			HK = None

		# initialize saved adjoint wavefields 
		adj_wave_fields = []
	
	elif mode == "forward": 
		# initialize list of saved wavefield
		wave_fields: list[np.ndarray] = []
		# set sensitivity and hessian kernels to None (for output)
		SK = None
		HK = None

	# initiaze shot gather array
	SG = np.zeros((aquisition_params.Nt, len(aquisition_params.rec_inds)))

	# initialize extra variables if sponge ABCs are used
	if model_params.sponge_ABCs:

		delta_t_sqrd = aquisition_params.delta_t**2

		gamma1 = np.zeros((1, model_params.sponge_pnts), dtype=np.float32)
		gamma1[0, :] = model_params.sponge_alpha * (1.0 - np.sin(np.pi * np.arange(model_params.sponge_pnts) / (2.0 * model_params.sponge_pnts))**2)
		gamma1_T = gamma1.T
		gamma1_sqrd = gamma1**2
		gamma1_T_sqrd = gamma1_T**2

		gamma2 = np.flip(gamma1)
		gamma2_T = gamma2.T
		gamma2_sqrd = gamma2**2
		gamma2_T_sqrd = gamma2_T**2

	# iterate through time steps
	for k in range(aquisition_params.Nt):

		# pull out data at reviever locations
		for ri in range(len(aquisition_params.rec_inds)):
			SG[k, ri] = u[aquisition_params.rec_inds[ri]]

		# set up force for time step
		for si in range(len(aquisition_params.source_inds)): 
			F[aquisition_params.source_inds[si]] = aquisition_params.STFS[k, si]
	
		# apply time step
		Time_Step(u, 
				u_prev, 
				u_next, 
				c_sqrd_over_h_sqrd, 
				F, 
				aquisition_params.delta_t**2, 
				model_params.Nx, 
				model_params.Nz)

		if model_params.stacey_ABCs: 
			apply_staceys_ABCs(u, 
					  u_next, 
					  model_params.c, 
					  aquisition_params.delta_t,
					  model_params.h, 
					  model_params.ABC_sides)
		
		if model_params.sponge_ABCs:
			apply_sponge_ABCs(u,
					 u_next,
					 u_prev, 
					 aquisition_params.delta_t, 
					 model_params.ABC_sides,  
					 model_params.sponge_pnts,
					 delta_t_sqrd,
					 gamma1,
					 gamma1_T,
					 gamma1_sqrd,
					 gamma1_T_sqrd,
					 gamma2,
					 gamma2_T,
					 gamma2_sqrd,
					 gamma2_T_sqrd)
			
		
		# for forward simulation, check if the time index is supposed to be saved
		if mode == "forward":
			if k in save_inds: 
				# save second deriviative of wavefield if 
				wave_fields.append(u_prev - 2*u + u_next)

		# for adjoint simulation, 
		elif mode == "adjoint": 
			# map the current time step index to the forward index
			fwd_indx = (aquisition_params.Nt-1)-k

			# check to see if the forward index was in save_inds
			if fwd_indx in save_inds:
				
				if save_adjoint_wavefield: 
					# save adjoint wavefield
					adj_wave_fields.append(np.copy(u))
				
				# update sensitivity kernel(s)
				if fwd_indx != 0 and fwd_indx != aquisition_params.Nt-1:

					# map the timestep index to the wavefield list index
					wfi = save_inds.index(fwd_indx)
					# pull wavefield out of wavefield list
					dt2u = wave_fields[wfi]
					SK += dt2u * u * inv_c_cubd

					# compute hessian kernel if requested
					if compute_hessian_kernel: 
						HK += dt2u**2

			
		u_prev = np.copy(u)
		u = np.copy(u_next)
	
	if mode == "forward":
		return wave_fields, SG
	if mode == "adjoint":
		return adj_wave_fields, SG, SK, HK


