import numpy as np
import Acoustic_2D_Cartesian_Grid
from PT_data_classes import Acoustic_2D_Grid_model_params, Aquisition_params
from Acoustic_2D_Cartesian_Grid import wave_solver
import pickle


def load_model(path):
	with open(path+"/model.pkl", 'rb') as f: 
		model: Acoustic_2D_Cartesian_Grid = pickle.load(f) 
	return model


class Acoustic_2D_Cartesian_Grid: 


	def __init__(self, model_params: Acoustic_2D_Grid_model_params):
		self.model_params = model_params


	def save_model(self, path: str) -> None:
		with open(path+"/model.pkl", "wb") as f:
			pickle.dump(self, f)

	
	def forward(self, 
			 aquisition_params: Aquisition_params,
			 save_inds: list[int],
			 u_prev: np.ndarray = None,
			 u: np.ndarray = None):	
		"""
		Forward modeling of the acoustic wave equation on a 2D Cartesian grid.
		This function is a wrapper for wave_solver from Acoustic_2D_Cartesian_Grid.
		Inputs:
			aquistion params: holds info such a source and receiver indices
			save_inds: list of integers indicating which time steps to save
			u_prev: previous wavefield (optional, for time-stepping)
			u: current wavefield (optional, for time-stepping)
		"""

		return wave_solver(model_params=self.model_params,
					 aquisition_params=aquisition_params,
					 mode="forward", 
					 save_inds=save_inds,
					 u_prev=u_prev,
					 u=u)
	
	
	def adjoint(self,
			 aquisition_params: Aquisition_params,
			 save_inds: list[int],
			 u_prev: np.ndarray,
			 u: np.ndarray,
			 wave_fields: list[np.ndarray],
			 compute_hessian_kernel: bool = False,
			 save_adjoint_wavefield: bool = False):
		"""
		Adjoint modeling of the acoustic wave equation on a 2D Cartesian grid.
		This functtion is wrapper for wave_solver from Acoustic_2D_Cartesian_Grid.
		Inputs:
			aquistion params: holds info such a source and receiver indices
			save_inds: list of integers indicating which time steps to save
			u_prev: previous wavefield (optional, for time-stepping)
			u: current wavefield (optional, for time-stepping)
			wave_fields: list of wavefields from the forward simulation
			compute_hessian_kernel: bool indicating whether to compute Hessian kernel
			save_adj_wavefield: bool indicating whether to save the adjoint wavefield
		"""

		return wave_solver(model_params=self.model_params,
					 aquisition_params=aquisition_params,
					 mode="adjoint", 
					 save_inds=save_inds,
					 u_prev=u_prev,
					 u=u,
					 wave_fields=wave_fields,
					 compute_hessian_kernel=compute_hessian_kernel,
					 save_adjoint_wavefield=save_adjoint_wavefield)


