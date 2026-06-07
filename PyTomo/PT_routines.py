import PT_data
import PT_model
import PT_data_classes
from joblib import Parallel, delayed
import numpy as np
import scipy
from copy import copy


def Compute_Data(model: PT_model.Acoustic_2D_Cartesian_Grid, 
				 aquisition_params: PT_data_classes.Aquisition_params) -> PT_data.shot_gather:
	"""
	Computes data for a given model, aquisition parameters. Note that we assume an initial 
		wavefield at rest and do not save the wavefield at any timesteps
	inputs: 
		model:
		aquisition_params
	outputs: 
		SG: a PyTomo shot gather object 
	"""

	# call forward solver
	SG: np.ndarray = model.forward(aquisition_params=aquisition_params,
						   save_inds=[])[1]

	# transform shot gather arrays to shot gather object 
	SG: PT_data.shot_gather = PT_data.shot_gather(aquisition_params=aquisition_params,
						  gather=SG)
	
	return SG


def Compute_Data_Set(model_list: list[PT_model.Acoustic_2D_Cartesian_Grid],
					 aquisition_params_list: list[PT_data_classes.Aquisition_params]) -> PT_data.data_set:
	"""
	Computes a dataset (which contains a list of shot gathers) for a set of soruce indicies
	inputs: 
		model_list:
		aquisition_params_list:
		"""

	# create a list of the forward Arguments
	arg_list = [i for i in zip(model_list, aquisition_params_list)]

	# compute data set in parallel
	data_set: list[PT_data.shot_gather] = Parallel(n_jobs=-1)(delayed(Compute_Data)(*args) for args in arg_list)

	# transform data set list into PT data set object
	data_set: PT_data.data_set = PT_data.data_set(data_set)

	return data_set


def Compute_Kernel(model: PT_model.Acoustic_2D_Cartesian_Grid, 
				   aquisition_params: PT_data_classes.Aquisition_params,
				   preprocess_params: PT_data_classes.Preprocess_params,
				   adjoint_params: PT_data_classes.Adjoint_params,
				   save_inds: list[int],
				   SG_obs: PT_data.shot_gather,
				   compute_hessian_kernel: bool = True) -> tuple[np.ndarray, np.ndarray, np.ndarray]:

	# Call Forward Solver
	u_out, SG = model.forward(aquisition_params=aquisition_params,
					  save_inds=save_inds)[0:2]
	
	# convert synthetic shotgather from nuympy array to shot gather object
	SG_syn = PT_data.shot_gather(aquisition_params=aquisition_params, gather=SG)
	del SG

	# Preprocess Data
	if preprocess_params.mute:
		print("Mute is not implemented yet")
	if preprocess_params.filter:
		if preprocess_params.filter == "lowpass":
			SG_syn.lowpass_filter(preprocess_params.filter_bounds)
			SG_obs.lowpass_filter(preprocess_params.filter_bounds)
		# implement other types of filter eventually
	if preprocess_params.normalize:
		SG_syn.normalize()
		SG_obs.normalize()

	# Compute Adjoint Source
	adj_STFS, misfits = adjoint_params.adjoint_func(*([SG_syn, SG_obs]+adjoint_params.additonal_parmas))

	# initialize aquisition parameters for adjiont simulation and replace STFS and receiver locations 
	adj_aquisition_params = copy(aquisition_params)
	adj_aquisition_params.STFS = adj_STFS
	adj_aquisition_params.source_inds = aquisition_params.rec_inds

	# call adjoint solver
	SK, HK = model.adjoint(aquisition_params=adj_aquisition_params, 
						save_inds=save_inds, 
						u_prev=None, 
						u=None, 
						wave_fields=u_out, 
						compute_hessian_kernel=compute_hessian_kernel)[2:4]
	del u_out

	# Return Sensitivity Kernel, Hessian Kernel and Adjoint Sources
	return SK, HK, adj_STFS, misfits	


def Compute_Kernel_Parallel(model_list: list[PT_model.Acoustic_2D_Cartesian_Grid],
							aquisition_params_list: list[PT_data_classes.Acoustic_2D_Grid_model_params],
							preprocess_params_lsit: list[PT_data_classes.Preprocess_params],
							adjoint_params_list: list[PT_data_classes.Adjoint_params],
							save_inds_list: list[list[int]],
							SG_OBS_list: list[PT_data.shot_gather]):
	
	arg_list = [
		i for i in zip(model_list, 
							aquisition_params_list,
							preprocess_params_lsit,
							adjoint_params_list,
							save_inds_list, 
							SG_OBS_list)
							]

	OUT = Parallel(n_jobs=-1)(delayed(Compute_Kernel)(*args) for args in arg_list)

	return OUT