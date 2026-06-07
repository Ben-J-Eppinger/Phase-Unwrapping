import numpy as np
import scipy
import pickle
from PT_data_classes import Aquisition_params


def load_data_set(path: str):
	with open(path+"/data_set.pkl", 'rb') as f: 
		data_set = pickle.load(f) 
	return data_set


class shot_gather:


	def __init__(self, 
			  aquisition_params: Aquisition_params,
			  gather: np.ndarray):
		"""
		Initialize a shot gather object.
		Inputs:
			aquisition_params: aquisition_params data class
			gather: 2D numpy array of shape (Nt, Nr) containing recorded data
		"""
		self.aquisition_params = aquisition_params
		self.gather = gather


	def lowpass_filter(self, cutoff_freq: float, order: int = 5):
		"""
		Apply a lowpass Butterworth filter to the shot gather data.
		Inputs:
			cutoff_freq: cutoff frequency of the lowpass filter
			order: order of the Butterworth filter
		"""
		fs = 1.0 / self.aquisition_params.delta_t
		sos = scipy.signal.butter(order, cutoff_freq, btype='lowpass', output='sos', fs=fs)
		self.gather = scipy.signal.sosfilt(sos, self.gather, axis=0)

	
	def normalize(self):
		self.gather = self.gather / np.max(self.gather, axis=0)


class data_set:


	def __init__(self, 
			  shot_gathers: list[shot_gather]):
		"""
		Initialize a data_set object, which consists of multiple shot gathers.
		Inputs:
			shot_gathers: list of shot_gather objects
		"""
		self.shot_gathers = shot_gathers


	def save_data_set(self, path: str):
		with open(path+"/data_set.pkl", "wb") as f:
			pickle.dump(self, f)


	# parralellize this function
	def lowpass_filter(self, cutoff_freq: float, order: int = 5):
		for shot in self.shot_gathers:
			shot.lowpass_filter(cutoff_freq, order)

	
	def normalize(self):
		for shot in self.shot_gathers:
			shot.normalize()