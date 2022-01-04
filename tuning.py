import numpy as np
import scipy 
import pdb

from decoders import standardize, expand_state_space 

def convert_polar(Z):

	rho = np.zeros((Z.shape[0], Z.shape[1], 3))
	theta = np.zeros((Z.shape[0], Z.shape[1], 3))

	for i in range(3):
		rho[..., i] = np.sqrt(np.power(Z[..., 2*i], 2) + np.power(Z[..., 2*i + 1], 2))
		theta[..., i] = np.sqrt()

	


def cosine_tuning(X, Z):

	X = standardize(X)
	Z = standardize(Z)
	Z = expand_state_space(Z, X)

	# Convert state space coordinates to polar coordinates
	rho, theta = convert_polar(Z)

	# Fit neural data tuning at a range of lags
