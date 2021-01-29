
"""
Module calcualtes standard errors 

Script must be run using Mpi with 8 * number of  cores per  group in Tau

Example (on Gadi normal compute node):

module load python3/3.7.4
module load openmpi/4.0.2

mpiexec -n 160 python3 -m mpi4py stderrs.py

"""

# Import external packages
import yaml
import gc
import numpy as np
import time
import warnings
warnings.filterwarnings('ignore')
import csv
import time
import dill as pickle 
import copy
import sys
import pandas as pd

# Import education model modules
import edumodel 
from solve_policies.studysolver import generate_study_pols

# put all the tabulate functions here or import 


if __name__ == "__main__":

	from mpi4py import MPI as MPI4py
	MPI4py.pickle.__init__(pickle.dumps, pickle.loads)
	from pathlib import Path

	
	# Estimation parameters  
	estimation_name = 'Preliminary_all_v5'
	scr_path = '/scratch/pv33/edu_model_temp/' + estimation_name
	
	# world communicator class
	world = MPI4py.COMM_WORLD
	world_size = world.Get_size()
	world_rank = world.Get_rank()

	# number of cores/ number of groups = number of paramters 
	# color tells me the tau group I belong to 
	# key tells me my rank within the group 
	number_tau_groups = 8
	block_size_layer_1 = world_size/number_tau_groups
	color_layer_1 = int(world_rank/block_size_layer_1)
	key_layer1 = int(world_rank%block_size_layer_1)

	# split the mpi communicators 
	tau_world = world.Split(color_layer_1,key_layer1)
	tau_world_rank = tau_world.Get_rank()


	# Eeach core opens the baseline parameters and model settings based on its color (tau group)
	# Load the data and sort and map the moments 
	# moments_data are moments for this group 
	moments_data_raw= pd.read_csv('{}moments_clean.csv'\
					.format(settings_folder))
	moments_data_mapped = map_moments(moments_data_raw)
	# Assign model tau group according to each core according to processor color
	model_name = list(moments_data_mapped.keys())[color_layer_1]
	moments_data = moments_data_mapped[model_name]['data_moments']

	# Load model settings 
	with open("{}settings.yml".format(settings_folder), "r") as stream:
		edu_config = yaml.safe_load(stream)

	param_random_bounds = {}
	with open('{}random_param_bounds.csv'.format(settings_folder), newline='') as pscfile:
		reader_ran = csv.DictReader(pscfile)
		for row in reader_ran:
			param_random_bounds[row['parameter']] = np.float64([row['LB'],
																row['UB']])

	# Load the exogenous shocks 
	U = np.load(scr_path+'/'+ 'U.npy')
	U_z = np.load(scr_path+'/'+ 'U_z.npy')

	# Load the parameters theta_0
	sampmom = pickle.load(open("/scratch/pv33/edu_model_temp/{}/latest_sampmom.smms".format(estimation_name + '/'+ model_name),"rb"))
	theta_0 = sampmom[0]

	# Define a function that takes in a paramter vector (possibly perturbed) and returns moments
	# Note we use the config for the tau group 
	# Covariance matrix is degenerate 

	def gen_moments(params):
		# Create an edu model class using the input params
		param_cov = np.zeros(np.shape(sampmom[1]))
		edu_model = edumodel\
				.EduModelParams(model_name,\
							edu_config[model_name],
							U, U_z, 
							random_draw = True,
							random_bounds = param_random_bounds, 
							param_random_means = params, 
							param_random_cov = param_cov, 
							uniform = False)
		# run the model 
		moments_sim = generate_study_pols(edu_model.og)

		# list moments into 1-D array (note order maintained)
		moments_sim_array = np.array(np.ravel(moments_sim))
		moments_data_array = np.array(np.ravel(moments_data))

		#moments where data moments are NA 
		moments_sim_array_clean = moments_sim_array[~np.isnan(moments_data_array)]

		return moments_sim_array_clean

	# ANDREW TO DO HERE
	# EACH CPU IN EACH TAU GROUP DOES CENTRAL DIFFERENCES FOR ONE PARAMETER 
	# THERE ARE 20 CPUS PER TAU GROUP 
	# THE TAU RANK WILL DETERMINE WHICH PARAMTERS THE CPU DOES CE FOR 

	epsilon = ?
	# generate pertubed parameter +e
	theta_e_plus = theta_0[tau_world_rank] + epsilon 
	moments_sim_array_clean_plus_epsilon = gen_moments(theta_e_plus)

	theta_e_minus = theta_0[tau_world_rank] - epsilon 
	moments_sim_array_clean_minus_epsilon = gen_moments(theta_e_minus)

	# NOW CALCULATE array of central differences wrt parameter tau here
	# delta_moments_del_tau = 

	# Then the master of each tau world gaters all the arrays
	# Note the order is maintained so the jacobian_T[0] will be the derivative wrt to 
	# parameter calculated on tau_world.rank = 0, hence the paramter with index zero in the theta_0 array
	
	tau_world.Barrier()

	jacobian_T = tau_world.gather(delta_moments_del_tau, root = 0)
	jacobian = np.transpose(np.array(jacobian_T))

	# Now, andrew, on each master tau_core do the manipulation to generate the standard erros and tabulate results table

	if tau_world_rank == 0:
		# do manupulation, transpose inverse etc. 
		# create SE errors array, take np.sqrt of diagolas
		# plot estimates table 



	