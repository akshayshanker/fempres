 
"""
Module estimates EduModel using Simualted Method of Moments
Minimisation performed using Cross-Entropy method (see Kroese et al)

Script must be run using Mpi with 8 * number of  cores per  group in Tau

Example (on Gadi normal compute node):

module load python3/3.7.4
module load openmpi/4.0.2

mpiexec -n 480 python3 -m mpi4py smm.py

"""

# Import packages
import yaml
import gc
import numpy as np
import time
import warnings
warnings.filterwarnings('ignore')
from collections import defaultdict
from numpy import genfromtxt
import csv
import time
import dill as pickle 
import copy
import sys
import importlib
import pandas as pd

# Education model modules
import edumodel 
from solve_policies.studysolver import generate_study_pols

def gen_RMS(edu_model,\
			moments_data):
	
	"""
	Simulate model for a given param vector and generate error  
	between simulated moments for parameterised EduModel instance 
	and sorted mapped data moments.

	Parameters
	---------
	edu_model: EduModel object
	moments_data: array
				  sorted moments for group

	Returns 
	------
	error : 

	"""
	moments_sim = generate_study_pols(edu_model.og)


	moments_sim_array = np.array(np.ravel(moments_sim))
	moments_data_array = np.array(np.ravel(moments_data))

	deviation = (moments_sim_array\
				[~np.isnan(moments_data_array)]\
				  - moments_data_array\
				  [~np.isnan(moments_data_array)])/moments_data_array[~np.isnan(moments_data_array)]
	
	norm = 1
	N_err = len(deviation)

	return 1-np.sqrt((1/N_err)*np.sum(np.square(deviation))/norm)


def load_tm1_iter():
	""" Initializes array of best performer and least best in the 
		elite set (see notationin Kroese et al)
	""" 

	S_star,gamma_XEM	= np.full(d,0), np.full(d,0)
	t = 0

	return gamma_XEM, S_star,t

def iter_SMM(config, 			 # configuration settings for the model name 
			 model_name, 		 # name of model (tau group name)
			 U, U_z, 
			 param_random_bounds, # bounds for parameter draws
			 sampmom, 	    # t-1 parameter means 
			 moments_data,  # data moments 
			 gamma_XEM, 	# lower elite performer
			 S_star, 		# upper elite performer
			 t,		# iteration 
			 tau_world, # communicate class for groups
			 N_elite):	# iteration number 
	
	""" Initializes parameters and EduModel model and peforms 
		one iteration of the SMM, returning updated sampling distribution
		for the parameters

	'"""

	# Draw uniform parameters for the first iteration 
	if t==0:
		uniform = True
	else:
		uniform = False

	if tau_world.rank == 0:
		print("...generating sampling moments")

	edu_model = edumodel\
					.EduModelParams(model_name,\
								config,
								U, U_z, 
								random_draw = True,
								random_bounds = param_random_bounds, 
								param_random_means = sampmom[0], 
								param_random_cov = sampmom[1], 
								uniform = uniform)

	parameters = edu_model.parameters

	if tau_world.rank == 0:
		print("Random Parameters drawn, distributng iteration {} \
					for tau_group {}".format(t,model_name))
		
	RMS =  gen_RMS(edu_model, moments_data)

	errors_ind = [edu_model.param_id, np.float64(RMS)]

	# Gather parameters and corresponding errors from all ranks 
	tau_world.Barrier()

	indexed_errors = tau_world.gather(errors_ind, root = 0)

	parameter_list = tau_world.gather(parameters, root = 0)

	# tau_world master does selection of elite parameters and drawing new means 
	if tau_world.rank == 0:
		indexed_errors = \
			np.array([item for item in indexed_errors if item[0]!='none'])
		parameter_list\
			 = [item for item in parameter_list if item is not None]
 
		parameter_list_dict = dict([(param['param_id'], param)\
							 for param in parameter_list])
		
		errors_arr = np.array(indexed_errors[:,1]).astype(np.float64)


		error_indices_sorted = np.take(indexed_errors[:,0],\
									 np.argsort(-errors_arr))
		errors_arr_sorted = np.take(errors_arr, np.argsort(-errors_arr))

		number_N = len(error_indices_sorted)
		
		elite_errors = errors_arr_sorted[0: N_elite]
		elite_indices = error_indices_sorted[0: N_elite]

		weights = np.exp((elite_errors - np.min(elite_errors))\
						/ (np.max(elite_errors)\
							- np.min(elite_errors)))
		gamma_XEM = np.append(gamma_XEM,elite_errors[-1])
		S_star = np.append(S_star,elite_errors[0])

		error_gamma = gamma_XEM[d +t] \
						- gamma_XEM[d +t -1]
		error_S = S_star[int(d +t)]\
						- S_star[int(d +t -1)]

		means, cov = gen_param_moments(parameter_list_dict,\
								param_random_bounds,\
								elite_indices, weights)

		print("...generated and saved sampling moments")
		print("...time elapsed: {} minutes".format((time.time()-start)/60))

		return number_N, [means, cov], gamma_XEM, S_star, error_gamma,\
											 error_S, elite_indices[0]
	else:
		return 1 

def gen_param_moments(parameter_list_dict,\
						param_random_bounds,\
						 selected,\
						 weights):

	""" Estimate params of a sampling distribution

	Parameters
	----------
	parameter_list_dict: Dict
						  Dictionary with all paramameters
						  with ID keys
	selected: 2D-array
						   set of elite paramters IDs and errors

	Returns
	-------

	means

	cov
	"""

	sample_params = []

	for i in range(len(selected)):
		rand_params_i = []
		for key in param_random_bounds.keys():
			rand_params_i.append(\
				float(parameter_list_dict[selected[i]][key]))
		
		sample_params.append(rand_params_i)

	sample_params = np.array(sample_params)
	means = np.average(sample_params, axis = 0)
	cov = np.cov(sample_params, rowvar = 0)

	return means, cov

def map_moments(moments_data):

	group_list = moments_data['group'].unique()

	def gen_group_list(somelist):
		return {x: {} for x in somelist}

	moments_grouped_sorted = gen_group_list(group_list)

	list_moments = ['av_final',\
					'av_mark',\
					'av_markw13_exp1',\
					'av_markw13_exp2',\
					'av_game_session_hours_cumul',\
					'av_ebook_session_hours_cumul',\
					'av_mcq_session_hours_cumul',\
					'av_saq_session_hours_cumul',\
					'av_player_happy_deploym_cumul',\
					'av_mcq_attempt_nonrev_cumul',\
					'av_sa_attempt_cumul', \
					'av_mcq_Cshare_nonrev_cumul',\
					'sd_final',\
					'sd_mark',\
					'sd_markw13_exp1',\
					'sd_markw13_exp2',\
					'sd_game_session_hours_cumul',\
					'sd_ebook_session_hours_cumul',\
					'sd_mcq_session_hours_cumul',\
					'sd_saq_session_hours_cumul',\
					'sd_player_happy_deploym_cumul',\
					'sd_mcq_attempt_nonrev_cumul',\
					'sd_sa_attempt_cumul', \
					'sd_mcq_Cshare_nonrev_cumul',\
					'acgame_session_hours',\
					'acebook_session_hours',\
					'acmcq_session_hours',\
					'acsaq_session_hours',\
					'acmcq_Cshare_nonrev',\
					'cmcsaq_session_hours',\
					'cgsaq_session_hours',\
					'cgmcq_session_hours',\
					'cesaq_session_hours',\
					'cemcq_session_hours',\
					'ceg_session_hours',\
					'c_atar_ii']

	for keys in moments_grouped_sorted:
		moments_grouped_sorted[keys]['data_moments'] = np.empty((11,36))

	for i in range(len(list_moments)):
		for key in moments_grouped_sorted:
			moments_data_for_gr = moments_data[moments_data['group'] == key]
			moments_grouped_sorted[key]['data_moments'][:,i] = moments_data_for_gr[list_moments[i]]

	return moments_grouped_sorted

if __name__ == "__main__":

	from mpi4py import MPI as MPI4py
	MPI4py.pickle.__init__(pickle.dumps, pickle.loads)
	from pathlib import Path

	
	# Estimation parameters  
	tol = 1E-8
	N_elite = 50
	d = 3
	estimation_name = 'test'
	world = MPI4py.COMM_WORLD
	number_tau_groups = 8 

	# Folder for settings in home and declare scratch path
	settings_folder = 'settings/'
	scr_path = '/scratch/pv33/edu_model_temp/' + '/' + estimation_name
	Path(scr_path).mkdir(parents=True, exist_ok=True)

	# Create communicators. Each processor sub-group gets a tau group
	world_size = world.Get_size()
	world_rank = world.Get_rank()
	blocks_layer_1 = number_tau_groups
	block_size_layer_1 = world_size/blocks_layer_1
	color_layer_1 = int(world_rank/block_size_layer_1)
	key_layer1 = int(world_rank%block_size_layer_1)
	tau_world = world.Split(color_layer_1,key_layer1)
	tau_world_rank = tau_world.Get_rank()

	# Load the data and sort and map the moments 
	moments_data = pd.read_csv('{}moments_clean.csv'\
					.format(settings_folder))
	moments_data_mapped = map_moments(moments_data)

	# Assign model tau group according to each core according to processor color
	model_name = list(moments_data_mapped.keys())[color_layer_1]
	Path(scr_path + '/' + model_name).mkdir(parents=True, exist_ok=True)

	# Load model settings 
	with open("{}settings.yml".format(settings_folder), "r") as stream:
		edu_config = yaml.safe_load(stream)

	param_random_bounds = {}
	with open('{}random_param_bounds.csv'.format(settings_folder), newline='') as pscfile:
		reader_ran = csv.DictReader(pscfile)
		for row in reader_ran:
			param_random_bounds[row['parameter']] = np.float64([row['LB'],
																row['UB']])

	# Generate random points for beta
	U = np.random.rand(edu_config[model_name]['parameters']['N'],\
						edu_config[model_name]['parameters']['T'])

	# Generate random points for ability and percieved ability 
	U_z = np.random.rand(edu_config[model_name]['parameters']['N'],2)

	# Initialize the model with uniform values on each core 
	with open('{}random_param_bounds.csv'\
	.format(settings_folder), newline='') as pscfile:
		reader_ran = csv.DictReader(pscfile)
		for row in reader_ran:
			param_random_bounds[row['parameter']] = np.float64([row['LB'],\
				row['UB']])

	start = time.time()

	# Initialize the SMM error grid
	gamma_XEM, S_star,t = load_tm1_iter()
	error = 1 
	sampmom = [0,0]
	
	# Iterate on the SMM
	while error > tol:
		iter_return = iter_SMM(edu_config[model_name],
								 model_name,
								 U, U_z, 
								 param_random_bounds,
								 sampmom,
								 moments_data_mapped[model_name]['data_moments'],
								 gamma_XEM,
								 S_star,
								 t, tau_world,N_elite) 
		tau_world.Barrier()	

		if tau_world.rank == 0:
			#print(sampmom[0])
			number_N, sampmom, gamma_XEM, S_star, error_gamma, error_S, top_ID = iter_return
			error_cov = np.abs(np.max(sampmom[1]))
			
			pickle.dump(gamma_XEM,open("/scratch/pv33/edu_model_temp/{}/{}/gamma_XEM.smms"\
						.format(estimation_name, model_name),"wb"))
			pickle.dump(S_star,open("/scratch/pv33/edu_model_temp/{}/{}/S_star.smms"\
						.format(estimation_name, model_name),"wb"))
			
			pickle.dump(t,open("/scratch/pv33/edu_model_temp/{}/{}/t.smms"\
						.format(estimation_name, model_name),"wb"))
			pickle.dump(sampmom,open("/scratch/pv33/edu_model_temp/{}/{}/latest_sampmom.smms"\
						.format(estimation_name, model_name),"wb"))
			pickle.dump(top_ID, open("/scratch/pv33/edu_model_temp/{}/{}/topid.smms"\
						.format(estimation_name, model_name),"wb"))
			
			print("...iteration {} on {} models,elite_gamma error are {} and elite S error are {}"\
						.format(t, number_N, error_gamma, error_S))
			
			print("...cov error is {}."\
					.format(np.abs(np.max(sampmom[1]))))
			error = error_cov

		else:
			sampmom = None
			gamma_XEM = None
			S_star = None

		t = t+1
		tau_world.Barrier()
		sampmom = tau_world.bcast(sampmom, root = 0)
		gamma_XEM = tau_world.bcast(gamma_XEM, root = 0) 
		S_star = tau_world.bcast(S_star, root = 0)
