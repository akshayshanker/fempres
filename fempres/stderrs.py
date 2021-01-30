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
from scipy import stats
import sys
import pandas as pd
from results import make_tables

# Import education model modules
import edumodel 
from solve_policies.studysolver import generate_study_pols

# put all the tabulate functions here or import 


if __name__ == "__main__":

	from mpi4py import MPI as MPI4py
	MPI4py.pickle.__init__(pickle.dumps, pickle.loads)
	from pathlib import Path
	
	# Estimation settings and labels   
	estimation_name = 'Preliminary_all_v5'
	scr_path = '/scratch/pv33/edu_model_temp/' + estimation_name
	settings_folder = 'settings/'
	
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

	# Split the mpi communicators 
	tau_world = world.Split(color_layer_1,key_layer1)
	tau_world_rank = tau_world.Get_rank()

	# Communicators with all masters
	tau_masters = world.Split(tau_world_rank,color_layer_1)

	# Eeach core opens the baseline parameters and model settings based on its color (tau group)
	# Load the data and sort and map the moments 
	# moments_data are moments for this group 
	moments_data_raw= pd.read_csv('{}moments_clean.csv'\
					.format(settings_folder))
	moments_data_mapped = edumodel.map_moments(moments_data_raw)
	# Assign model tau group according to each core according to processor color
	model_name = list(moments_data_mapped.keys())[color_layer_1]
	moments_data = moments_data_mapped[model_name]['data_moments']

	# Load model settings 
	with open("{}settings.yml".format(settings_folder), "r") as stream:
		edu_config = yaml.safe_load(stream)

	param_random_bounds = {}
	with open('{}random_param_bounds_stdrr.csv'.format(settings_folder), newline='') as pscfile:
		reader_ran = csv.DictReader(pscfile)
		for row in reader_ran:
			param_random_bounds[row['parameter']] = np.float64([row['LB'],
																row['UB']])

	# Load the exogenous shocks 
	U = np.load(scr_path+'/'+ 'U.npy')
	U_z = np.load(scr_path+'/'+ 'U_z.npy')

	# Load the parameters theta_0
	sampmom = pickle.load(open("/scratch/pv33/edu_model_temp/{}/latest_sampmom.smms".format(estimation_name + '/'+ model_name),"rb"))
	theta_0 = np.array(sampmom[0])

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

	#Ideal epsilon default for finite difference
	#machine_epsilon = np.finfo(float).eps
	machine_epsilon = 10**(-3)
	epsilon = theta_0[tau_world_rank] * np.sqrt(machine_epsilon)  

	# generate pertubed parameter +e
	theta_e_plus = np.copy(theta_0)
	theta_e_plus[tau_world_rank] = theta_0[tau_world_rank] + epsilon 
	moments_sim_array_clean_plus_epsilon = gen_moments(theta_e_plus)

	theta_e_minus = np.copy(theta_0)
	theta_e_minus[tau_world_rank] = theta_0[tau_world_rank] - epsilon 
	moments_sim_array_clean_minus_epsilon = gen_moments(theta_e_minus)

	# NOW CALCULATE array of central differences wrt parameter tau here
	delta_moments_del_tau = (moments_sim_array_clean_plus_epsilon - moments_sim_array_clean_minus_epsilon)/(2*epsilon)

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
		S = 1000 #no. of simulations
		var_cov_matrix = (1 + 1/S)*np.linalg.pinv(np.dot(np.transpose(jacobian), jacobian))
		std_errs = np.sqrt(np.diagonal(var_cov_matrix))
		#print(std_errs)
	else: 
		std_errs = None

	world.barrier()
	# On master of tau_masters, collect the arrays 
	param_stds = np.array(tau_masters.gather(std_errs, root = 0))
	param_all = np.array(tau_masters.gather(theta_0, root = 0))

	# Collect the 8 std errors and tau_0s on the main 
	
	if world_rank == 0:
		print(param_all)
	#	 tabulate the results as below:
		param_names_old = [
		'beta_bar',
		'rho_beta',
		'sigma_beta',
		'delta',
		'zeta_star',
		'sigma_zeta',
		'zeta_hstar',
		'sigma_hzeta',
		'alpha',
		'lambda_E',
		'gamma_1',
		'gamma_2',
		'gamma_3',
		'sigma_M',
		'kappa_1',
		'kappa_2',
		'kappa_3',
		'kappa_4',
		'd',
		'varphi_sim']

		param_names_new = [
			'alpha',
			'beta_bar',
			'rho_beta',
			'sigma_beta',
			'delta',
			'kappa_3',
			'kappa_4',
			'kappa_1',
			'kappa_2',
			'd',
			'gamma_3',
			'gamma_1',
			'gamma_2',
			'sigma_M',
			'varphi_sim',
			#Gamma symbols below do not match table
			'lambda_E',
			'zeta_star',
			'sigma_zeta',
			'zeta_hstar',
			'sigma_hzeta']
		
		#Row names for table
		table_row_names = [
			"Course grade utility weight",
			#"Hyperbolic discount factor",
			"\hspace{0.4cm}Discount factor mean",
			"\hspace{0.4cm}Discount factor persistence",
			"\hspace{0.4cm}Discount factor std. deviation",
			"Hyperbolic discount factor",
			#"Study effectiveness for knowledge creation",
			"\hspace{0.4cm}Time solving MCQs",
			"\hspace{0.4cm}Time earning happiness units",
			"\hspace{0.4cm}Time answering SAQs",
			"\hspace{0.4cm}Time studying the textbook",
			"Knowledge stock depreciation",
			"\hspace{0.4cm}Solving MCQs",
			"\hspace{0.4cm}Answering SAQs",
			"\hspace{0.4cm}Studying the textbook",
			"Study elasticity of substitution",
			"Knowledge effectiveness of study output",
			#"Final exam ability",
			"Final exam difficulty parameter",
			"\hspace{0.4cm}Real exam ability mean",
			"\hspace{0.4cm}Real exam ability std. deviation",
			"\hspace{0.4cm}Perceived exam ability mean",
			"\hspace{0.4cm}Perceived exam ability std. deviation",
			]

		# Latex symbols for table row names, prefix r for raw string, or else causes unicode error
		table_row_symbols = [
			r"$\alpha$",
			r"$\bar{\beta}$",
			r"$\rho_{\beta}$",
			r"$\sigma_{\beta}$",
			r"$\delta$",
			r"$e^{mcq}$",
			r"$e^{sim}$",
			r"$e^{saq}$",
			r"$e^{book}$",
			r"$d$",
			r"$\gamma^{mcq}$",
			r"$\gamma^{saq}$",
			r"$\gamma^{book}$",
			r"$1/(1-\rho)$",
			r"$\vartheta$",
			r"$\lambda^E$",
			r"$\xi$",
			r"$\sigma_{\varepsilon^\xi{^*}}$",
			r"$\xi^*$",
			r"$\sigma_{\varepsilon^\xi}$"
				]

		make_tables(param_all, param_stds, param_names_new, param_names_old, 
			table_row_names, table_row_symbols, compile=True)


		# Get p values
		tstat_1 = (param_all[7] - param_all[5])/np.sqrt((param_stds[7]**2)/350 + (param_stds[5]**2)/350)
		tstat_2 = (param_all[7] - param_all[3])/np.sqrt((param_stds[7]**2)/350 + (param_stds[3]**2)/350)
		pval_1 = stats.t.sf(np.abs(tstat_1), 350-1)*2
		pval_2 = stats.t.sf(np.abs(tstat_2), 350-1)*2

		np.save(scr_path+'/'+ 'pval_1.npy',pval_1)
		np.save(scr_path+'/'+ 'pval_2.npy',pval_2)

		print(pval_1)




		


