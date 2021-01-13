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

def plot_results():
	return 

if __name__ == "__main__":

	from mpi4py import MPI as MPI4py
	MPI4py.pickle.__init__(pickle.dumps, pickle.loads)
	from pathlib import Path

	# Make MPI world 
	world = MPI4py.COMM_WORLD

	# Name of the model that is estimated
	estimation_name = 'test'
	
	# Folder for settings in home and declare scratch path
	settings_folder = 'settings/'
	scr_path = '/scratch/pv33/edu_model_temp/' + '/' + estimation_name
	Path(scr_path).mkdir(parents=True, exist_ok=True)

	# Load the data and sort and map the moments 
	moments_data = pd.read_csv('{}moments_clean.csv'\
					.format(settings_folder))
	moments_data_mapped = edumodel.map_moments(moments_data)

	# Assign model tau group according to each processor according to its rank
	model_name = list(moments_data_mapped.keys())[world.rank]

	# Load generic model settings 
	with open("{}settings.yml".format(settings_folder), "r") as stream:
		edu_config = yaml.safe_load(stream)

	# Load estimated moments dor tau group assigned to processor 
	sampmom = pickle.load(open(scr_path + "/{}/latest_sampmom.smms"\
						.format(model_name),"rb"))

	# To run the model with the latest estimated moments, set the covariance
	# of the paramter distribution to zero so the mean is drawn 

	param_cov = np.zeros(np.shape(sampmom[1]))
	param_means = sampmom[0]

	# Load simulation shock draws
	# Akshay to change this later so it is drawn from the HD
	U = np.random.rand(edu_config['baseline_lite']['parameters']['N'],\
						edu_config['baseline_lite']['parameters']['T'])
	U_z = np.random.rand(edu_config['baseline_lite']['parameters']['N'],2)


	# Run model (note, param_random_bounds serves no purpose here, it should be deprecated)
	param_random_bounds = {}
	with open('{}random_param_bounds.csv'.format(settings_folder), newline='') as pscfile:
		reader_ran = csv.DictReader(pscfile)
		for row in reader_ran:
			param_random_bounds[row['parameter']] = np.float64([row['LB'],
																row['UB']])

	# Run the model and generate the moments on each processor
	edu_model = edumodel.EduModelParams('test',
							edu_config['tau_00'],
							U,
							U_z,
							random_draw = True,
							uniform = True,
							param_random_means = sampmom[0], 
							param_random_cov = sampmom[1], 
							random_bounds = param_random_bounds)

	moments_sim = generate_study_pols(edu_model.og)

	moments_sim_all = world.gather(moments_sim, root = 0)

	# Now gather all the moments to the master processor (rank 0)
	if world.rank == 0:
		
		moments_sim_all =  np.array(moments_sim_all)

		np.save(moments_all,moments_sim_all )

		# Each item in moment list is a 11 x 36 array
		# The rows are are 

		list_moments = ['Final exam mark',\
						'Overall course grade',\
						'Expected final exam mark',\
						'Expected final exam mark (other)',\
						'Game session hours',\
						'E-book hours',\
						'MCQ session hours',\
						'SAQ session hours',\
						'av_player_happy_deploym_cumul',\
						'av_mcq_attempt_nonrev_cumul',\
						'SAQ attempts', \
						'av_mcq_Cshare_nonrev_cumul',\
						'SD final exam mark',\
						'SD Overall course grade',\
						'SD Expected final exam mark',\
						'SD Expected final exam mark (other)',\
						'SD Game session hours',\
						'SD E-book hours',\
						'SD MCQ session hours',\
						'SD SAQ session hours',\
						'sd_player_happy_deploym_cumul',\
						'sd_mcq_attempt_nonrev_cumul',\
						'SD SAQ attempts', \
						'sd_mcq_Cshare_nonrev_cumul',\
						'AC game session hours',\
						'AC e-book session hours',\
						'AC MCQ session hours',\
						'AC SAQ session hours',\
						'acmcq_Cshare_nonrev',\
						'Corr. MCQ and SAQ hours',\
						'Corr. game and SAQ hours',\
						'Corr. game and MCQ hours',\
						'Corr. e-book and SAQ hours',\
						'Corr. e-book. and MCQ hours',\
						'Corr. e-book and game hours',\
						'Corr. ATAR and final exam mark']












