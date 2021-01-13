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
	moments_data_mapped = map_moments(moments_data)

	# Assign model tau group according to each core according to processor rank
	model_name = list(moments_data_mapped.keys())[world.rank]

	# Load model settings 
	with open("{}settings.yml".format(settings_folder), "r") as stream:
		edu_config = yaml.safe_load(stream)
