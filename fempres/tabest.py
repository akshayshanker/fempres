import dill as pickle
from tabulate import tabulate   
import numpy as np
import csv

mod_name = 'Preliminary_all_v5/tau_20'
 
dict_means = pickle.load(open("/scratch/pv33/edu_model_temp/{}/latest_sampmom.smms".format(mod_name),"rb"))
gamma = pickle.load(open("/scratch/pv33/edu_model_temp/{}/gamma_XEM.smms".format(mod_name),"rb")) 
s_star = pickle.load(open("/scratch/pv33/edu_model_temp/{}/S_star.smms".format(mod_name),"rb")) 
settings_folder = 'settings'

param_random_bounds = {}


with open('{}/random_param_bounds.csv'\
	.format(settings_folder), newline='') as pscfile:
	reader_ran = csv.DictReader(pscfile)
	for row in reader_ran:
		param_random_bounds[row['parameter']] = np.float64([row['LB'],\
			row['UB']])


results = {}

for i,key in zip(np.arange(len(dict_means[0])),param_random_bounds.keys()):
	results[key]  = dict_means[0][i], np.sqrt(dict_means[1][i,i])


headers = ["Parameter", "Estimate", "Std. Err."]



print(tabulate([(k,) + v for k,v in results.items()], headers = headers))


mod_name = 'Preliminary_all_v5/tau_30'
 
dict_means = pickle.load(open("/scratch/pv33/edu_model_temp/{}/latest_sampmom.smms".format(mod_name),"rb"))
gamma = pickle.load(open("/scratch/pv33/edu_model_temp/{}/gamma_XEM.smms".format(mod_name),"rb")) 
s_star = pickle.load(open("/scratch/pv33/edu_model_temp/{}/S_star.smms".format(mod_name),"rb")) 

settings_folder = 'settings'

param_random_bounds = {}


with open('{}/random_param_bounds.csv'\
	.format(settings_folder), newline='') as pscfile:
	reader_ran = csv.DictReader(pscfile)
	for row in reader_ran:
		param_random_bounds[row['parameter']] = np.float64([row['LB'],\
			row['UB']])


results = {}

for i,key in zip(np.arange(len(dict_means[0])),param_random_bounds.keys()):
	results[key]  = dict_means[0][i], np.sqrt(dict_means[1][i,i])


headers = ["Parameter", "Estimate", "Std. Err."]



print(tabulate([(k,) + v for k,v in results.items()], headers = headers))


