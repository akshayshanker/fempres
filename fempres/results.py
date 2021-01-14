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
import matplotlib.pyplot as plt
import numpy as np
import collections
import pandas as pd


# Education model modules
import edumodel 
from solve_policies.studysolver import generate_study_pols

def plot_results(moments_sim_all,
				 moments_data_all,
				 variable_list,
				 group_list):

	line_names =['data', 'sim']
	linestyles=["-","-"]
	col_dict = {'data': 'black', 'sim':'gray'}
	markers=['x', 'o']

	# loop through each of the variables in the list
	for i, name in zip(np.arange(len(variable_list)),variable_list):
		# create a plot for this variable
		fig, ax = plt.subplots(4,2)
		ax = ax.flatten()

		#loop through each of the group
		for j, group_id in zip(np.arange(len(group_list)),group_list):
			xs = np.arange(11)
			ys = moments_data_all[j,:,i]
			p = ax[j].plot(xs, ys, marker=markers[0], color=col_dict['data'], linestyle=linestyles[0],
						label=line_names[0], linewidth=2)
			ys = moments_sim_all[j,:,i]
			p = ax[j].plot(xs, ys, marker=markers[1], color=col_dict['sim'], linestyle=linestyles[1],
			label=line_names[1], linewidth=2)

			ax[j].set_xlabel('Week')
			ax[j].set_title(group_id)
			#ax[j].set_ylim(ylim)
			ax[j].spines['top'].set_visible(False)
			ax[j].spines['right'].set_visible(False)
			#ax[j].legend(loc='upper left', ncol=2)
		
		handles, labels = ax[7].get_legend_handles_labels()
		fig.legend(handles, labels, loc='upper center', ncol=2)
		fig.savefig("plot_test/{}.png".format(name +'_'+group_id), transparent=True)


	return 

def make_table(params, param_names_old, param_names_new):

	#Row names for the table, headers (i.e. no values) commented out
	table_row_names = [
	    "Course grade utility weight",
	    #"Exponential discount factor",
	    "\hspace{0.4cm}Discount factor mean",
	    "\hspace{0.4cm}Discount factor persistence",
	    "\hspace{0.4cm}Discount factor std. deviation",
	    "Hyperbolic discount factor",
	    #"Study effectiveness for knowledge creation",
	    "\hspace{0.4cm}Time solving MCQs",
	    "\hspace{0.4cm}Time earning happiness units",
	    "\hspace{0.4cm}Time answering SAQs",
	    "\hspace{0.4cm}Time studying the textbook",
	    "\hspace{0.4cm}Time solving MCQs \& earning happiness",
	    "\hspace{0.4cm}Time solving MCQs \& SAQ",
	    "\hspace{0.4cm}Time solving MCQs \& studying textbook",
	    "\hspace{0.4cm}Time earning happiness \& solving SAQs",
	    "\hspace{0.4cm}Time earning happiness \& studying textbook",
	    "\hspace{0.4cm}Time solving SAQs \& studying textbook",
	    #"Final exam ability",
	    "\hspace{0.4cm}Real exam ability mean",
	    "\hspace{0.4cm}Real exam ability std. deviation",
	    "\hspace{0.4cm}Perceived exam ability mean",
	    "\hspace{0.4cm}Perceived exam ability std. deviation",
	    "\hspace{0.4cm}Exam difficulty parameter"
	    "\hspace{0.4cm}MCQ question difficulty parameter"
	]

	#Latex symbols for table row names, prefix r for raw string, or else causes unicode error
	table_row_symbols = [
	    r"$\alpha$",
	    r"$\bar{\beta}$",
	    r"$\rho_{\beta}$",
	    r"$\sigma_{\beta}$",
	    r"$\delta$",

	    r"$\gamma^{mcq}$",
	    r"$\gamma^{sim}$",
	    r"$\gamma^{saq}$",
	    r"$\gamma^{book}$",
	    r"$\gamma^{mcq\&sim}$",
	    r"$\gamma^{mcq\&saq}$",
	    r"$\gamma^{mcq\&book}$",
	    r"$\gamma^{sim\&saq}$",
	    r"$\gamma^{sim\&book}$",
	    r"$\gamma^{saq\&book}$",

	    r"$\xi$",
	    r"$\sigma_{\varepsilon^\xi}$",
	    r"$\xi^*$",
	    r"$\sigma_{\varepsilon^\xi{^*}}$",
	    r"$\lambda^E$"
	    r"$\varphi$"
	]

	#Convert numpy array to a list of lists
	A_list = params.tolist()

	#Reorder parameter values according to param_names_new and store in new list of lists
	A_list_new = []
	for lst in A_list:
	    dict_old = collections.OrderedDict(zip(param_names_old, lst)) 
	    dict_new = collections.OrderedDict.fromkeys(param_names_new)
	    for key in dict_new.keys():
	        dict_new[key] = dict_old[key]
	    A_list_new.append(list(dict_new.values()))

	#Reorder elements of list to match male-female column orders in the table (means only)
	mf_order = [1, 0, 3, 2, 5, 4, 7, 6]
	A_list_new[:] = [A_list_new[i] for i in mf_order]

	#Combine the numbers with row names and symbols. Note, because of mismatch between parameter list
	#given and parameter list in the pdf, there will be NaNs/Nones in the dataframe df.
	A_list_new.insert(0, table_row_symbols)
	A_list_new.insert(0, table_row_names)
	pd.set_option('display.max_colwidth', None) #Stop long strings being cut off with ...
	df = pd.DataFrame(A_list_new).transpose()

	#Convert list to tex code, 3 decimal places, no row numbering
	ncols = len(A_list_new)
	col_format = "l" + ncols*"c" #note the extra centered cell for formatting purposes

	table_tex = df.to_latex(
	    header = False,
	    float_format="{:0.3f}".format, 
	    column_format=col_format,
	    escape = False, #stops reading of latex command symbols as text (eg. $ -> \$, \ -> \backslash)
	    index = False)

	#Add code before/after tabular environment
	table_env_start = [
	    r"\begin{table}[htbp]",
	    r"\caption{\textsc{Parameter estimates}}\label{table:estimates}",
	    r"\centering"
	]
	table_env_end = r"\end{table}"
	table_tex = "\n".join(table_env_start) + "\n" + table_tex + table_env_end
	table_tex = table_tex.replace(r"\bottomrule", r"\hline" + "\n" + r"\hline")


	#Add code for multi-level column headings
	headings = [
	    r"& & \multicolumn{4}{c}{Undisclosed Gender} &\multicolumn{4}{c}{Disclosed Gender}\\",
	    r"\cmidrule(l{0.2cm}r{0.3cm}){3-6}\cmidrule(l{0.2cm}r{0.3cm}){7-10}",
	    r"& & \multicolumn{2}{c}{Males} &\multicolumn{2}{c}{Females} & \multicolumn{2}{c}{Males} &\multicolumn{2}{c}{Females}\\",
	    r"\cmidrule(l{0.2cm}r{0.3cm}){3-4}\cmidrule(l{0.2cm}r{0.3cm}){5-6}\cmidrule(l{0.2cm}r{0.3cm}){7-8} \cmidrule(l{0.2cm}r{0.3cm}){9-10}",
	    r"& & Estimates  & S.E.  & Estimates  & S.E. & Estimates  & S.E. & Estimates  & S.E. & \tabularnewline",
	    r" &  &  (1) & (2) & (3)  & (4) & (5) & (6) & (7) & (8) \\",
	    r"\hline",
	    r"\\",
	    r"\textit{\textbf{Panel A: Male President}} & & & & & & & & & \\",
	]
	table_tex = table_tex.replace(r"\toprule", r"\hline" + "\n" + r"\hline" + "\n".join(headings) + "\n")


	#Add multi-level row-headings
	row_headings = [
	    r"Exponential discount factor & & & & & & & & & \\" + "\n",
	    r"\hline" + "\n" + r"Study effectiveness for knowledge creation & & & & & & & & & \\" + "\n",
	    r"\hline" + "\n" +  r"Final exam ability & & & & & & & & & \\" + "\n"
	]
	row_heading_locations = [
	    "\hspace{0.4cm}Discount factor mean",
	    "\hspace{0.4cm}Time solving MCQs",
	    "\hspace{0.4cm}Real exam ability mean"
	]

	for i in range(0, 3):
	    table_tex = table_tex.replace(row_heading_locations[i], row_headings[i] + row_heading_locations[i], 1)

	print(table_tex)

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
							uniform = False,
							param_random_means = sampmom[0], 
							param_random_cov = sampmom[1], 
							random_bounds = param_random_bounds)

	moments_sim = generate_study_pols(edu_model.og)


	moments_sim_all = world.gather(moments_sim, root = 0)
	param_all = world.gather(param_means, root = 0)

	# Now gather all the moments to the master processor (rank 0)
	if world.rank == 0:
		
		moments_sim_all  =  np.array(moments_sim_all)
		param_all = np.array(param_all)
		moments_data_all = np.empty(np.shape(moments_sim_all))

		# re-format data moments so in the same format as sim moments, combined

		for i, keys in zip(np.arange(8),list(moments_data_mapped.keys())):
			moments_data_all[i] = moments_data_mapped[keys]['data_moments']

		# Generate new data-frames for plotting 

		#np.save('moments_all',moments_sim_all )
		#np.save('param_all', param_all)

		# Each item in moment list is a 11 x 36 array
		# The rows are are 

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
						'acmcq_Cattempt_nonrev',\
						'cmcsaq_session_hours',\
						'cgsaq_session_hours',\
						'cgmcq_session_hours',\
						'cesaq_session_hours',\
						'cemcq_session_hours',\
						'ceg_session_hours',\
						'c_atar_ii']


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
				    'gamma_4',
				    'gamma_12',
				    'gamma_21',
				    'gamma_34',
				    'gamma_13',
				    'gamma_22',
				    'gamma_23',
				    'gamma_33',
				    'gamma_11',
				    'd',
				    'varphi']

		param_names_new = [
			    'd',

			    'alpha',
			    'beta_bar',
			    'rho_beta',
			    'sigma_beta',
			    'delta',

			    'gamma_3',
			    'gamma_4',
			    'gamma_1',
			    'gamma_2',
			    'gamma_34',
			    'gamma_13',
			    'gamma_12',

			    #Gamma symbols below do not match table
			    'gamma_21',
			    'gamma_22',
			    'gamma_23',
			    'gamma_33',
			    'gamma_11',

			    'zeta_star',
			    'sigma_zeta',
			    'zeta_hstar',
			    'sigma_hzeta',
			    'lambda_E',
			    'varphi'
			]

		#list_params = param_random_bounds.keys()
		#make_table(param_all,param_names_old,param_names_new)
		plot_results(moments_sim_all,moments_data_all, list_moments, list(moments_data_mapped.keys()))














