# Import packages
import yaml
import gc
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
	ID = np.random.randint(0,9999)
	plot_path = "plots/plot_test_{}/".format(ID)
	Path(plot_path).mkdir(parents=True, exist_ok=True)
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

			#ax[j].set_xlabel('Week')
			ax[j].set_title(group_id)
			#ax[j].set_ylim(ylim)
			ax[j].spines['top'].set_visible(False)
			ax[j].spines['right'].set_visible(False)
			#ax[j].legend(loc='upper left', ncol=2)
		
		handles, labels = ax[7].get_legend_handles_labels()
		fig.legend(handles, labels, loc='upper center', ncol=2)
		fig.tight_layout()
		#handles, labels = ax.get_legend_handles_labels()
		#fig.legend(handles, labels, loc='upper center', ncol=2)
		#fig.tight_layout()

		fig.savefig("plots/plot_test_{}/{}.png".format(ID, name + '_Jan18'), transparent=True)

	return 

##############################################################################
# BEGIN FUNCTIONS FOR TABLE CONVERSION
##############################################################################
#INPUT: Numpy array containing parameter estimates (for either all the means or all the S.Es)
#OUTPUT: List of lists with 8 elements in order of columns in the pdf
# eg. 0 = Male president, Undisclosed gender, Males
#     7 = Female president, disclosed gender, Females
def array_to_ordered_list(array, param_names_new, param_names_old):
	#Convert numpy array to a list of lists
	A_list = array.tolist()

	#Reorder parameter values according to param_names_new and store in new list of lists
	A_list_new = []
	dict_old = collections.OrderedDict(zip(param_names_old, A_list[0])) 
	dict_new = collections.OrderedDict.fromkeys(param_names_new)

	for lst in A_list:
		dict_old = collections.OrderedDict(zip(param_names_old, lst)) 
		dict_new = collections.OrderedDict.fromkeys(param_names_new)
		for key in dict_new.keys():
			dict_new[key] = dict_old[key]
		A_list_new.append(list(dict_new.values()))

	#Reorder elements of list to match male-female column orders in the table
	mf_order = [1, 0, 3, 2, 5, 4, 7, 6]
	A_list_new[:] = [A_list_new[i] for i in mf_order]
	return A_list_new

#INPUT: The means and standard error lists returned from "array_to_ordered_list"
#and the row/symbol name lists
#OUTPUT: Pandas data frame
def ordered_lists_to_df(mean_list, se_list, table_row_names, table_row_symbols, pres = "male"):
	#Stop long strings being cut off with ...
	pd.set_option('display.max_colwidth', None)

	#Interlace the mean/S.E estimates in same order as table
	param_list = []

	#Fill param_list with estimates from mean_list/se_list, indices a:b, b not included
	a = 0
	b = 4
	if pres == "female":
		a += 4
		b += 4
	
	for i in range(a, b):
		param_list.append(mean_list[i])
		param_list.append(se_list[i])

	#Combine the numbers with row names and symbols. 
	param_list.insert(0, table_row_symbols)
	param_list.insert(0, table_row_names)

	df = pd.DataFrame(param_list).transpose()
	return df

#Convert pandas dataframe to tex string, no headings, only the numeric part
def df_to_tex_basic(df): 
	#Convert df to tex code, 3 decimal places, no row numbering
	table_tex = df.to_latex(
	header = False,
	float_format="{:0.4f}".format, 
	column_format="lcccccccccc",
	escape = False, #stops reading of latex command symbols as text (eg. $ -> \$, \ -> \backslash)
	index = False)
	return table_tex

#Add tabular environment/landscape to table_tex string
def tex_add_environment(table_tex, pres = "male"):
	#Add code before/after tabular environment
	table_env_start = [
		r"\begin{landscape}",
		r"\begin{table}[htbp]",
		r"\caption{\textsc{Parameter estimates}}\label{table:estimates_male_pres}",
		r"\centering"
	]
	if pres == "female":
		table_env_start[2] = r"\caption{\textsc{Parameter estimates}}\label{table:estimates_female_pres}"
	table_env_end = [
		r"\end{table}",
		r"\end{landscape}"
	]
	table_tex = "\n".join(table_env_start) + \
		"\n" + \
		table_tex + \
		"\n".join(table_env_end)
	table_tex = table_tex.replace(r"\bottomrule", r"\hline" + "\n" + r"\hline")
	return table_tex

#Add multi-level column headings to table tex string
def tex_add_col_headings(table_tex, pres = "male"):
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
	if pres == "female":
		headings[-1] = r"\textit{\textbf{Panel B: Female President}} & & & & & & & & & \\"
	table_tex = table_tex.replace(r"\toprule", r"\hline" + "\n" + r"\hline" + "\n".join(headings) + "\n")
	return table_tex

#Add extra (multi-level) row headings:
def tex_add_row_headings(table_tex):
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
		table_tex = table_tex.replace(row_heading_locations[i], 
									  row_headings[i] + row_heading_locations[i], 1)
	return table_tex

#Wrapper function for complete transformation of df to tex table
def df_to_tex_complete(df, pres = "male"):
	table_tex = df_to_tex_basic(df)
	table_tex = tex_add_environment(table_tex, pres)
	table_tex = tex_add_col_headings(table_tex, pres)
	table_tex = tex_add_row_headings(table_tex)
	return table_tex

#Given arrays of mean/S.E. estimates, and new vs old order of parameters, makes two tables.
#If compile = True, then adds a preamble to tex so that output file can directly compile in LaTeX
def make_tables(mean_array, se_array, param_names_new, param_names_old, table_row_names, table_row_symbols, compile = False):
	#Convert arrays to lists ready to be processed into dataframes.
	mean_list = array_to_ordered_list(mean_array, param_names_new, param_names_old)
	se_list = array_to_ordered_list(se_array, param_names_new, param_names_old)

	#Convert mean_list and se_list to male/female president dataframes
	df_male_pres = ordered_lists_to_df(mean_list, se_list, table_row_names, table_row_symbols, pres = "male")
	df_female_pres = ordered_lists_to_df(mean_list, se_list, table_row_names, table_row_symbols, pres = "female")

	#Convert both dataframes to tex code
	tex_male_pres = df_to_tex_complete(df_male_pres, pres = "male")
	tex_female_pres = df_to_tex_complete(df_female_pres, pres = "female")

	tex_code_final = tex_male_pres + "\n\n\n" + tex_female_pres 
	if compile:
		preamble = [
			r"\documentclass[12pt]{article}",
			r"\usepackage{geometry}",
			r"\geometry{verbose,letterpaper,tmargin=1in,bmargin=1in ,lmargin=1in,rmargin=1in,headheight=0in,headsep=0in,footskip=1.5\baselineskip}",
			r"\usepackage{array}",
			r"\usepackage{booktabs}",
			r"\usepackage{caption}",
			r"\usepackage{multirow}",
			r"\usepackage{pdflscape}",
			r"\begin{document}",
			r"\restoregeometry",
			r"\newgeometry{left=1cm, right=2cm, top=1cm, bottom=0.5cm}",
		]
		end = r"\end{document}"
		tex_code_final = "\n".join(preamble) + tex_code_final + "\n" + end
	
	#Print tex to the file table.tex. File will be created if it does not exist.
	f = open("table.tex", "w+")
	f.write(tex_code_final)
	f.close()
	return

##############################################################################
# END FUNCTIONS FOR TABLE CONVERSION
##############################################################################


if __name__ == "__main__":

	from mpi4py import MPI as MPI4py
	MPI4py.pickle.__init__(pickle.dumps, pickle.loads)
	from pathlib import Path

	# Make MPI world 
	world = MPI4py.COMM_WORLD

	# Name of the model that is estimated
	estimation_name = 'Preliminary_all_v1'
	
	# Folder for settings in home and declare scratch path
	settings_folder = 'settings/'
	scr_path = '/scratch/pv33/edu_model_temp/' + estimation_name
	Path(scr_path).mkdir(parents=True, exist_ok=True)

	# Load the data and sort and map the moments 
	moments_data = pd.read_csv('{}moments_clean.csv'\
					.format(settings_folder))
	moments_data_mapped = edumodel.map_moments(moments_data)

	# Assign model tau group according to each processor according to its rank
	model_name = list(moments_data_mapped.keys())[world.rank]
	#model_name = 'tau_01'

	# Load generic model settings 
	with open("{}settings.yml".format(settings_folder), "r") as stream:
		edu_config = yaml.safe_load(stream)

	# Load estimated moments dor tau group assigned to processor 
	sampmom = pickle.load(open(scr_path + "/{}/latest_sampmom.smms"\
						.format(model_name),"rb"))


	# To run the model with the latest estimated moments, set the covariance
	# of the paramter distribution to zero so the mean is drawn 

	#param_cov = np.zeros(np.shape(sampmom[1]))
	param_means = sampmom[0]
	#param_cov = sampmom[1]
	param_cov = np.zeros(np.shape(sampmom[1]))

	#param_means[2]  = 2
	#param_means[2]  = 1.2
	#param_means[0]  = .86
	#param_means[5]  = .8
	#param_means[6]  = 4.9
	#param_means[7]  = .33
	#param_means[13]  = 1.5


	# Blocked out code to adjust previous estimate values and number of params
	#param_means = np.append(param_means,.4)
	#param_cov = np.append(param_cov, [np.ones(len(param_means)-1)], 0)
	#param_cov = np.append(param_cov, np.transpose([np.ones(len(param_means))]), 1)

	#sampmom[0] = param_means
	#sampmom[1] = param_cov

	#pickle.dump(sampmom,open(scr_path + "/{}/latest_sampmom.smms"\
	#					.format(model_name),"wb"))

	# Load simulation shock draws
	# Akshay to change this later so it is drawn from the HD
	#U = np.random.rand(edu_config['baseline_lite']['parameters']['N'],\
	#					edu_config['baseline_lite']['parameters']['T'])
	#U_z = np.random.rand(edu_config['baseline_lite']['parameters']['N'],2)

	U = np.load(scr_path+'/'+ 'U.npy')
	U_z = np.load(scr_path+'/'+ 'U_z.npy')



	# Run model (note, param_random_bounds serves no purpose here, it should be deprecated)
	param_random_bounds = {}
	with open('{}random_param_bounds.csv'.format(settings_folder), newline='') as pscfile:
		reader_ran = csv.DictReader(pscfile)
		for row in reader_ran:
			param_random_bounds[row['parameter']] = np.float64([row['LB'],
																row['UB']])

	# Run the model and generate the moments on each processor
	edu_model = edumodel.EduModelParams(model_name,
							edu_config[model_name],
							U,
							U_z,
							random_draw = True,
							uniform = False,
							param_random_means = param_means, 
							param_random_cov = param_cov, 
							random_bounds = param_random_bounds)

	moments_sim = generate_study_pols(edu_model.og)


	moments_sim_all = world.gather(moments_sim, root = 0)
	param_all = world.gather(param_means, root = 0)
	param_all_cov = world.gather(sampmom[1], root = 0)

	# Now gather all the moments to the master processor (rank 0)
	if world.rank == 0:
		
		moments_sim_all  =  np.array(moments_sim_all)
		param_all = np.array(param_all)
		param_all_cov = np.array(param_all_cov) 

		moments_data_all = np.empty(np.shape(moments_sim_all))

		std_errs = np.empty((8, len(param_all[0,:])))

		for i in range(len(std_errs)):
			std_errs[i,:] = np.sqrt(np.diag(param_all_cov[i]))


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
						'Knowledge accumulation',\
						'av_game_session_hours_cumul',\
						'av_ebook_session_hours_cumul',\
						'av_mcq_session_hours_cumul',\
						'av_saq_session_hours_cumul',\
						'av_player_happy_deploym_cumul',\
						'av_mcq_attempt_nonrev_cumul',\
						'av_sa_attempt_cumul', \
						'av_totebook_pageviews_cumul',\
						'sd_final',\
						'sd_mark',\
						'sd_markw13_exp1',\
						'Coursework grade',\
						'sd_game_session_hours_cumul',\
						'sd_ebook_session_hours_cumul',\
						'sd_mcq_session_hours_cumul',\
						'sd_saq_session_hours_cumul',\
						'sd_player_happy_deploym_cumul',\
						'sd_mcq_attempt_nonrev_cumul',\
						'sd_sa_attempt_cumul', \
						'sd_totebook_pageviews_cumul',\
						'acgame_session_hours',\
						'acebook_session_hours',\
						'acmcq_session_hours',\
						'acsaq_session_hours',\
						'actotebook_pageviews',\
						'acmcq_Cattempt_nonrev',\
						'cmcsaq_session_hours',\
						'cgsaq_session_hours',\
						'cgmcq_session_hours',\
						'cesaq_session_hours',\
						'cemcq_session_hours',\
						'ceg_session_hours',\
						'co_fgame_session_hours_cumul',\
						'co_febook_session_hours_cumul',\
						'co_fmcq_session_hours_cumul',
						'co_fgame_session_hours_cumul',\
						'co_fsa_attempt_cumul',\
						'co_ftotebook_pageviews_cumul',\
						'co_fmcq_attempt_nonrev_cumul',\
						'co_fsa_attempt_cumul',\
						'co_gam',\
						'co_eboo',\
						'co_mc',\
						'co_sa',\
						'co_fatar_ii']

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
				'd',
				'alpha',
				'beta_bar',
				'rho_beta',
				'sigma_beta',
				'delta',
				'gamma_3',
				'gamma_1',
				'gamma_2',
				'sigma_M',
				'kappa_3',
				'kappa_1',
				'kappa_2',
				'kappa_4',
				#Gamma symbols below do not match table
				'zeta_star',
				'sigma_zeta',
				'zeta_hstar',
				'sigma_hzeta',
				'lambda_E',
				'varphi_sim'
			]
		
		#Row names for table
		table_row_names = [
			"Depreciation d",
			"Course grade utility weight",
			#"Hyperbolic discount factor",
			"\hspace{0.4cm}Discount factor mean",
			"\hspace{0.4cm}Discount factor persistence",
			"\hspace{0.4cm}Discount factor std. deviation",
			"Exponential discount factor",
			#"Study effectiveness for knowledge creation",
			"\hspace{0.4cm}Time solving MCQs",
			"\hspace{0.4cm}Time answering SAQs",
			"\hspace{0.4cm}Time studying the textbook",
			"\hspace{0.4cm}Overall study elasticity",
			"\hspace{0.4cm}Previous knowledge share in CES nest",
			"\hspace{0.4cm}Study elasticity of substitution",
			"\hspace{0.4cm}SAQ and knowledge stock elasticity of substitution",
			"\hspace{0.4cm}Effort cost of MCQ ",
			"\hspace{0.4cm}Effort cost of SAQ",
			"\hspace{0.4cm}Effort cost of book",
			"\hspace{0.4cm}Effort cost of sim",
			#"Final exam ability",
			"\hspace{0.4cm}Real exam ability mean",
			"\hspace{0.4cm}Real exam ability std. deviation",
			"\hspace{0.4cm}Perceived exam ability mean",
			"\hspace{0.4cm}Perceived exam ability std. deviation",
			"\hspace{0.4cm}Exam difficulty parameter",
			"\hspace{0.4cm}Knowledge to study output parameter"
		]

		#Latex symbols for table row names, prefix r for raw string, or else causes unicode error
		table_row_symbols = [
			r"$d$",
			r"$\alpha$",
			r"$\bar{\beta}$",
			r"$\rho_{\beta}$",
			r"$\sigma_{\beta}$",
			r"$\delta$",
			r"$\gamma^{MCQ}$",
			r"$\gamma^{SAQ}$",
			r"$\gamma^{book}$",
			r"$\sigma^{M}$",
			r"$\kappa^{MCQ}$",
			r"$\kappa^{SAQ}$",
			r"$\kappa^{book}$",
			r"$\kappa^{sim}$",
			r"$\xi$",
			r"$\sigma_{\varepsilon^\xi}$",
			r"$\xi^*$",
			r"$\sigma_{\varepsilon^\xi{^*}}$",
			r"$\lambda^E$",
			r"$\vartheta$"
		]
		#list_params = param_random_bounds.keys()
		#NOTE: CHANGE mean_array and se_array to the names of your two 8 x n arrays 
		#which contain all mean estimates and all S.E estimates respectively.

		#group_list = ['Draw_1']

		group_list = list(moments_data_mapped.keys())

		#make_tables(param_all, std_errs, param_names_new, param_names_old, 
		#			table_row_names, table_row_symbols, compile=True)

		plot_results(moments_sim_all,moments_data_all, list_moments, group_list)