"""
This module contains the Education model class
 
"""
# Import packages
from util.randparam import rand_p_generator
from util.edu_model_functions import edumodel_function_factory
import numpy as np
from numba import njit, prange, jit
import time
import random
import string
import dill as pickle
from sklearn.utils.extmath import cartesian
from quantecon import tauchen
from interpolation.splines import UCGrid, CGrid, nodes, eval_linear
import copy
import pandas as pd

import warnings
warnings.filterwarnings('ignore')

class EduModel:

    """
    Generates Education Model class
    with all params, functions and grids 
    for a paramterised Education Model 

    """

    def __init__(self,
                 config,  # Settings dictionary
                 U, 
                 U_z,
                 param_id,# ID for parameter value set
                 mod_name # Name of model (group_ID)
                 ):
        
        self.parameters = config['parameters']
        self.theta = config['theta']
        self.config = config
        self.__dict__.update(config['parameters'])

        self.u_grade, self.fin_exam_grade,\
            self.S_effort_to_IM, self.u_l, self.correct_mcq_rate\
             = edumodel_function_factory(config['parameters'],
                                            config['theta'],
                                            config['share_saq'],
                                            config['share_eb'],
                                            config['share_mcq'],
                                            config['share_hap'],)

        # 1D grids
        self.M  = np.linspace(self.M_min, self.M_max, self.grid_size_M)
        self.Mh = np.linspace(self.Mh_min, self.Mh_max, self.grid_size_Mh)
        self.U = U
        self.U_z = U_z

        # Create 2D endogenous state-space grid
        self.MM = UCGrid((self.M_min, self.M_max, self.grid_size_M),
                         (self.Mh_min, self.Mh_max, self.grid_size_Mh))

        self.MM_gp = nodes(self.MM)

        # Individual stochastic processes 

        lnr_beta_bar = np.log(1/self.beta_star - 1)
        y_bar = lnr_beta_bar*(1-self.rho_beta)
        self.lnr_beta_mc = tauchen(self.rho_beta,
                                   self.sigma_beta,
                                   b = y_bar,
                                   n = self.grid_size_beta)

        self.beta_hat = 1/(1+np.exp(self.lnr_beta_mc.state_values))

        self.P_beta = self.lnr_beta_mc.P
        self.beta_stat = self.lnr_beta_mc.stationary_distributions[0]


        self.zeta_mc = tauchen(0,
                                   self.sigma_zeta,
                                   b = self.zeta_star,
                                   n = self.grid_size_zeta)
        self.zeta_hat = np.exp(self.zeta_mc.state_values)
        self.P_zeta = self.zeta_mc.P
        self.zeta_stat = self.zeta_mc.stationary_distributions[0]


        # Perceived actual exam grading ability shock (normally distributed)
        # means and sd and rho is for the (e_c) process 

        self.zetah_mc = tauchen(0,
                                   self.sigma_hzeta,
                                   b=self.zeta_hstar,
                                   n=self.grid_size_zeta)
        self.zeta_hhat = np.exp(self.zetah_mc.state_values)
        self.P_zetah = self.zetah_mc.P
        self.zetah_stat = self.zetah_mc.stationary_distributions[0]


        # Study ability shock (log-normally distributed)
        # means and sd and rho is for the log(e_s) process 

        self.es_mc = tauchen(self.rho_es,
                                   self.sigma_es,
                                   b = self.es_star,
                                   n = self.grid_size_es)
        self.es_hat = np.exp(self.es_mc.state_values)/np.dot(np.exp(self.es_mc.state_values), self.es_mc.stationary_distributions[0])
        self.P_es = self.es_mc.P
        self.es_stat = self.es_mc.stationary_distributions[0]


        # Combine all shock values into cartesian product 

        #self.Q_shocks = cartesian([self.beta_hat,self.es_hat])
        self.Q_shocks_ind = cartesian([np.arange(len(self.beta_hat)),np.arange(len(self.es_hat))])

        # Build joint beta and es shock transition matrix
        # EBA_P[i,j, :] gives joint PMF of beta and es_hat for t+1 for beta_ind = j and es_ind = i at t

        self.EBA_P = np.zeros((len(self.beta_hat), len(self.es_hat), int(len(self.beta_hat)*len(self.es_hat))))

        sizeEBA =   int(len(self.beta_hat)*
                                len(self.es_hat))

        #self.EBA_P2 = self.EBA_P.reshape((sizeEBA, sizeEBA))


        for j in self.Q_shocks_ind:

            EBA_P_temp = cartesian([self.P_beta[j[0]],
                                    self.P_es[j[1]]])

            self.EBA_P[j[0], j[1], :]\
                = EBA_P_temp[:, 0]*EBA_P_temp[:, 1]

        self.EBA_P2 = self.EBA_P.reshape((sizeEBA, sizeEBA))

        # Generate final period T expected continuation value

        VF_UC = np.zeros((len(self.zeta_hhat)*len(self.M)*len(self.Mh)))

        T_state_all = cartesian([self.zeta_hhat, self.M, self.Mh])

        for i in range(len(T_state_all)):
            zetah = T_state_all[i,0]
            m = T_state_all[i,1]
            mh = T_state_all[i,2]
            exam_mark = self.fin_exam_grade(zetah,m)
            #print(exam_mark)
            utilT = self.u_grade(exam_mark, mh)

            VF_UC[i] = utilT

        # Condition this back
        VF_UC_1 = VF_UC.reshape((len(self.zeta_hhat),len(self.M),len(self.Mh)))
        VF_UC_2 = VF_UC_1.transpose((1,2,0))

        self.VT = np.dot(VF_UC_2,self.P_zetah[0])

def map_moments(moments_data):

    """ Takes data moments, where each row corrsponds to a group x week
        group referes to M x F x RCT group

        The moments_data df should have group id column
    """

    # Get the group names 
    group_list = moments_data['group'].unique()

    # Make a function to turn the group names into a list
    def gen_group_list(somelist):
        return {x: {} for x in somelist}

    moments_grouped_sorted = gen_group_list(group_list)

    # List the moments names (col names in the data frame)
    # and order them in the same order as the word doc 
    # the order of the list should be the same as in sim
    # moments mapped in gen_moments
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
                    'av_totebook_pageviews_cumul',\
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


    # For each group, create an empty array of sorted moments 
    for keys in moments_grouped_sorted:
        moments_grouped_sorted[keys]['data_moments'] = np.empty((11,49))

    # Map the moments to the array with cols as they are ordered
    # in list_moments for each group
    for i in range(len(list_moments)):
        for key in moments_grouped_sorted:
            moments_data_for_gr = moments_data[moments_data['group'] == key]
            moments_grouped_sorted[key]['data_moments'][:,i] = moments_data_for_gr[list_moments[i]]
    # Return the sorted moments for each group
    return moments_grouped_sorted


class EduModelParams:
    """ Parameterises class for the EduModelk
            Instance of EduModelk contains 
            parameter list, a paramterized EduModel

            Instance of Parameter class identified by
            unique paramter ID, param_id

            Todo
            ----

    """

    def __init__(self,
                 mod_name, # name of model 
                 param_dict, 
                 U,        # beta shocks uniform draw
                 U_z,       # zeta shocks uniform draw
                 random_draw = False,   # True iff parameters generated randomly
                 random_bounds = None,  # Parameter bounds for random draws 
                 param_random_means = None,  # mean of random param distribution
                 param_random_cov = None, # cov of random param distribution
                 uniform = False):  # True iff  draw is uniform

        # Generate a random ID for this param draw 
        self.param_id = ''.join(random.choices(
            string.ascii_uppercase + string.digits, k=6))+'_'+time.strftime("%Y%m%d-%H%M%S") + '_'+mod_name
        self.mod_name = mod_name

        # If random draw is false, assign paramters from paramter dictionary pre-sets 
        if random_draw == False:
            param_deterministic = param_dict['parameters']
            parameters_draws = rand_p_generator(param_deterministic,
                                    random_bounds,
                                    deterministic = 1,
                                    initial = uniform)

            self.og = EduModel(param_dict, U, U_z, self.param_id, mod_name=mod_name)
            self.parameters = parameters_draws
            self.parameters['param_id'] = self.param_id

        if random_draw == True:
            param_deterministic = param_dict['parameters']

            parameters_draws = rand_p_generator(param_deterministic,
                                                random_bounds,
                                                deterministic = 0,
                                                initial = uniform,
                                                param_random_means = param_random_means,
                                                param_random_cov = param_random_cov)
            param_dict_new = copy.copy(param_dict)
            param_dict_new['parameters'] = parameters_draws

            self.og = EduModel(param_dict_new, U, U_z, self.param_id, mod_name=mod_name)

            self.parameters = parameters_draws
            self.parameters['param_id'] = self.param_id



if __name__ == "__main__":

    # Generate  instance of LifeCycleParams class with
    # an instance of DB LifeCycle model and DC LS model
    import yaml
    import csv
    import pandas as pd
    from solve_policies.studysolver import generate_study_pols
    from pathlib import Path

    # Folder contains settings (grid sizes and non-random params)
    # and random param bounds
    settings = 'settings/'
    # Name of model
    model_name = 'tau01'
    estimation_name = 'Preliminary_all_v5'
    # Path for scratch folder (will contain latest estimated means)
    scr_path = "/scratch/pv33/edu_model_temp/"

    with open("{}settings.yml".format(settings), "r") as stream:
        edu_config = yaml.safe_load(stream)

    param_random_bounds = {}
    with open('{}random_param_bounds.csv'.format(settings), newline='') as pscfile:
        reader_ran = csv.DictReader(pscfile)
        for row in reader_ran:
            param_random_bounds[row['parameter']] = np.float64([row['LB'],
                                                                row['UB']])

    sampmom = pickle.load(open("/scratch/pv33/edu_model_temp/{}/latest_sampmom.smms".format('test_CES3/tau_01'),"rb"))

    # Generate random points for beta and es
    U = np.random.rand(edu_config['baseline_lite']['parameters']['S'],
                        edu_config['baseline_lite']['parameters']['N'],\
                        edu_config['baseline_lite']['parameters']['T'],2)

    # Generate random points for ability and percieved ability 
    U_z = np.random.rand(edu_config['baseline_lite']['parameters']['S'],
                        edu_config['baseline_lite']['parameters']['N'],2)


    #scr_path2 = '/scratch/pv33/edu_model_temp/' + '/' + estimation_name
    #Path(scr_path2).mkdir(parents=True, exist_ok=True)

    #np.save(scr_path2+'/'+ 'U.npy',U)
    #np.save(scr_path2+'/'+ 'U_z.npy',U_z)

    moments_data = pd.read_csv('{}moments_clean.csv'\
                    .format(settings))

    moments_grouped_sorted  = map_moments(moments_data)

    edu_model = EduModelParams('test',
                                edu_config['tau_01'],
                                U,
                                U_z,
                                random_draw = True,
                                uniform = True,
                                param_random_means = sampmom[0], 
                                param_random_cov = np.zeros(np.shape(sampmom[1])), 
                                random_bounds = param_random_bounds)

    #moments_sim = generate_study_pols(edu_model.og)


