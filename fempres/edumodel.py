
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


import warnings
warnings.filterwarnings('ignore')


class EduModel:

    """

    """

    def __init__(self,
                 config,  # Settings dictionary
                 param_id,# ID for parameter value set
                 mod_name # Name of model
                 ):
        
        self.parameters = config['parameters']
        self.__dict__.update(config['parameters'])
        self.u_grade, self.fin_exam_grade,self.cw_grade,\
            self.S_effort, self.u_l\
             = edumodel_function_factory(config['parameters'])


        # 1D grids
        self.M  = np.linspace(self.M_min, self.M_max, self.grid_size_M)
        self.Mh = np.linspace(self.Mh_min, self.Mh_max, self.grid_size_Mh)

        # Create 2D endogenous state-space grid
        self.MM = UCGrid((self.M_min, self.M_max, self.grid_size_M),
                         (self.Mh_min, self.Mh_max, self.grid_size_Mh))

        self.MM_gp = nodes(self.MM)

        # Individual stochastic processes 

        #  beta and delta processes
        #  Recall beta_star is the mean of beta the discount rate

        lnr_beta_bar = np.log(1/self.beta_star - 1)
        y_bar = lnr_beta_bar*(1-self.beta_star)
        self.lnr_beta_mc = tauchen(self.rho_beta,
                                   self.sigma_beta,
                                   b = y_bar,
                                   n = self.grid_size_beta)

        self.beta_hat = 1/(1+np.exp(self.lnr_beta_mc.state_values))

        self.P_beta = self.lnr_beta_mc.P
        self.beta_stat = self.lnr_beta_mc.stationary_distributions[0]


        lnr_delta_bar = np.log(1/self.deltas_star - 1)
        y_bar = lnr_delta_bar*(1-self.deltas_star)
        self.lnr_delta_mc = tauchen(self.rho_delta,
                                   self.sigma_delta,
                                   b=y_bar,
                                   n=self.grid_size_delta)
        
        self.delta_hat = 1/(1+np.exp(self.lnr_delta_mc.state_values))

        self.P_delta = self.lnr_delta_mc.P
        self.delta_stat = self.lnr_delta_mc.stationary_distributions[0]

        # leisure shock (normally distributed)

        self.xi_mc = tauchen(self.rho_xi,
                                   self.sigma_xi,
                                   b=self.xi_star,
                                   n=self.grid_size_xi)
        self.xi_hat = self.xi_mc.state_values
        self.P_xi = self.xi_mc.P
        self.xi_stat = self.xi_mc.stationary_distributions[0]

        # Study ability shock (log-normally distributed)
        # means and sd and rho is for the log(e_s) process 

        self.es_mc = tauchen(self.rho_es,
                                   self.sigma_es,
                                   b = self.es_star,
                                   n = self.grid_size_es)
        self.es_hat = np.exp(self.es_mc.state_values)
        self.P_es = self.es_mc.P
        self.es_stat = self.es_mc.stationary_distributions[0]

        # CW grading ability shock (normally distributed)
        # means and sd and rho is for the (e_c) process 

        self.ec_mc = tauchen(self.rho_ec,
                                   self.sigma_ec,
                                   b = self.ec_star,
                                   n = self.grid_size_ec)
        self.ec_hat = self.ec_mc.state_values
        self.P_ec = self.ec_mc.P
        self.ec_stat = self.ec_mc.stationary_distributions[0]

        # Actual exam grading ability shock (normally distributed)
        # means and sd and rho is for the (e_c) process 

        self.zeta_mc = tauchen(0,
                                   self.sigma_zeta,
                                   b = self.zeta_star,
                                   n = self.grid_size_zeta)
        self.zeta_hat = self.zeta_mc.state_values
        self.P_zeta = self.zeta_mc.P
        self.zeta_stat = self.zeta_mc.stationary_distributions[0]


        # Perceived actual exam grading ability shock (normally distributed)
        # means and sd and rho is for the (e_c) process 

        self.zetah_mc = tauchen(0,
                                   self.sigma_hzeta,
                                   b=self.zeta_hstar,
                                   n=self.grid_size_zeta)
        self.zeta_hhat = self.zetah_mc.state_values
        self.P_zetah = self.zetah_mc.P
        self.zetah_stat = self.zetah_mc.stationary_distributions[0]

        # Grid for stochastic processes 

        # Combine all shock values into cartesian product 

        self.Q_shocks = cartesian([self.beta_hat,self.delta_hat,\
                                    self.xi_hat,self.es_hat,self.ec_hat])
        self.Q_shocks_ind = cartesian([np.arange(len(self.beta_hat)),np.arange(len(self.delta_hat)),\
                                    np.arange(len(self.xi_hat)),np.arange(len(self.es_hat)),np.arange(len(self.ec_hat))])

        self.EBA_P = np.zeros((len(self.beta_hat),
                                len(self.delta_hat),
                                len(self.xi_hat),
                                len(self.es_hat),
                                len(self.ec_hat),
                                int(len(self.beta_hat)*
                                len(self.delta_hat)*
                                len(self.xi_hat)*
                                len(self.es_hat)*
                                len(self.ec_hat))))

        sizeEBA =   int(len(self.beta_hat)*
                                len(self.delta_hat)*
                                len(self.xi_hat)*
                                len(self.es_hat)*
                                len(self.ec_hat))

        self.EBA_P2 = self.EBA_P.reshape((sizeEBA, sizeEBA))


        for j in self.Q_shocks_ind:

            EBA_P_temp = cartesian([self.P_beta[j[0]],
                                    self.P_delta[j[1]],
                                    self.P_xi[j[2]],
                                    self.P_es[j[3]],
                                    self.P_ec[j[4]]])

            self.EBA_P[j[0], j[1], j[2],j[3],j[4], :]\
                = EBA_P_temp[:, 0]*EBA_P_temp[:, 1]*EBA_P_temp[:, 2]\
                    *EBA_P_temp[:, 3]*EBA_P_temp[:, 4]

        # Generate empty value functions and policy functions

        self.VF_e = np.zeros((len(self.Q_shocks), len(self.M), len(self.Mh)))
        self.IM_e = np.zeros((len(self.Q_shocks), len(self.M), len(self.Mh)))
        self.IM_h = np.zeros((len(self.Q_shocks), len(self.M), len(self.Mh)))


 
class EduModelParams:
    """ Parameterised class for the LifeCycleModel
            Instance of LifeCycleParams contains 
            parameter list, a paramterized EduModel

            Instance of Parameter class identified by
            unique paramter ID, param_id

            Todo
            ----


    """

    def __init__(self,
                 mod_name,
                 param_dict,
                 random_draw=False,
                 random_bounds=None,  # parameter bounds for randomly generated params
                 param_random_means=None,  # mean of random param distribution
                 param_random_cov=None,
                 uniform = False):  # cov of random param distribution :
        self.param_id = ''.join(random.choices(
            string.ascii_uppercase + string.digits, k=6))+'_'+time.strftime("%Y%m%d-%H%M%S") + '_'+mod_name
        self.mod_name = mod_name

        if random_draw == False:
            param_deterministic = param_dict['parameters']
            parameters_draws = rand_p_generator(param_deterministic,
                                    random_bounds,
                                    deterministic=1,
                                    initial=uniform)

            self.og = EduModel(param_dict, self.param_id, mod_name=mod_name)
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

            self.og = EduModel(param_dict_new, self.param_id, mod_name=mod_name)

            self.parameters = parameters_draws
            self.parameters['param_id'] = self.param_id


if __name__ == "__main__":

    # Generate  instance of LifeCycleParams class with
    # an instance of DB LifeCycle model and DC LS model
    import yaml
    import csv

    # Folder contains settings (grid sizes and non-random params)
    # and random param bounds
    settings = 'settings/'
    # Name of model
    model_name = 'test'
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

    #sampmom = pickle.load(
    #    open("{}{}/latest_means_iter.smms".format(scr_path, model_name), "rb"))

    edu_model = EduModelParams('test',
                                edu_config['baseline_lite'],
                                random_draw = False,
                                # Parameter bounds for randomly generated params
                                random_bounds = param_random_bounds)
                                # Mean of random param distribution
                                #param_random_means =sampmom[0],
                                #param_random_cov = sampmom[1],
                                #uniform = False)
