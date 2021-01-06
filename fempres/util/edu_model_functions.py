 

# Import packages

import numpy as np

from quantecon import tauchen

import matplotlib.pyplot as plt
from itertools import product
from numba import njit, prange, jit, vectorize

from sklearn.utils.extmath import cartesian 


from interpolation.splines import UCGrid, CGrid, nodes, eval_linear
import time
import pandas as pd
from pathos.pools import ProcessPool 
#from pathos.multiprocessing import Pool as ProcessPool
import copy



def edumodel_function_factory(params):   

	@njit
	def u_grade(FG, Mh_T):
		""" Utility for agent from final course grade
		"""

		FC = params['rho_E']*FG + (1-params['rho_E'])*Mh_T

		return params['alpha_FC']*np.log(FC)

	@njit 
	def fin_exam_grade(zeta_T, M_T):
		""" Final grade of student 
		"""

		return 100*((1-np.exp(-params['psi_E']*zeta_T*M_T))/params['psi_E'])

	@njit 
	def cw_grade(e_c, IMh):
		""" Weekly coursework assesment grade 
		"""

		return 100*((1-np.exp(-params['psi_C']*e_c*Mh))/params['psi_C'])


	@njit 
	def S_effort(e_s, IM, IMh):
		""" Study effort required for investment in exam and CW 
		"""


		return e_s*(IM**(params['alpha_M']))*(IM**(1-params['alpha_M']))

	@njit
	def u_l(xi_t,e_s, IM, IMh):
		""" Per-period utility leisure for investment in exam and CW 
		"""

		l = params['H'] - S_effort(e_s, IM, IMh)

		l = np.min(l, 1e-200)

		return xi_t*np.log(l)

	return u_grade, fin_exam_grade,cw_grade, S_effort, u_l