
"""
Module generates function that
creates EduModel functions given a 
set of parameters
"""

# Import packages
import numpy as np
from numba import njit, prange, jit, vectorize

def edumodel_function_factory(params, theta,share_saq,\
								share_eb,share_mcq,share_hap):   

	""" Function to create EduModel functions
	"""

	# Read in parameters from parameter dictionary 
	rho_E = params['rho_E']
	psi_E = params['psi_E']
	alpha = params['alpha']
	H  = params['H']
	phi = params['phi']*100/2.5
	a = params['a']
	b = params['b']
	M_max = params['M_max']
	varphi = params['varphi']
	varphi_sim = params['varphi_sim']

	# Translog parameters
	gamma_1 = params['gamma_1']
	gamma_2 = params['gamma_2']
	gamma_3 = params['gamma_3']
	gamma_4 = params['gamma_4']
	#gamma_4 = 1 -gamma_1 -gamma_2 - gamma_3

	# First row of zero sum diag matrix = 0
	gamma_11= params['gamma_11']
	gamma_12= params['gamma_12']
	gamma_13= params['gamma_13']
	gamma_14=  0 - gamma_11 - gamma_12 - gamma_13

	#gamma_14 = params['gamma_14']
	# Second row = 0
	gamma_22 = params['gamma_22']
	gamma_23 = params['gamma_23']
	gamma_24 = 0 - gamma_12 - gamma_23 - gamma_22
	#gamma_24 = params['gamma_24'] 

	# Third row = 0
	gamma_33 = params['gamma_33']
	gamma_34 = 0 - gamma_13 - gamma_23 - gamma_33

	# Fourth row = 0 by setting col.sum = 0 
	gamma_44 = 0 - gamma_14 -gamma_24- gamma_34
	#gamma_44 = params['gamma_44']

	iota_c = params['iota_c']
	iota_i = params['iota_i']

	# MCQ correct rate and study share vectors 
	s_share_saq = np.array(share_saq)
	s_share_eb = np.array(share_eb)
	s_share_mcq = np.array(share_mcq)
	s_share_hap = np.array(share_hap)


	@njit
	def u_grade(FG, Mh_T):
		""" Utility for agent from final course grade
			Note the final course grade is out of 100

		Parameters
		----------
		FG: float64
			 Exam grade
		Mh_T: float64
			   Coursework grade

		"""

		# Final course grade 
		FC = rho_E*FG + (1-rho_E)*Mh_T

		# Return utility
		return alpha*np.log(FC)

	@njit 
	def fin_exam_grade(zeta_T, M_T):
		""" Final grade of student, out of 100

		Parameters
		----------
		zeta_T: float64
				 ability shock realisation 
		M_T: float64
				exam knowledge capital 
		
		Returns
		-------
		grade: float64
			    final exam grade 

		"""

		return ((1-np.exp(-psi_E*(zeta_T+M_T))))*100

	@njit 
	def correct_mcq_rate(m,t):
		return ((1-np.exp(-varphi*(m)/((t+1)**2))))

	@njit 
	def hours_to_hap(m):
		return (1-np.exp(-varphi_sim*(m)))/varphi_sim

	@njit 
	def S_effort_to_IM(S_saq, S_eb, S_mcq,S_hap, m,mh,t):
		""" Converts study effort to exam knowledge 
			and CW grade accurred in the week

			Note coursework grades throughout the semester
			cap at 100

		Parameters
		----------
		S_saq: float64
				hours on SAQ
		S_eb: float64
				hours on e-books
		S_mcq: float64
				hours on MCQ points
		S_hap: float64
				hours in the sim 
		m: 	   float64
				knowledge capital at time period t (after it has depreciated)
		mh: float64
			 coursework grade at time t
		t: int
			week 

		Returns
		------

		"""

		# Translog production for investment into exam knowledge
		# stock 

		# Make sure study hours are non-negative 
		S_saq = max(0.01,S_saq)
		S_eb = max(0.01,S_eb)
		S_mcq = max(0.01,S_mcq)
		S_hap = max(0.01,S_hap)

		if t < 4:
			S_hap = .01
		if t == 0:
			S_saq = .01

		# Calculate the translog
		lnIM = np.log(phi)  + gamma_1*np.log(S_saq) \
							+ gamma_2*np.log(S_eb)\
							+ gamma_3*np.log(S_mcq)\
							+ gamma_4*np.log(S_hap)\
							+ gamma_11*np.log(S_saq)*np.log(S_saq)\
							+ gamma_12*np.log(S_saq)*np.log(S_eb)\
							+ gamma_13*np.log(S_saq)*np.log(S_mcq)\
							+ gamma_14*np.log(S_saq)*np.log(S_hap)\
							+ gamma_23*np.log(S_eb)*np.log( S_mcq)\
							+ gamma_24*np.log(S_eb)*np.log(S_hap)\
							+ gamma_34*np.log(S_mcq)*np.log(S_hap)\
							+ gamma_22*np.log(S_eb)*np.log(S_eb)\
							+ gamma_33*np.log(S_mcq)*np.log(S_mcq)\
							+ gamma_44*np.log(S_hap)*np.log(S_hap)
		
		# Knowledge creation is augmented by previous knowledge 

		IM = min(100,np.exp(lnIM))
		# Observable study outputs

		if s_share_mcq[t] > 0:
			S_mcq_hat = (S_mcq/s_share_mcq[t])

		else:
			S_mcq_hat  = 0

		if s_share_hap[t] >0:
		  	S_hap_hat =  (S_hap/s_share_hap[t])*hours_to_hap(m)
		  	
		else:
		 	S_hap_hat = 0

		if s_share_eb[t]> 0:
			S_eb_hat = 	(S_eb/s_share_eb[t])
		else: 
			S_eb_hat =0

		if s_share_saq[t]> 0:
			S_saq_hat = (S_saq/s_share_saq[t])
		else:
			S_saq_hat = 0

		# Rate of current MCQ answers 
		rate_of_correct = correct_mcq_rate(m,t)

		# Calculate coursework grade points generated 
		IMh1 = max(0, a*(rate_of_correct*iota_c + (1-rate_of_correct)*iota_i)*S_mcq_hat\
					+ b*S_hap_hat)

		# Ensure overall coursework grade does not exceed 100
		if mh + IMh1< 100:
			IMh = min(100- mh, IMh1)
			
		else:
			IMh = 0

		return IM, IMh, S_mcq_hat,S_hap_hat,S_eb_hat,S_saq_hat

	@njit
	def u_l(S,IMh):
		""" Per-period utility leisure for study and CW grade improvement
		"""
		l = H - S
		# Ensure non-course hours do not exceed 168
		l = max(l, 1e-200)

		return np.log(l) + alpha*np.log(1+ (1-rho_E)*IMh)

	return u_grade, fin_exam_grade, S_effort_to_IM, u_l, correct_mcq_rate