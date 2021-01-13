
"""
Module generates operator to solve study policies and 
generate a timeseries of simulation and generate moments

"""

# Import packages
import sys

from interpolation.splines import extrap_options as xto
from interpolation.splines import eval_linear

from quantecon.optimize import nelder_mead
from numba import njit
import time
import numpy as np

import warnings
warnings.filterwarnings('ignore')


def edu_solver_factory(og,verbose=False):

    """Generates solvers of worker policies,\
        time-series and moments

    Parameters
    ----------
    og : Instance of EduModel class
    
    Returns
    ----------
    edu_iterate : Bellman iterator to solve study policies
    TSALL : Timeseries of N student generator 
    gen_moments : Moment generator 
    
    """

    # Unpack  paramters from class
    # Reall T = number of teaching weeks (10) + study week (1) + final exam week (1) = 12
    T = og.T
    N = og.N
    rho_E = og.rho_E
    H = og.H
    delta = og.delta
    alpha = og.alpha
    d = og.d

    # Unpack uniform shocks for sim.
    U = og.U # beta shocks
    U_z = og.U_z # zeta shocks
    
    # Grids and shocks
    MM = og.MM
    M = og.M
    Mh = og.Mh
    beta_hat = og.beta_hat
    P_beta = og.P_beta
    beta_stat = og.beta_stat
    zeta_hhat = og.zeta_hhat
    zeta_hat = og.zeta_hat
    P_zeta = og.P_zeta

    # Unpack functions
    u_l = og.u_l
    S_effort_to_IM = og.S_effort_to_IM
    fin_exam_grade = og.fin_exam_grade

    # Final period value function (EV_{T})
    VF_prime_T = og.VT

    # Unpack thetas

    theta = np.array(og.config['theta'])


    @njit
    def eval_obj(S_vec, # Study investment vector 
                 m, # current exam prep
                 mh, # current CW grade (as max possible at time t-1)
                 beta, # discount realisation 
                 VF_prime_ind, # continuation value conditioned on t exogenous state index
                  t):
        
        """ Evaluates objective of Walue function for each t

        Parameters
        --------
        S_vec : 4-D array
                Vector of study hours
        m : float64
             Exam knowledge at t
        mh: float64
             Coursework grade at t
        beta: float64
               Exp discount rate at t
        VF_prime_ind : 2D array
                        Value function conditioned on t shock
        t : int
             week

        Returns
        -------
        util : float64

        """

        # If study hours exceed total hours, return penalty 
        # this ensures study hopurs are bounded 
        S_total = np.sum(S_vec)
        if S_total >= H or S_vec[0]<0 or S_vec[1]<0 or S_vec[2]<0 or S_vec[3]<0 :
            return np.inf

        else:
            # Calculate total knowlege produced and new CW grade 
            IM, IMh, S_mcq_hat,S_hap_hat,S_eb_hat,S_saq_hat\
                 = S_effort_to_IM(S_vec[0], S_vec[1], S_vec[2],S_vec[3], m,mh,t)

            # Intra-period utility from leisure + CW grade 
            period_utl = u_l(S_total,IMh)

            # Next period exam knowledge 
            m_prime = (1-d)*m + IM

            # Evalate next period CW grade  
            # (recall upper bound is already determined by S_effort_to_IM)
            mh_prime = mh + IMh

            # Evaluate continuation value 
            points = np.array([m_prime,mh_prime])
            v_prime_val = eval_linear(MM,VF_prime_ind,points)

            return period_utl + beta*delta*v_prime_val

    @njit
    def do_bell(t,VF_prime):

        """ Evaluates the Bellman Equation
        """

        # Generate empty value and policies
        VF_new = np.empty((len(beta_hat), len(M), len(Mh)))
        S_pol = np.empty((4,len(beta_hat), len(M), len(Mh)))

        # Loop through all time t state points and 
        # evaluate Walue funciton 
        for i in range(len(beta_hat)):
            # Get the beta values for i
            beta = beta_hat[i]
            for j in range(len(M)):
                for k in range(len(Mh)):
                    m = M[j]
                    mh = Mh[k]

                    # Get the index of the cartesian product of all exogenous shocks 
                    # the exogenous shock indexes the continuation value grids for t+1
                    #print(all_shocks_ind)
                    VF_prime_ind = VF_prime[i]

                    initial = np.array([5,5,5,5])
                    bounds = np.array([[0, 100], [0, 100],[0, 100], [0, 100]])
                    x = eval_obj(initial, m, mh,beta, VF_prime_ind,t)
                    sols = nelder_mead(eval_obj,\
                                         initial,\
                                         bounds = bounds,\
                                         args = (m, mh,beta, VF_prime_ind,t))

                    # Solve for the RHS of Walue
                    S_pol[0,i,j,k] = sols.x[0]
                    S_pol[1,i,j,k] = sols.x[1]
                    S_pol[2,i,j,k] = sols.x[2]
                    S_pol[3,i,j,k] = sols.x[3]

                    # Calculate total knowlege produced and new CW grade for study vector 
                    IM, IMh, S_mcq_hat,S_hap_hat,S_eb_hat,S_saq_hat\
                            = S_effort_to_IM(S_pol[0,i,j,k], S_pol[1,i,j,k], S_pol[2,i,j,k],S_pol[3,i,j,k], m,mh,t)
                    m_prime = (1-d)*m + IM
                    mh_prime = mh + IMh
                    points_star = np.array([m_prime, mh_prime])

                    # Calculate new continuation value 
                    VF_new[i,j,k] = sols.fun + beta*(1-delta)*eval_linear(MM,VF_prime_ind,points_star)

        return VF_new,S_pol

    def cond_VF(VF_new):
        """ Condition the t+1 continuation vaue on 
         time t information"""

        # make the exogenuos state index the last
        matrix_A = VF_new.transpose((1,2,0)) 

        # rows of EBA_P2 correspond to time t all exogenous state index
        # cols of EBA_P2 correspond to transition to t+1 expgenou state index
        matrix_B = P_beta 

        # numpy dot sum product over last axis of matrix_A (t+1 continuation value unconditioned)
        # see nunpy dot docs
        VF_prime = np.copy(np.dot(matrix_A,matrix_B))
        VF_prime_out = VF_prime.transpose((2,0,1))

        return VF_prime_out

    @njit
    def run_TS_i(i, S_pol_all):

        # The indices for time in the moments in this function and time-series will be as follows:

        #   TS_all[0]: Week minus 1 is a dummy week to generate auto-corrs easily
        #   TS_all[1]: Teaching week 1
        #   TS_all[t]: Teaching week t
        #   ----
        #   TS_all[10]: Teaching week 10
        #   TS_all[11]: Study week
        #   TS_all[12]: Final exam week

        # Generate empty grid
        TS_all = np.zeros((T+1,28))

        beta_ind = np.arange(len(beta_hat))\
                [np.searchsorted(np.cumsum(beta_stat), U[i,0])]
        beta = beta_hat[beta_ind]

        zetah_ind = np.arange(len(zeta_hhat))\
                [np.searchsorted(np.cumsum(P_zeta), U_z[i,0])]

        zeta_ind = np.arange(len(zeta_hat))\
                [np.searchsorted(np.cumsum(P_zeta), U_z[i,1])]
        
        for t in range(1,T):

            m = TS_all[t-1, 1]*(1-d)
            mh = TS_all[t-1, 4]

            points  = np.array([m,mh])

            S_saq = eval_linear(MM, S_pol_all[t-1, 0,beta_ind,:],points)
            S_eb = eval_linear(MM, S_pol_all[t-1, 1,beta_ind,:],points)
            S_mcq = eval_linear(MM, S_pol_all[t-1, 2,beta_ind,:],points)
            S_hap = eval_linear(MM, S_pol_all[t-1, 3,beta_ind,:],points)
            S_total = S_hap + S_eb + S_mcq + S_hap

            IM, IMh, S_mcq_hat,S_hap_hat,S_eb_hat,S_saq_hat\
                = S_effort_to_IM(S_saq, S_eb, S_mcq,S_hap, m,mh,t-1)

            # Update with vals for time t
            TS_all[t,0] = TS_all[t-1, 1] # t knowledge capital 
            TS_all[t,1] = m + IM # t+1 knowledge capotal
            TS_all[t,2] = mh # t-1 coursework grade
            TS_all[t,3] = min(mh + IMh,100) # t+1 coursework grade (coursework grade at the end of t)
            TS_all[t,4] = TS_all[t-1,5]  # t-1  S_saq 
            TS_all[t,5] = TS_all[t-1,5] + S_saq # t  S_saq cum
            TS_all[t,6] = TS_all[t-1,7]  # t-1  S_eb 
            TS_all[t,7] = S_eb + TS_all[t-1,7]   # t   S_eb cum
            TS_all[t,8] = TS_all[t-1,9]  # t-1  S_mcq 
            TS_all[t,9] = TS_all[t-1,9] + S_mcq # t  S_mcq cum
            TS_all[t,10] = TS_all[t-1,11] # t-1  S_hap
            TS_all[t,11] = S_hap + TS_all[t-1,11] # t  S_hap cum
            TS_all[t,12] = TS_all[t-1,13] # t-1  total study
            TS_all[t,13] = S_total # t total study
            TS_all[t,14] = TS_all[t-1,15]  # t-1  S_saq_hat 
            TS_all[t,15] = S_saq_hat + TS_all[t-1,15] # t  S_saq_hat cum
            TS_all[t,16] = TS_all[t-1,17]  # t-1  S_eb_hat
            TS_all[t,17] = S_eb_hat + TS_all[t-1,17]  # t S_eb_hat cum
            TS_all[t,18] = TS_all[t-1, 19]  # t-1  S_mcq 
            TS_all[t,19] = TS_all[t-1, 19] + S_mcq_hat # t  S_mcq_hat cum
            TS_all[t,20] = TS_all[t-1,21] # t-1  S_hap_hat
            TS_all[t,21] = TS_all[t-1,21] + S_hap_hat # t  S_hap_hat cum

            TS_all[t,22] = TS_all[t-1, 23]
            TS_all[t,23] = theta[t]

            beta_ind = np.arange(len(beta_hat))\
                [np.searchsorted(np.cumsum(beta_stat), U[i,t])]
            beta = beta_hat[beta_ind]

        TS_all[T,24] = fin_exam_grade(zeta_hat[zeta_ind], TS_all[T-1,1]*(1-d)) # Actual exam grade
        TS_all[T,25] = fin_exam_grade(zeta_hat[zetah_ind], TS_all[T-1,1]*(1-d)) # Randomised percieved exam grade
        TS_all[T,26] = rho_E*TS_all[T,24] + (1-rho_E)*TS_all[T-1,3] # Actual final mark
        TS_all[T,27] = zeta_hat[zeta_ind] # Actual final shock 

        # Fill in final exam marks into previous periods 
        for t in range(1,T):
            TS_all[t,24] = TS_all[T,24]
            TS_all[t,25] = TS_all[T,25]
            TS_all[t,26] = TS_all[T,26]
            TS_all[t,27] = TS_all[T,27]


        return TS_all

    @njit 
    def TSALL(S_pol_all):
        TS_all = np.empty((N,T+1, 28))

        for i in range(N):
            TS_all[i,:,:] = run_TS_i(i,S_pol_all )

        return TS_all

    def edu_iterate():

        # start with the final period continuation value given 
        # by utility from total course grade
        VF_prime = np.repeat(og.VT[np.newaxis, :,:],repeats = len(beta_hat), axis = 0)  

        # Generate empty policy functions for lentgh T-1
        S_pol_all = np.empty((T-1, 4, len(P_beta), len(M), len(Mh)))

        for t in np.arange(0,int(T-1))[::-1]:
            #print("solving week {}".format(t))
            start = time.time()
            VF_UC,S_pol = do_bell(t,VF_prime)
            #print("do bell in {}".format((time.time()-start)/60))
            S_pol_all[t,:] = S_pol

            start = time.time()
            VF_cond = cond_VF(VF_UC)
            #print("cond in {}".format((time.time()-start)/60))
            VF_prime = VF_cond

        return S_pol_all, VF_prime

    def gen_moments(TSALL):

        means = np.empty((T+1,28))
        cov = np.empty((T+1,28,28))
        
        for t in range(1,T+1):

            means[t] = np.mean(TSALL[:,t,:], axis = 0)
            cov[t,:] = np.cov(TSALL[:,t,:], rowvar=False)

        # Create moments in the same order as data moments

        moments_out = np.empty((T, 36))

        # Final grades
        moments_out[:,0] = means[1:T+1,24]# av_final
        moments_out[:,1] = means[1:T+1,26]# av_mark
        moments_out[:,2] = means[1:T+1,25]# av_markw13_exp1
        moments_out[:,3] = means[1:T+1,25]# av_markw13_exp2

        # Study hours inputs
        moments_out[:,4] = means[1:T+1,11]# av_game_session_hours_cumul
        moments_out[:,5] = means[1:T+1,7] # av_ebook_session_hours_cumul
        moments_out[:,6] = means[1:T+1,9] # avg mcq session hours cum
        moments_out[:,7] = means[1:T+1,5] # avg saq session hours cum

        # Observable study outputs 
        moments_out[:,8] = means[1:T+1,21] # av_player_happy_deploym_cumul
        moments_out[:,9] = means[1:T+1,19] # av_mcq_attempt_nonrev_cumul'
        moments_out[:,10] = means[1:T+1,15] # av_saq_attempt_cumul
        moments_out[:,11] = means[1:T+1,23] # av_mcq_Cshare_nonrev_cumul

        # SDs 
        moments_out[:,12] = cov[1:T+1,24,24] # sd_final
        moments_out[:,13] = cov[1:T+1,26,26] # sd_mark
        moments_out[:,14] = cov[1:T+1,25,25] # sd_markw13_exp1
        moments_out[:,15] = cov[1:T+1,25,25] # sd_markw13_exp2

        moments_out[:,16] = cov[1:T+1,11,11] # sd_game_session_hours_cumul
        moments_out[:,17] = cov[1:T+1,7,7] # sd_ebook_session_hours_cumul
        moments_out[:,18] = cov[1:T+1,9,9] # sd_mcq_session_hours_cumul
        moments_out[:,19] = cov[1:T+1,5,5] # sd_saq_session_hours_cumul

        moments_out[:,20] = cov[1:T+1,21,21] # sd_player_happy_deploym_cumul
        moments_out[:,21] = cov[1:T+1,19,19] # sd_mcq_attempt_nonrev_cumul
        moments_out[:,22] = cov[1:T+1,15,15] # sd_sa_attempt_cumul
        moments_out[:,23] = cov[1:T+1,23,23] # sd_mcq_Cshare_nonrev_cumul

        # Autocorrelations
        moments_out[:,24] = cov[1:T+1,11,10] # acgame_session_hours
        moments_out[:,25] = cov[1:T+1,7,6] # acebook_session_hours
        moments_out[:,26] = cov[1:T+1,21,20] # acmcq_session_hours
        moments_out[:,27] = cov[1:T+1,5,4] # acsaq_session_hours
        moments_out[:,28] = cov[1:T+1,23,22] # acmcq_Cshare_nonrev

        # Covarainces 
        moments_out[:,29] = cov[1:T+1,9,5] # cmcsaq_session_hours
        moments_out[:,30] = cov[1:T+1,5,11] # cgsaq_session_hours
        moments_out[:,31] = cov[1:T+1,9,11] # cgmcq_session_hours
        moments_out[:,32] = cov[1:T+1,7,5] # cesaq_session_hours
        moments_out[:,33] = cov[1:T+1,7,9] # cemcq_session_hours
        moments_out[:,34] = cov[1:T+1,7,11] # ceg_session_hours

        # Atar 
        moments_out[:,35] = cov[1:T+1,27,24] # c_atar_ii 

        return moments_out[0:T-1,:]
   
    return edu_iterate, TSALL,gen_moments

def generate_study_pols(og):

    edu_iterate,TSALL,gen_moments = edu_solver_factory(og)
    S_pol_all, VF_prime = edu_iterate()
    TS_all = TSALL(S_pol_all)
    moments_out = gen_moments(TS_all)

    return moments_out
