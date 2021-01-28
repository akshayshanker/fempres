import numpy as np

def rand_p_generator(param_deterministic, # dictionary containing deterministic parameters
                    param_random_bounds, # dictionary containing random parameters 
                    param_random_means = 0,  # array of means of sampling distribution
                    param_random_cov = 0,  # array of means of sampling distribution
                    deterministic = 1,   # flag to make all parmaeters determinstic 
                    initial = 1, # if initial then draws from uniform disitrubution within bounds 
                    ):

    """Function generates list of parameters with 
    random parameters generated for params in the dictionary "pram_bounds"
    and deterministic parmameters for params in the dictionary "parameters". 

    Note: parameters in param_bounds will override any deterministic parameters in parameters
    If you want to make a parameter deterministic, remove it from the param_bounds list 
    """

    parameters = {}

    # first pull out all the parameters that are deterministic

    for key in param_deterministic:
        parameters[key] = param_deterministic[key]

    random_param_bounds_ar = \
        np.array([(bdns) for key, bdns in param_random_bounds.items()] ) 

    # generate random sample
    param_random_cov = np.array(param_random_cov)
    random_draw = np.random.uniform(0, 1)
    # noise injection rate of 10%
    if random_draw< 0.02 and initial==0:
        np.fill_diagonal(param_random_cov, param_random_cov.diagonal()*2)

    in_range = False
    if deterministic == 0 and initial == 0:

        while in_range == False:
            draws = np.random.multivariate_normal(param_random_means, param_random_cov)
            for i,key in zip(np.arange(len(draws)),param_random_bounds.keys()):
                parameters[key]  = draws[i]

            if np.sum(draws < random_param_bounds_ar[:,0]) + np.sum(draws > random_param_bounds_ar[:,1]) == 0 and\
                parameters['gamma_1'] + parameters['gamma_2'] + parameters['gamma_3'] < 1 and np.abs(parameters['sigma_M'] -1)>.01:
                in_range = True
               #print("in range")
            else:
                pass

    if deterministic == 0 and initial == 1:
        while in_range == False:

            for key in param_random_bounds:
                parameters[key]  = np.random.uniform(param_random_bounds[key][0], param_random_bounds[key][1])

            if parameters['gamma_1'] + parameters['gamma_2'] + parameters['gamma_3'] < 1 and np.abs(parameters['sigma_M'] -1)>.01:
                #print('yes')
                in_range = True
            #in_range = True

    parameters['ID'] = np.random.randint(0,999999999999)

    return parameters