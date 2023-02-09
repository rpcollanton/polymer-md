import numpy as np
from scipy.stats import linregress

def estimate_autocorrelation_time(A: np.ndarray):
    '''
    Args:
        A (np.ndarray): Array of dimension (n x d) where n is the number of samples and 
                        d is the dimensionality of the property A

    Returns:
        tau (float):    Autocorrelation time estimated using block averaging and linear regression 
    '''

    # number of samples of A
    n = np.shape(A)[0]
    shape_A_sample = np.shape(A)[1:]

    # compute variance of A over entire simulation
    variance_A = np.var(A,axis=0)

    # number of blocks for each sample
    nblocks = list(range(5,100,5))
    y_variance_estimate = np.zeros((len(nblocks),) + shape_A_sample)    # lhs of variance equation
    x_adjusted_variance = np.zeros((len(nblocks),) + shape_A_sample)    # rhs of variance equation without slope

    for i,nblock in enumerate(nblocks):
        A_blocks = np.zeros((nblock,) + shape_A_sample)
        l_block = int(n/nblock) # number of samples in each block, round down

        # compute average for each block
        for j in range(nblock):
            indices = list(range(j*l_block, (j+1)*l_block))
            A_blocks[j] = np.average(A[indices], axis=0)
        
        # compute variance of averages
        y_variance_estimate[i] = np.var(A_blocks, axis=0)

        # store the relevant quantity 2*variance/t_block and the block variance
        x_adjusted_variance[i] = 2*variance_A/l_block

    # perform regression to estimate the autocorrelation time
    tau = np.zeros(shape_A_sample)
    it = np.nditer(tau,flags=['multi_index'])
    for i in it:
        idx = it.multi_index
        yvals = [y_variance_estimate[i][idx] for i in range(len(nblocks))]
        xvals = [x_adjusted_variance[i][idx] for i in range(len(nblocks))]
        res = linregress(xvals, yvals)
        tau[idx] = res.slope

    return tau

def estimator_variance(A: np.ndarray): 
    '''
    Args:
        A (np.ndarray): Array of dimension (n x d) where n is the number of samples and 
                        d is the dimensionality of the property A. Must be evenly spaced.

    Returns:
        sigma (float):  Estimated variance of the finite-length average of A based on 
                        the estimated autocorrelation time
    '''
    
    # variance over entire simulation
    variance_A = np.var(A,axis=0)
    
    # estimated autocorrelation time
    tau_A = estimate_autocorrelation_time(A)

    # compute estimator variance
    t_total = np.shape(A)[0]
    sigma = 2*tau_A / t_total * variance_A

    return sigma