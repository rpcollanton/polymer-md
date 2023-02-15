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
    nblocks = list(range(5,200,5))
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

def num_independent_samples(A: np.ndarray):
    '''
    Args:
        A (np.ndarray): Array of dimension (n x d) where n is the number of samples and 
                        d is the dimensionality of the property A. Must be evenly spaced.

    Returns:
        nsamples (int):  Number of independent samples based on estimated autocorrelation time
    '''

    t_total = np.shape(A)[0]
    tau_A = estimate_autocorrelation_time(A)
    nsamples = t_total/(2*tau_A)
    
    return nsamples

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

    # compute estimator variance
    nsamples = num_independent_samples(A)
    sigma = variance_A / nsamples

    return sigma

def get_independent_samples(A,factor=1):
    '''
    Args:
        A (np.ndarray): Array of dimension (n x d) where n is the number of samples and 
                        d is the dimensionality of the property A. Must be evenly spaced.
        
        factor (int):   Factor by which to extend the samples. Default is 1

    Returns:
        samples (float):  Estimated variance of the finite-length average of A based on 
                        the estimated autocorrelation time
    '''

    # compute number of samples
    nsamples = num_independent_samples(A)/factor
    samplesize = int(np.shape(A)[0]/nsamples)
    samples = np.array([np.mean(A[i*samplesize:(i+1)*samplesize]) for i in range(int(nsamples))])

    return samples