import numpy as np



keys = ['dt', 'num_steps','init','gamma','kBT', 'mass', 'coarse']
vals = [.01, 100, [1,1,1], 1, 1, 1, 1 ]

params = { k:v for k,v in zip(keys, vals)}

def simulate_free_diffusion_underdamped(N, all_params):
    '''
    all_params is a dict with keys ['sim,'init','gamma', 'kBT', 'mass'], the first two keys are 3 element list
        sim = [N, num_steps, dt]
        init = [p0, Sp0, Sx0]
        'gamma', 'kBT', 'mass' as expected

    save_skip = n skips every 'n' sim steps when saving the output 
    '''

    p0, Sp0, Sx0 = all_params['init']
    dt, num_steps  = [all_params[k] for k in ['dt','num_steps'] ]
    gamma, kBT, mass  = [all_params[k] for k in ['gamma','kBT', 'mass'] ]
    sigma = np.sqrt(2 * gamma * kBT)
    save_skip = params['coarse']

    save_indices = range(0,num_steps+1)[::save_skip]
    # Initialize the array to store x position and x momentum (x,px)
    phase_data = np.zeros((N, len(save_indices), 2))
    # Sample the initial momentum 
    phase_data[:,0] = np.random.normal([0,p0],[Sx0,Sp0], (N,2))
    # initialize vars
    i=0
    curr_data = phase_data[:,i]
     # Iterate over each time step and simulate the process
    for t in range(1, num_steps + 1):
        curr_data = curr_data + dt * (curr_data[...,1][:,None] * np.array([1,-gamma]))/mass
        curr_data[:,1] += sigma * np.random.normal(0,np.sqrt(dt), N)

        if t in save_indices:
            i += 1
            phase_data[:,i] = curr_data

    return phase_data


def realepr(t, P0, SigmaPi, SigmaXi):
    '''
    gives the epr as a function of time assuming gamma=kBT=mass=1
    '''
    numerator = (np.exp(-2*t) * (np.exp(4*t) + 2*np.exp(3*t)*(-2 + SigmaPi**2) -
                                 4*np.exp(t)*(-2 + SigmaPi**2)*(-1 + P0**2 + SigmaPi**2) +
                                 (-1 + P0**2 + SigmaPi**2)*(-SigmaXi**2 - 2*(2 + t) +
                                  SigmaPi**2*(3 + SigmaXi**2 + 2*t)) +
                                 np.exp(2*t)*(7 - 7*SigmaPi**2 + SigmaPi**4 +
                                 P0**2*(-4 + SigmaPi**2 + SigmaXi**2 + 2*t))))
    denominator = (-4 - 4*np.exp(t)*(-2 + SigmaPi**2) - SigmaXi**2 - 2*t +
                   SigmaPi**2*(3 + SigmaXi**2 + 2*t) +
                   np.exp(2*t)*(-4 + SigmaPi**2 + SigmaXi**2 + 2*t))
    return numerator / denominator

from scipy.stats import multivariate_normal

def distribution(t, P0, SigmaPi, SigmaXi):
    '''
    gives the distribution as a function of time assuming gamma=kBT=mass=1
    '''
    time_decay = np.exp(-t)
    mean = [(1-time_decay) * P0, P0 * time_decay]

    var_p = (SigmaPi**2-1) * time_decay**2 + 1
    var_x = -(1-time_decay) * ( (3-time_decay) - SigmaPi**2 * (1-time_decay) ) + 2*t + SigmaXi**2
    cov = (1-time_decay)*(1-(1-SigmaPi**2)*time_decay)

    var = [[var_x, cov],[cov, var_p]]

    return  multivariate_normal(mean, var, allow_singular=True)


def traj_delta_Shannon (coords, times, P0, SigmaPi, SigmaXi, skip=1):
    '''
    gives change in shannon entropy for each timestep, assumes gamma=kB T = mass =1
    '''
    parameters = P0, SigmaPi, SigmaXi
    assert coords.shape[1] == len(times), 'trajectories need to match times in length'

    delShannon = []
    for idx in range(len(times)-1)[::skip]:
        c1, c2 = coords[:,idx], coords[:,idx+1]
        t1, t2 = times[idx:idx+2]
        delShannon.append(distribution(t1, *parameters).logpdf(c1) - distribution(t2, *parameters).logpdf(c2))

    return np.array(delShannon).T

def traj_dEnt(coords, times, P0, SigmaPi, SigmaXi, skip=1):
    '''
    gives stoch entropy production for each time step, assumed parameter = 1
    '''
    return -np.diff(coords[...,1]**2/2, axis=1)[:,::skip,] + traj_delta_Shannon(coords, times, P0, SigmaPi, SigmaXi, skip=skip)