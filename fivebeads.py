import numpy as np
import torch


keys = ['dt', 'num_steps', 'init', 'kBT', 'mob', 'k', 'coarse']
vals = [.01, 100, np.array([0.56,-0.23,0.14,-0.12,0.09,0.87,-0.15,0.08,-0.19,0.92,-0.21,0.11,0.68,-0.17,0.79]), [1, 2], 1, 1, 1 ]

params = { k:v for k,v in zip(keys, vals)}
"""
this class gives the probability distribution of five beads with zero initial mean.

The special solution is 
dz/dt = cov_evo@z + ode_constant
The general solution is
z(t)= stable_solution + P exp(-Jt) P^(-1) z_0
"""

def simulate_five_spring_overdamped(N, all_params):
    dt = all_params['dt']
    k = all_params['k']
    T0,T1= all_params['kBT']
    zero_mean = np.array([0, 0, 0, 0, 0])
    diff_const = np.linspace(T0, T1, 5)
    noise_cov = np.diag(2*diff_const*dt)
    steps = all_params['num_steps']
    num_paths = N

# cov_matrix
    mat_sigma = np.zeros((5, 5))
    mat_sigma[np.triu_indices(5)] = all_params['init']
    mat_sigma = mat_sigma + mat_sigma.T - np.diag(mat_sigma.diagonal())
    x_cov = mat_sigma
# A is the deterministic dynamics part
    A = np.array([
        [-2*k, k, 0, 0, 0],
        [k, -2*k, k, 0, 0],
        [0, k, -2*k, k, 0],
        [0, 0, k, -2*k, k],
        [0, 0, 0, k, -2*k]
    ])
# Create the space to save data
    phase_data = np.zeros((num_paths,steps,5))
# Initialize phase data
    phase_data[:,0,:] =  np.random.multivariate_normal(zero_mean, x_cov, num_paths)
    for time in range(1, steps):
        phase_data[:,time,:] = phase_data[:,time-1,:] + np.einsum('ij,jk->ik', phase_data[:,time-1,:],A)*dt + np.random.multivariate_normal(zero_mean, noise_cov, size=num_paths)

    return phase_data

# define the class which can return related values.
class five_beads:
    # The decay modes, i.e., the eigenvector of cov_evo
    # cov_evo matrix
    cov_evo = np.array([
    [-4, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [1, -4, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
    [0, 1, -4, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0], 
    [0, 0, 1, -4, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 1, -4, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0], 
    [0, 2, 0, 0, 0, -4, 2, 0, 0, 0, 0, 0, 0, 0, 0], 
    [0, 0, 1, 0, 0, 1, -4, 1, 0, 1, 0, 0, 0, 0, 0], 
    [0, 0, 0, 1, 0, 0, 1, -4, 1, 0, 1, 0, 0, 0, 0], 
    [0, 0, 0, 0, 1, 0, 0, 1, -4, 0, 0, 1, 0, 0, 0], 
    [0, 0, 0, 0, 0, 0, 2, 0, 0, -4, 2, 0, 0, 0, 0], 
    [0, 0, 0, 0, 0, 0, 0, 1, 0, 1, -4, 1, 1, 0, 0], 
    [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, -4, 0, 1, 0], 
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, -4, 2, 0], 
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, -4, 1], 
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, -4]
    ])
    A = np.array([
        [-2, 1, 0, 0, 0],
        [1, -2, 1, 0, 0],
        [0, 1, -2, 1, 0],
        [0, 0, 1, -2, 1],
        [0, 0, 0, 1, -2]
    ])
    matrix_p = np.load('./matrix_p.npy')
    matrix_p_inv = np.load('./matrix_p_inv.npy')
    matrix_j = np.diag([-6.0,-5.0,-4.0,-4.0,-4.0,-3.0,-2.0,-5-np.sqrt(3),-5+np.sqrt(3),-4-2*np.sqrt(3),-4-np.sqrt(3),-4+np.sqrt(3),-4+2*np.sqrt(3),-3-np.sqrt(3),-3+np.sqrt(3)])

    
    def __init__(self, x_cov, bath_sigma2):
        # input z = [E(x1^2), E(x1x2), ..., E(x4x5), E(x5^2)] a 15-dim vec as x_cov
        # bath_sigma2 as (sigma1^2, sigma2^2, sigma3^2, sigma4^2, sigma5^2)
        self.cov = x_cov
        self.bath_sigma2 = bath_sigma2
        self.D_mat = np.diag(1/2*bath_sigma2)
        self.D_inv = np.linalg.inv(np.diag(1/2*bath_sigma2))
    
    # The EoM is dz/dt= cov_evo@z + sigma2

    def D_inverse(self):
        return self.D_inv

    # Return the multivarite Gaussian distribution Sigma matrix given z=x_cov
    def mat_sigma(self, x_cov):
        mat_sigma = np.zeros((5, 5))
        # Fill the upper triangular part with the vector
        mat_sigma[np.triu_indices(5)] = x_cov
        mat_sigma = mat_sigma + mat_sigma.T - np.diag(mat_sigma.diagonal())
        return mat_sigma    

    # Compute P exp(-Jt) P^(-1) z for any z and t
    def exp_z_t(self, z, time):
        return self.matrix_p @ np.diag(np.exp(np.diag(self.matrix_j*time))) @ self.matrix_p_inv @ z
    
    def ode_constant(self):
        sigma2 = np.array([self.bath_sigma2[0],0,0,0,0,self.bath_sigma2[1],0,0,0,self.bath_sigma2[2],0,0,self.bath_sigma2[3],0,self.bath_sigma2[4]])
        return sigma2
    # compute stable solution
    def stable_solution(self):
        return -np.linalg.inv(self.cov_evo) @ self.ode_constant()
    
    def stable_covmatrix(self):
        return self.mat_sigma(self.stable_solution())
    
    def stable_covmatrix_inv(self):
        return np.linalg.inv(self.mat_sigma(self.stable_solution()))

    def z_0(self):
        return self.cov - self.stable_solution()
    
    def z(self, t):
        return self.stable_solution() + self.exp_z_t(self.z_0(),t) 
    
    def time_covmatrix(self,t):
        return self.mat_sigma(self.z(t))

    def time_covmatrix_inv(self,t):
        return np.linalg.inv(self.time_covmatrix(t))
    
    def stable_ent_production(self):
        u = (self.A + self.D_mat @ self.stable_covmatrix_inv() )
        ent_production= np.trace(u.T @self.D_inverse()@ u @ self.stable_covmatrix())
        return ent_production
    
    # Compute u(x,t)
    def u(self,x,t):
        u_mat=(self.A+self.D_mat@self.time_covmatrix_inv(t))
        return x@(u_mat).T
    
    # Compute the probability distribution at time t given the position x
    def prob(self,x,t):
        cov_mat = self.time_covmatrix(t)
        det = np.linalg.det(cov_mat)
        cov_mat_inv = np.linalg.inv(cov_mat)
        x_column = np.hsplit(x,5) 
        exponent = np.zeros((len(x),1))
        for index_i in range(0,5):
            for index_j in range(0,5):
                exponent += x_column[index_i]*cov_mat_inv[index_i][index_j]*x_column[index_j]
        coefficient = 1/np.sqrt((2* np.pi)**5*det)
        prob = coefficient * np.exp(-1/2*exponent)
        return prob
    
    def logf(self,x,t):
        return np.log(self.prob(x,t))
    
    def dtlogf(self, x, t):
        dt = 1e-6
        return (np.log(self.prob(x, t+dt)) - np.log(self.prob(x, t-dt)))/(2*dt)
    ################Ent part##########
    def path_heat_theo(self, data, step_begin, step_end):
    # extract the data needed
        traj = data[:,step_begin:step_end+1,:]
    # compute the linear force
        force = np.einsum('ijk,kl->ijl', traj, self.A)
    # the average force
        force_adv = force[:,:-1,:]
        force_next = force[:,1:,:]
        force_avg = (force_adv + force_next)/2
    # compute the F/D  
        foverD = np.einsum('ijk,kl->ijl', force_avg, self.D_inverse())
    # dx part
        dx= np.diff(traj, axis=1)
    # heat: the definition of the heat is F/D circ dx
        fcircdx_overD = np.einsum('ijk,ijk->ij',foverD,dx)
        heat_path = np.sum(fcircdx_overD, axis=-1)
        return heat_path

    def path_ent_theo(self, data, step_begin, step_end):
    # return change in stochastic entropy
        x_initial = data[:,step_begin,:]
        x_final= data[:,step_end,:]
        logf_initial= self.logf(x_initial, t=step_begin*params['dt'])
        logf_final= self.logf(x_final, t=step_end*params['dt'])
        ent_change = logf_initial-logf_final
        return ent_change.ravel()
    
    def path_diss_defi_theo(self, data, step_begin, step_end):
        '''
        compute dissipation from the definition Heat + Stochastic entropy change
        '''
        diss = self.path_heat_theo(data, step_begin, step_end) + self.path_ent_theo(data, step_begin, step_end)
        return diss

    def path_diss_path_theo(self, data, step_begin, step_end):
        '''
        compute dissipation from path defintion \int dx-dtlogfdt
        '''
        dtlogf = np.zeros((len(data),1))
        udx = np.zeros((len(data),1))
        for step in range(step_begin, step_end):
        # x, nextx and delta x
            data_current = data[:,step,:]
            data_next = data[:,step+1,:]
            dx = data_next - data_current
            u_average = (self.u(data_current, step*params['dt']) + self.u(data_next, (step+1)*params['dt']))/2
            udx_over_D = u_average*dx @ self.D_inverse()
            dtlogf += -1*self.dtlogf(data_current, step*params['dt']).reshape(-1,1)*params['dt']
            udx += np.sum(udx_over_D,axis=-1).reshape(-1,1)
        return udx+dtlogf

    
    def path_diss_path_theo_cum(self, data, step_begin, step_end):
        # return cumulant ent trajectory-wise
        num_steps = step_end - step_begin
        diss_path = np.zeros((len(data),num_steps))
        for step in range(step_begin, step_end):
        # x, nextx and delta x
            data_current = data[:,step,:]
            data_next = data[:,step+1,:]
            dx = data_next - data_current
            u_average = (self.u(data_current, step*params['dt']) + self.u(data_next, (step+1)*params['dt']))/2
            udx_over_D = u_average*dx @ self.D_inverse()
            diss_path[:,step] = (-1*self.dtlogf(data_current, step*params['dt'])*params['dt']).flatten() + np.sum(udx_over_D,axis=-1)
            diss = np.cumsum(diss_path, axis=1)
        return diss

    def path_diss_nn(self, u, dlogf, data, step_begin, step_end):
    # compute dissipation from path defintion \int dx-dtlogfdt
        path_ent = torch.zeros((len(data),1))
        with torch.no_grad():
            for step in range(step_begin, step_end):
            # x, nextx and delta x
                data_current = data[:,step,:]
                data_next = data[:,step+1,:]
                dx = data_next - data_current
                u_average = (u[step](data_current) + u[step](data_next))/2
                udx_over_D =  torch.matmul(u_average*dx, torch.from_numpy(self.D_inverse()).float())
                dtlogf = -1*dlogf[step](data_current).reshape(-1,1)*params['dt']
                udx = torch.sum(udx_over_D,axis=-1).reshape(-1,1)
                path_ent += dtlogf + udx
        return path_ent
    
    def path_diss_step_nn(self, u, dlogf, data, step_begin, step_end):
    # compute cumulant dissipation
        path_ent = torch.zeros((len(data),step_end-step_begin))
        with torch.no_grad():
            for step in range(step_begin, step_end):
        # x, nextx and delta x
                data_current = data[:,step,:]
                data_next = data[:,step+1,:]
                dx = data_next - data_current
                u_average = (u[step](data_current) + u[step](data_next))/2
                udx_over_D =  torch.matmul(u_average*dx, torch.from_numpy(self.D_inverse()).float())
                dtlogf = (-1*dlogf[step](data_current)*params['dt']).flatten()
                udx = torch.sum(udx_over_D,axis=-1)
                path_ent[:,step]= dtlogf + udx
        return path_ent
        
