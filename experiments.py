import torch
import numpy as np

'''
all loss/inference functions take in:

    s : trajectories tensors with dims [N_traj , steps , coords_values]
        coord values are typically assumed to be in the order [x_1,x_2..., v_1,v_2,...]
        if a time coord is added, it should be the last coordinate index

    w : always just model(s)

    traj_generator : TrajectoryGenerator instance that was used to make the trajectories, 

loss functions must return a torch scalar that can have .backwards run on it

inference functions return a torch tensor that could be a scalar
    inference functions are assumed to be run with torch.no_grad
'''

def get_UD_currents(s, w, traj_generator, vjs_multiplier = 2):
    params = traj_generator.params
    #trims off the time vector, if there was one 
    if s.shape[-1]%2 == 1 and s.shape[-1] > 1:
        s = s[...,:-1]
    
    w_len = w.shape[1]
    s_len = s.shape[1]
    n_steps = min(w_len, s_len) - 1

    n_dim = int(s.shape[-1]/2)

    s = s[:,:n_steps+1]
    w = w[:,:n_steps+1]

    assert n_steps > 0, 'must have at least 1 time step'

    w1 = w[:,:-1,]
 
    ds, dw = s.diff(axis=1), w.diff(axis=1)
    dx = ds[...,:n_dim]
    dp = ds[...,n_dim:]
    
    factor = .5
    if params['coarse'] > 1 and traj_generator.infer_velocity:
        factor = .75
    J = params['gamma']*w1*dx-factor*dw*dp
    #J = model.params['gamma']*w1*dx-.75*dw*dp
    VJS = vjs_multiplier*np.sqrt(params['gamma']*params['kBT'])*w1**2*params['Dt']
    if n_steps > 1:
            J = J.sum(axis=1)
            VJS = VJS.sum(axis=1)

    return J.mean(axis=0), VJS.mean(axis=0)

def entropy_loss_TUR(s, w, traj_generator): 
    J, VJS = get_UD_currents(s, w, traj_generator)
    return torch.sum( VJS/len(VJS) - J)

def entropy_infer_TUR(s, w, traj_generator):
    J, VJS = get_UD_currents(s, w, traj_generator)
    return 2*torch.sum(J**2/VJS).squeeze()

def entropy_loss_ML(s, w, traj_generator): 
    J, VJS = get_UD_currents(s, w, traj_generator, vjs_multiplier = .5)
    return torch.sum( VJS/len(VJS) - J)

def entropy_infer_ML(s, w, traj_generator):
    J, VJS = get_UD_currents(s, w, traj_generator, vjs_multiplier=.5)
    return traj_generator.params['Dt']*torch.mean(w**2)

def force_loss(s, w, traj_generator):
    params = traj_generator.params
    #trims off the time vector, if there was one 
    if s.shape[-1]%2 == 1 and s.shape[-1] > 1:
        s = s[...,:-1]
    
    n_dim = int(s.shape[-1]/2)
    n_steps = s.shape[1] - 1

    #we dont really care about time-averaged force; so this makes sure it's only for one time step
    assert n_steps == 1, 'trajs must have exactly 1 time step'

    dv = s.diff(axis=1)[...,n_dim:]
    w1 = w[:,:-1]

    loss = -(dv*w1).mean(axis=0) + (.5*w1**2*params['Dt']).mean(axis=0)
    if params['coarse'] > 1 and traj_generator.infer_velocity:
        loss += (.25*dv*w.diff(axis=1)).mean(axis=0)

    return torch.sum(loss)

def force_infer(s, w, traj_generator):
    return w


def tut_loss(s, Q, traj_generator):
    return (Q**2).mean() + 1000*torch.abs(1-Q.mean())

def tut_infer(s, Q, traj_generator):
                return torch.log( ((Q**2).mean() + Q.mean()*Q) / ((Q**2).mean() - Q.mean()*Q) )



