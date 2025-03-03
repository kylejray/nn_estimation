import torch
import numpy as np
import matplotlib.pyplot as plt
import json
import os
from datetime import datetime

from argparse import Namespace
from training import ModelTrainer, TrajectoryGenerator
from models import SingleTimeStep

from experiments import force_infer, od_force_loss, od_dtlogf_loss, od_dxlogf_loss, od_sigmasq_dxlogf_loss, od_force_loss_2nd, od_entropy_loss_ML_2nd, od_dtlogf_loss_2nd
from experiments import entropy_loss_ML, entropy_infer_ML, od_entropy_loss_ML, od_entropy_infer_ML

import fivebeads

from fivebeads import simulate_five_spring_overdamped, five_beads
#from free_diffusion import params, simulate_free_diffusion_underdamped, realepr, traj_dEnt


############################################
def multistep_train(options_list, optimizer, u_loss, dtlogf_loss, infer, dynamics, params, cg_num_steps, cg_data_train, cg_data_validate, coarse_step, directory, order):
    u_model_options, dtlogf_model_options, training_options = options_list
    WeightFunction_u_multisteps = [SingleTimeStep(u_model_options) for _ in range(cg_num_steps)]
    WeightFunction_dtlogf_multisteps = [SingleTimeStep(dtlogf_model_options) for _ in range(cg_num_steps)]

    Fivebeads_multisteps = [TrajectoryGenerator(dynamics, params) for _ in range(cg_num_steps)]

    u_multisteps = [ModelTrainer(WeightFunction_u_multisteps[i], Fivebeads_multisteps[i], optimizer, u_loss, infer, training_options) for i in range(cg_num_steps)]
    dtlogf_multisteps  = [ModelTrainer(WeightFunction_dtlogf_multisteps[i], Fivebeads_multisteps[i], optimizer, dtlogf_loss, infer, training_options) for i in range(cg_num_steps)]

    p = -15
    for step in range(len(u_multisteps)):
        p_new = round(100*step/len(u_multisteps))
    # Setting up training data and validation data
        Fivebeads_multisteps[step].infinite_data = False
        Fivebeads_multisteps[step].data = cg_data_train[:,step:step+order+1,:]
        u_multisteps[step].validation_data = cg_data_validate[:,step:step+order+1,:]
        dtlogf_multisteps[step].validation_data = cg_data_validate[:,step:step+order+1,:] 
        #print(hasattr(EntProd_ML_multisteps[step], 'validation_data'))  # Check if 'data' exists
     # Training
        if p_new >= p + 15:
            print(f'Training step {step} using coarse step {coarse_step} and loss order {order}')
            p = p_new
        u_multisteps[step].train()
        dtlogf_multisteps[step].train()
        ##Reset parameter every 10 steps
        #if (step+1)%10 !=0:
    # Load parameters from the previous step except for the last step
        try:
            WeightFunction_u_multisteps[step+1].load_state_dict(WeightFunction_u_multisteps[step].state_dict())
            WeightFunction_dtlogf_multisteps[step+1].load_state_dict(WeightFunction_dtlogf_multisteps[step].state_dict())
        except:
            pass

        data_save(directory, params, u_multisteps, dtlogf_multisteps, WeightFunction_u_multisteps, WeightFunction_dtlogf_multisteps, cg_num_steps, coarse_step)
    return WeightFunction_u_multisteps, WeightFunction_dtlogf_multisteps

def data_save(directory, params, u_multisteps, dtlogf_multisteps, WeightFunction_u_multisteps, WeightFunction_dtlogf_multisteps, cg_num_steps, coarse_step):
    u_training_loss=[]
    dtlogf_training_loss=[]
    u_validation_loss=[]
    dtlogf_validation_loss=[]

    for step in range(cg_num_steps):
        u_training_loss.append(u_multisteps[step].all_loss)
        dtlogf_training_loss.append(dtlogf_multisteps[step].all_loss)
        u_validation_loss.append(u_multisteps[step].epoch_validation_loss)
        dtlogf_validation_loss.append(dtlogf_multisteps[step].epoch_validation_loss)

    data={'u_training_loss':u_training_loss, 'dtlogf_training_loss':dtlogf_training_loss, 'u_validation_loss':u_validation_loss, 'dtlogf_validation_loss':dtlogf_validation_loss}
    
    coarse_directory= directory + f'/coarse_{coarse_step}'
    model_directory = coarse_directory + '/models'
    os.makedirs(model_directory, exist_ok=True)

    filename = f"loss.json"
    file_path = os.path.join(coarse_directory, filename)
    with open(file_path, 'w') as file:
        json.dump(data, file, indent=4)

    filename = f"parameters.json"
    file_path = os.path.join(coarse_directory, filename)
    with open(file_path, 'w') as file:
        json.dump(params, file, indent=4)

    for idx, model_u in enumerate(WeightFunction_u_multisteps):
        torch.save(model_u.state_dict(), f'{model_directory}/model_u_{idx}.pth')

    for idx, model_dtlogf in enumerate(WeightFunction_dtlogf_multisteps):
        torch.save(model_dtlogf.state_dict(), f'{model_directory}/model_dtlogf_{idx}.pth')

def r2_score(y_true, y_pred):
    """
    Compute the R^2 (coefficient of determination) score.

    Parameters:
    y_true (torch.Tensor): Ground truth values.
    y_pred (torch.Tensor): Predicted values.

    Returns:
    float: R^2 score.
    """
    ssr = (np.add(y_true, -y_pred)**2).sum(axis=0)
    tss = (np.add(y_true, -y_true.mean(axis=0)[None,:]) ** 2).sum(axis=0)
    r2 = 1 - (ssr / tss)
    return r2
