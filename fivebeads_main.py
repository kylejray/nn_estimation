import torch
import numpy as np
import matplotlib.pyplot as plt
import json
import os
from datetime import datetime

from argparse import Namespace
from training import ModelTrainer, TrajectoryGenerator
from models import SingleTimeStep

from experiments import force_infer, od_force_loss, od_dtlogf_loss, od_dxlogf_loss, od_sigmasq_dxlogf_loss, od_force_loss_2nd, od_entropy_loss_ML_2nd
from experiments import entropy_loss_ML, entropy_infer_ML, od_entropy_loss_ML, od_entropy_infer_ML

import fivebeads

from fivebeads import simulate_five_spring_overdamped, five_beads
#from free_diffusion import params, simulate_free_diffusion_underdamped, realepr, traj_dEnt


############################################

def multistep_train(u_multisteps, dtlogf_multisteps, WeightFunction_u_multisteps, WeightFunction_dtlogf_multisteps, Fivebeads_multisteps, cg_data_train, cg_data_validate, coarse_step):
    for step in range(len(u_multisteps)):

    # Setting up training data and validation data
        Fivebeads_multisteps[step].infinite_data = False
        Fivebeads_multisteps[step].data = cg_data_train[:,step:step+2,:]
        u_multisteps[step].validation_data = cg_data_validate[:,step:step+2,:]
        dtlogf_multisteps[step].validation_data = cg_data_validate[:,step:step+2,:] 
        #print(hasattr(EntProd_ML_multisteps[step], 'validation_data'))  # Check if 'data' exists
     # Training
        print(f'Training u step {step} using coarse step {coarse_step}')
        u_multisteps[step].train()
        print(f'Training dtlogf step {step} using coarse step {coarse_step}')
        dtlogf_multisteps[step].train()
        ##Reset parameter every 10 steps
        #if (step+1)%10 !=0:
    # Load parameters from the previous step except for the last step
        try:
            WeightFunction_u_multisteps[step+1].load_state_dict(WeightFunction_u_multisteps[step].state_dict())
            WeightFunction_dtlogf_multisteps[step+1].load_state_dict(WeightFunction_dtlogf_multisteps[step].state_dict())
        except:
            pass

def data_save(directory, params, u_multisteps, dtlogf_multisteps, WeightFunction_u_multisteps, WeightFunction_dtlogf_multisteps, cg_num_steps, coarse_step):
    u_training_loss=[]
    dtlogf_training_loss=[]
    u_validation_loss=[]
    dtlogf_validation_loss=[]

    for step in range(cg_num_steps-1):
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

if __name__ == "__main__":
    # Optimizer
    optimizer = torch.optim.Adam
    # Simulation parameters
    keys = ['dt', 'num_steps', 'init', 'kBT', 'mob', 'k', 'coarse']
    vals = [.01, 6, np.array([0.56,-0.23,0.14,-0.12,0.09,0.87,-0.15,0.08,-0.19,0.92,-0.21,0.11,0.68,-0.17,0.79]), [1, 2], 1, 1, 1 ] 

    params = { k:v for k,v in zip(keys, vals)}
    # change the coarse step
    params["coarse_steps"] = [1,2,4,5]
    ############################################    
    

    # Training options for the model    

    training_options = Namespace()
    u_model_options = Namespace()
    dtlogf_model_options = Namespace()  

    training_options.n_epoch = 1_500
    training_options.epoch_s = 8_000    

    training_options.n_iter = 2
    training_options.iter_s = 4_096 

    training_options.n_infer = 1
    training_options.infer_s = 2_000    

    training_options.lr = 1E-4
    training_options.wd = 5E-5  

    training_options.patience = 5
    training_options.min_delta = 0  

    #Earlystop
    training_options.patience = 5
    training_options.min_delta = 0  
    

    # Model options
    u_model_options.n_input = 5
    u_model_options.n_hidden = 32
    u_model_options.n_output = 5
    u_model_options.num_inner = 2   

    dtlogf_model_options.n_input = 5
    dtlogf_model_options.n_hidden = 32
    dtlogf_model_options.n_output = 1
    dtlogf_model_options.num_inner = 2

    # synthetic data generation
    total_data_train= TrajectoryGenerator(simulate_five_spring_overdamped, params).batch(training_options.epoch_s)
    total_data_validate= TrajectoryGenerator(simulate_five_spring_overdamped, params).batch(training_options.infer_s)

    # convert numpy array to list for json serialization
    params['init'] = params['init'].tolist()
    timestamp = datetime.now().strftime("%m-%d_%H-%M-%S")

    # compute the theoretical cumsum of ep
    theo_model = fivebeads.five_beads(params['init'], 2 * np.linspace(params['kBT'][0], params['kBT'][1], 5))
    step_begin,step_end = 0, params['num_steps']-1
    theo_path_diss_cum=theo_model.path_diss_path_theo_cum(total_data_validate.cpu().numpy(), step_begin, step_end)
    # save the theoretical cumsum of ep and data
    directory = f'results_{timestamp}'
    os.makedirs(directory, exist_ok=True)
    file_path = os.path.join(directory, "theo_path_diss_cum.npy")
    np.save(file_path, theo_path_diss_cum)
    # save the data
    torch.save(total_data_train, f'{directory}/data_train.pt')
    torch.save(total_data_validate, f'{directory}/data_validate.pt')

    meta_parameters = {
        **vars(training_options),
        **vars(u_model_options),
        **vars(dtlogf_model_options)
    }
    filename = f"nn_parameters.json"
    file_path = os.path.join(directory, filename)
    with open(file_path, 'w') as file:
        json.dump(meta_parameters, file, indent=4)


    # start coase grained training


    for coarse_step in params["coarse_steps"]:

        params["Dt"] = params["dt"] * coarse_step
        print(f'Coarse step {coarse_step}')
        # Set the data up
        cg_data_train = total_data_train[:,::coarse_step,:]
        cg_data_validate = total_data_validate[:,::coarse_step,:]
        cg_num_steps = cg_data_train.shape[1]
        # model generation
        WeightFunction_u_multisteps = [SingleTimeStep(u_model_options) for _ in range(cg_num_steps-1)]
        WeightFunction_dtlogf_multisteps = [SingleTimeStep(dtlogf_model_options) for _ in range(cg_num_steps-1)]
        # Train part initialization
        Fivebeads_multisteps = [TrajectoryGenerator(simulate_five_spring_overdamped, params) for _ in range(cg_num_steps-1)]
        u_multisteps  = [ModelTrainer(WeightFunction_u_multisteps[i], Fivebeads_multisteps[i], optimizer, od_entropy_loss_ML, od_entropy_infer_ML, training_options) for i in range(cg_num_steps-1)]
        dtlogf_multisteps  = [ModelTrainer(WeightFunction_dtlogf_multisteps[i], Fivebeads_multisteps[i], optimizer, od_dtlogf_loss, od_entropy_infer_ML, training_options) for i in range(cg_num_steps-1)]
        # Training step by step
        multistep_train(u_multisteps, dtlogf_multisteps, WeightFunction_u_multisteps, WeightFunction_dtlogf_multisteps, Fivebeads_multisteps, cg_data_train, cg_data_validate, coarse_step)
        # Compute entropy production per trajectory and save
        cg_step_begin = 0
        cg_step_end = cg_num_steps-1

        nn_diss_path_step = theo_model.path_diss_step_nn( WeightFunction_u_multisteps, WeightFunction_dtlogf_multisteps, cg_data_validate, params, cg_step_begin, cg_step_end)
        nn_path_diss_cum = nn_diss_path_step.cumsum(axis=1)
        nn_path_diss_cum_cpu = nn_path_diss_cum.cpu().numpy()
        cg_ep_file_path = os.path.join(directory, f'nn_path_diss_cum_coarse_{coarse_step}.npy')
        np.save(cg_ep_file_path, nn_path_diss_cum_cpu)
        # Save data
        data_save(directory, params, u_multisteps, dtlogf_multisteps, WeightFunction_u_multisteps, WeightFunction_dtlogf_multisteps, cg_num_steps, coarse_step)
