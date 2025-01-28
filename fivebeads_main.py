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


# Simulation parameters

keys = ['dt', 'num_steps', 'init', 'kBT', 'mob', 'k', 'coarse']
vals = [.01, 6, np.array([0.56,-0.23,0.14,-0.12,0.09,0.87,-0.15,0.08,-0.19,0.92,-0.21,0.11,0.68,-0.17,0.79]), [1, 2], 1, 1, 1 ]

params = { k:v for k,v in zip(keys, vals)}
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

# Optimizer
optimizer = torch.optim.Adam
############################################

# synthetic data generation
total_data_train= TrajectoryGenerator(simulate_five_spring_overdamped, params).batch(training_options.epoch_s)
total_data_validate= TrajectoryGenerator(simulate_five_spring_overdamped, params).batch(training_options.infer_s)

# model generation
WeightFunction_u_multisteps = [SingleTimeStep(u_model_options) for _ in range(params['num_steps'])]
WeightFunction_dtlogf_multisteps = [SingleTimeStep(dtlogf_model_options) for _ in range(params['num_steps'])]

############################################

# Train part initialization

Fivebeads_multisteps = [TrajectoryGenerator(simulate_five_spring_overdamped, params) for _ in range(params['num_steps'])]
u_multisteps  = [ModelTrainer(WeightFunction_u_multisteps[0], Fivebeads_multisteps[0], optimizer, od_entropy_loss_ML, od_entropy_infer_ML, training_options) for _ in range(params['num_steps'])]
dtlogf_multisteps  = [ModelTrainer(WeightFunction_dtlogf_multisteps[0], Fivebeads_multisteps[0], optimizer, od_dtlogf_loss, od_entropy_infer_ML, training_options) for _ in range(params['num_steps'])]

# Training step by step
for step in range(params['num_steps']-1):
    # Set up the training part
    u_multisteps[step] = ModelTrainer(WeightFunction_u_multisteps[step], Fivebeads_multisteps[step], optimizer, od_entropy_loss_ML, od_entropy_infer_ML, training_options)
    dtlogf_multisteps[step] = ModelTrainer(WeightFunction_dtlogf_multisteps[step], Fivebeads_multisteps[step], optimizer, od_dtlogf_loss, od_entropy_infer_ML, training_options)

    # Setting up training data and validation data
    Fivebeads_multisteps[step].infinite_data = False
    if step!=params['num_steps']-2:
        Fivebeads_multisteps[step].data = total_data_train[:,step:step+2,:]
        u_multisteps[step].validation_data = total_data_validate[:,step:step+2,:]
        dtlogf_multisteps[step].validation_data = total_data_validate[:,step:step+2,:] 
    if step==params['num_steps']-2:
        Fivebeads_multisteps[step].data = total_data_train[:,-2:,:]
        u_multisteps[step].validation_data = total_data_validate[:,-2:,:] 
        dtlogf_multisteps[step].validation_data = total_data_validate[:,-2:,:] 
    #print(hasattr(EntProd_ML_multisteps[step], 'validation_data'))  # Check if 'data' exists
    
    # Training
    print(f'Training u step {step}')
    u_multisteps[step].train()
    print(f'Training dtlogf step {step}')
    dtlogf_multisteps[step].train()
    ##Reset parameter every 10 steps
    #if (step+1)%10 !=0:
    WeightFunction_u_multisteps[step+1].load_state_dict(WeightFunction_u_multisteps[step].state_dict())
    WeightFunction_dtlogf_multisteps[step+1].load_state_dict(WeightFunction_dtlogf_multisteps[step].state_dict())

    
# Compute entropy production for each path
theo_model = fivebeads.five_beads(params['init'], 2 * np.linspace(params['kBT'][0], params['kBT'][1], 5))
# Specify the start and the end
step_begin, step_end=0, params['num_steps']-1

#diss_path = theo_model.path_diss_path_theo(total_data_validate.cpu().numpy(), step_begin, step_end)
#diss_path_nn = theo_model.path_diss_nn( WeightFunction_u_multisteps, WeightFunction_dtlogf_multisteps, total_data_validate, step_begin, step_end)
diss_path_step = theo_model.path_diss_step_nn( WeightFunction_u_multisteps, WeightFunction_dtlogf_multisteps, total_data_validate, step_begin, step_end)
nn_cumsum_diss = diss_path_step.cumsum(axis=1)
nn_cumsum_diss_cpu = nn_cumsum_diss.cpu().numpy()

theo_cumsum_diss = theo_model.path_diss_path_theo_cum(total_data_validate.cpu().numpy(), step_begin, step_end)
 


# Save data
params['init'] = params['init'].tolist()
parameters = {
    **params,
    **vars(training_options),
    **vars(u_model_options),
    **vars(dtlogf_model_options)
}

u_training_loss=[]
dtlogf_training_loss=[]
u_validation_loss=[]
dtlogf_validation_loss=[]

for step in range(params['num_steps']-1):
    u_training_loss.append(u_multisteps[step].all_loss)
    dtlogf_training_loss.append(dtlogf_multisteps[step].all_loss)
    u_validation_loss.append(u_multisteps[step].epoch_validation_loss)
    dtlogf_validation_loss.append(dtlogf_multisteps[step].epoch_validation_loss)

data={'u_training_loss':u_training_loss, 'dtlogf_training_loss':dtlogf_training_loss, 'u_validation_loss':u_validation_loss, 'dtlogf_validation_loss':dtlogf_validation_loss}

timestamp = datetime.now().strftime("%m-%d_%H-%M-%S")
directory = f'results_{timestamp}'
os.makedirs(directory, exist_ok=True)

model_directory = directory + '/models'
os.makedirs(model_directory, exist_ok=True)

filename = f"loss.json"
file_path = os.path.join(directory, filename)
with open(file_path, 'w') as file:
    json.dump(data, file, indent=4)

filename = f"parameters.json"
file_path = os.path.join(directory, filename)
with open(file_path, 'w') as file:
    json.dump(parameters, file, indent=4)

for idx, model_u in enumerate(WeightFunction_u_multisteps):
    torch.save(model_u.state_dict(), f'{model_directory}/model_u_{idx}.pth')

for idx, model_dtlogf in enumerate(WeightFunction_dtlogf_multisteps):
    torch.save(model_dtlogf.state_dict(), f'{model_directory}/model_dtlogf_{idx}.pth')

file_path = os.path.join(directory, "nn_cumsum_diss.npy")
np.save(file_path, nn_cumsum_diss_cpu)
file_path = os.path.join(directory, "theo_cumsum_diss.npy")
np.save(file_path, theo_cumsum_diss)