import torch
import numpy as np
import matplotlib.pyplot as plt
import json
import os
import sys
from datetime import datetime

from argparse import Namespace
from training import ModelTrainer, TrajectoryGenerator
from models import SingleTimeStep

from experiments import force_infer, od_force_loss, od_dtlogf_loss, od_dxlogf_loss, od_sigmasq_dxlogf_loss, od_force_loss_2nd, od_entropy_loss_ML_2nd
from experiments import entropy_loss_ML, entropy_infer_ML, od_entropy_loss_ML, od_entropy_infer_ML

import fivebeads

from fivebeads import simulate_five_spring_overdamped, five_beads
from fivebeads_main import multistep_train, data_save, r2_score
from mpi4py import MPI

comm = MPI.COMM_WORLD
size = comm.Get_size()  # number of MPI prdocs
rank = comm.Get_rank()  # i.d. for local proc

#setting up base_dir and  synchronizing start time

base_dir = './parallel_tests/'
start_time = datetime.now()
if rank == 0:
    start_time = datetime.now()
    base_dir += f'{start_time:%m_%d_%H_%M%S}'
    os.makedirs(base_dir, exist_ok=True)

base_dir = comm.bcast(base_dir, root=0)
start_time = comm.bcast(start_time, root=0)

print(f'rank {rank}', base_dir, start_time)
sys.stdout.flush()
comm.barrier()
# Simulation parameters

keys = ['dt', 'num_steps', 'init', 'kBT', 'mob', 'k', 'coarse']
vals = [.01, 6, np.array([0.56,-0.23,0.14,-0.12,0.09,0.87,-0.15,0.08,-0.19,0.92,-0.21,0.11,0.68,-0.17,0.79]), [1, 2], 1, 1, 1 ]

params = { k:v for k,v in zip(keys, vals)}
# change the coarse step
params["coarse_steps"] = [1,2,3]
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
if rank ==0:
    print(f'making data')
    sys.stdout.flush()

total_data_train= TrajectoryGenerator(simulate_five_spring_overdamped, params).batch(training_options.epoch_s)
total_data_validate= TrajectoryGenerator(simulate_five_spring_overdamped, params).batch(training_options.infer_s)

print(f'rank{rank} has data, making making models and class lists')
sys.stdout.flush()

# convert numpy array to list for json serialization
params['init'] = params['init'].tolist()


# compute the theoretical cumsum of ep
theo_model = fivebeads.five_beads(params['init'], 2 * np.linspace(params['kBT'][0], params['kBT'][1], 5))
step_begin,step_end = 0, params['num_steps']-1
theo_path_diss_cum=theo_model.path_diss_path_theo_cum(total_data_validate.cpu().numpy(), step_begin, step_end)
# save the theoretical cumsum of ep and data

timestamp = int( 1_000*(datetime.now().timestamp() - start_time.timestamp()))
directory = base_dir+f'/{timestamp}'
os.makedirs(directory, exist_ok=True)

file_path = os.path.join(directory, "theo_path_diss_cum.npy")
np.save(file_path, theo_path_diss_cum)
# save the data
torch.save(total_data_train, f'{directory}/data_train.pt')
torch.save(total_data_validate, f'{directory}/data_validate.pt')

if rank == 0:
    index_dict = {}
    
for coarse_step in params["coarse_steps"]:
    cname = f'_coarse_{coarse_step}'
    if rank == 0:
        index_dict.update({'ID'+cname:[],f'index'+cname:[]})

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
    print(f'rank{rank} starting training')
    sys.stdout.flush()

    multistep_train(u_multisteps, dtlogf_multisteps, WeightFunction_u_multisteps, WeightFunction_dtlogf_multisteps, Fivebeads_multisteps, cg_data_train, cg_data_validate, coarse_step)
    # Compute entropy production per trajectory and save
    cg_step_begin = 0
    cg_step_end = cg_num_steps-1
    
    nn_diss_path_step = theo_model.path_diss_step_nn( WeightFunction_u_multisteps, WeightFunction_dtlogf_multisteps, cg_data_validate, params, cg_step_begin, cg_step_end)
    nn_path_diss_cum = nn_diss_path_step.cumsum(axis=1)
    nn_path_diss_cum_cpu = nn_path_diss_cum.cpu().numpy()
    cg_ep_file_path = os.path.join(directory, f'nn_path_diss_cum{cname}.npy')
    np.save(cg_ep_file_path, nn_path_diss_cum_cpu)
    # Save data
    
    data_save(directory, params, u_multisteps, dtlogf_multisteps, WeightFunction_u_multisteps, WeightFunction_dtlogf_multisteps, cg_num_steps, coarse_step)

    timestamps = comm.gather([timestamp, rank], root=0)

    if rank ==0:
        for item in timestamps :
            index_dict[f'ID{cname}'].append(item[0])
            index_dict[f'index{cname}'].append(item[1])

# Save summary data
if rank ==0:
    nn_params = [{**vars(item)} for item in [training_options, u_model_options, dtlogf_model_options]]
    nn_names = ['training', 'u_nn', 'dtlogf_nn']
    all_parameters = {k:v for k,v in zip(nn_names,nn_params)}
    all_parameters['sim'] = params
 
    filename = f"all_parameters.json"
    file_path = os.path.join(base_dir, filename)
    with open(file_path, 'w') as file:
        json.dump(all_parameters, file, indent=4)

    filename = f"IDs.json"
    file_path = os.path.join(base_dir, filename)
    with open(file_path, 'w') as file:
            json.dump(index_dict, file, indent=4)



    
    









