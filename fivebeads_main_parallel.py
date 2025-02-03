import torch
import numpy as np
import matplotlib.pyplot as plt
import json
import os
import sys
from datetime import datetime
import shutil

from training import ModelTrainer, TrajectoryGenerator
from models import SingleTimeStep

from argparse import Namespace, ArgumentParser
import yaml
from experiments import force_infer, od_force_loss, od_dtlogf_loss, od_dxlogf_loss, od_sigmasq_dxlogf_loss, od_force_loss_2nd, od_entropy_loss_ML_2nd
from experiments import entropy_loss_ML, entropy_infer_ML, od_entropy_loss_ML, od_entropy_infer_ML

import fivebeads

from fivebeads import simulate_five_spring_overdamped, five_beads
from fivebeads_main import multistep_train, data_save, r2_score
from mpi4py import MPI

comm = MPI.COMM_WORLD
size = comm.Get_size()  # number of MPI prdocs
rank = comm.Get_rank()  # i.d. for local proc

#setting up option variables on all ranks
base_dir = ''
start_time = datetime.now()
params = {}
training_options = Namespace()
u_model_options = Namespace()
dtlogf_model_options = Namespace ()

#grab config from YAML on rank 0
if rank == 0:
    parser = ArgumentParser()
    parser.add_argument('config_path', type=str, help='name of .yaml config')
    config_file = parser.parse_args().config_path

    start_time = datetime.now()
    with open(config_file, "r") as f:
        config = yaml.safe_load(f)
    base_dir = config['base_directory']
    base_dir += f'{start_time:%m_%d_%H_%M%S}'
    os.makedirs(base_dir, exist_ok=True)
    # Simulation parameters
    params = config['simulation']
    params['init'] = np.array(params['init'])
    #training and model options
    training_options = Namespace(**config['training'])
    u_model_options = Namespace(**config['u_model'])
    dtlogf_model_options = Namespace(**config['dtlogf_model'])

#broadcast to other ranks
base_dir = comm.bcast(base_dir, root=0)
start_time = comm.bcast(start_time, root=0)
params = comm.bcast(params, root=0)
training_options = comm.bcast(training_options, root=0)
u_model_options = comm.bcast(u_model_options, root=0)
dtlogf_model_options = comm.bcast(dtlogf_model_options, root=0)


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
directory = base_dir+f'/{timestamp}_rank{rank}'
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
if rank == 0:
    shutil.copy(config_file, base_dir+'/config.yaml')

    filename = f"IDs.json"
    file_path = os.path.join(base_dir, filename)
    with open(file_path, 'w') as file:
            json.dump(index_dict, file, indent=4)



    
    









