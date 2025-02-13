from mpi4py import MPI

comm = MPI.COMM_WORLD
size = comm.Get_size()  # number of MPI prdocs
rank = comm.Get_rank()  # i.d. for local proc

import os
if rank != 0:
    os.environ["TQDM_DISABLE"] = 'True'

import torch
import numpy as np
import matplotlib.pyplot as plt
import json
import sys
from datetime import datetime
import shutil

from training import ModelTrainer, TrajectoryGenerator
from models import SingleTimeStep

from argparse import Namespace, ArgumentParser
import yaml
from experiments import force_infer, od_force_loss, od_dtlogf_loss, od_dxlogf_loss, od_sigmasq_dxlogf_loss, od_force_loss_2nd, od_entropy_loss_ML_2nd, od_dtlogf_loss_2nd
from experiments import entropy_loss_ML, entropy_infer_ML, od_entropy_loss_ML, od_entropy_infer_ML

import fivebeads

from fivebeads import simulate_five_spring_overdamped, five_beads
from fivebeads_main import multistep_train, data_save, r2_score
from mpi4py import MPI

import contextlib
import io

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
    base_dir += f'{start_time:%m_%d_%H_%M%S}/'
    os.makedirs(base_dir, exist_ok=True)
    # Simulation parameters
    params = config['simulation']
    params["num_steps"] = params["path_length"] + params["coarse_steps"][-1] + 1 
    params['init'] = np.array(params['init'])
    # training and model options
    training_options = Namespace(**config['training'])
    u_model_options = Namespace(**config['u_model'])
    dtlogf_model_options = Namespace(**config['dtlogf_model'])
    # save config
    shutil.copy(config_file, base_dir+'config.yaml')

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

print(f'rank {rank} has data, making theo model')
sys.stdout.flush()

# convert numpy array to list for json serialization
params['init'] = params['init'].tolist()

# compute the theoretical cumsum of ep
theo_model = fivebeads.five_beads(params['init'], 2 * np.linspace(params['kBT'][0], params['kBT'][1], 5))
step_begin,step_end = 0, params['path_length']
theo_path_diss_cum=theo_model.diss_path_theo_defi_cum(total_data_validate.cpu().numpy(), step_begin, step_end)
theo_final_diss_cum = theo_path_diss_cum[:,-1]



# save the theoretical cumsum of ep and data

print(f'rank {rank} has theo, saving input data')
sys.stdout.flush()

timestamp = int( 1_000*(datetime.now().timestamp() - start_time.timestamp()))
directory = base_dir+f'/{timestamp}_rank{rank}'
os.makedirs(directory, exist_ok=True)


file_path = os.path.join(directory, "theo_path_diss_cum.npy")
np.save(file_path, theo_path_diss_cum)

# save the data

theo_final_diss_cum = comm.gather(theo_final_diss_cum, root=0)
if rank == 0:
    np.save(base_dir + f'theo_final_diss', theo_final_diss_cum)

train_dir = f'{directory}/data_train.pt'
validate_dir = f'{directory}/data_validate.pt'
torch.save(total_data_train, train_dir)
torch.save(total_data_validate, validate_dir)

#cleanup memory
theo_final_diss_cum = None
theo_path_diss_cum = None


if rank == 0:
    index_dict = {}



for coarse_step in params["coarse_steps"]:
    print(f'rank {rank} initializing training for coarse = {coarse_step}')
    sys.stdout.flush()
    cname = f'_coarse_{coarse_step:02}'
    if rank == 0:
        index_dict.update({'ID'+cname:[],f'index'+cname:[]})

    params["Dt"] = params["dt"] * coarse_step
    # Set the data up
    cg_data_train = total_data_train[:,::coarse_step,:]
    cg_data_validate = total_data_validate[:,::coarse_step,:]

    # cg_num_steps is the number of nns needed
    cg_num_steps = params["path_length"]//coarse_step 
    # model generation
    WeightFunction_u_multisteps = [SingleTimeStep(u_model_options) for _ in range(cg_num_steps)]
    WeightFunction_dtlogf_multisteps = [SingleTimeStep(dtlogf_model_options) for _ in range(cg_num_steps)]

    WeightFunction_u_multisteps_2nd = [SingleTimeStep(u_model_options) for _ in range(cg_num_steps)]
    WeightFunction_dtlogf_multisteps_2nd = [SingleTimeStep(dtlogf_model_options) for _ in range(cg_num_steps)]
    # Train part initialization
    Fivebeads_multisteps = [TrajectoryGenerator(simulate_five_spring_overdamped, params) for _ in range(cg_num_steps)]

    u_multisteps  = [ModelTrainer(WeightFunction_u_multisteps[i], Fivebeads_multisteps[i], optimizer, od_entropy_loss_ML, od_entropy_infer_ML, training_options) for i in range(cg_num_steps)]
    dtlogf_multisteps  = [ModelTrainer(WeightFunction_dtlogf_multisteps[i], Fivebeads_multisteps[i], optimizer, od_dtlogf_loss, od_entropy_infer_ML, training_options) for i in range(cg_num_steps)]

    u_multisteps_2nd  = [ModelTrainer(WeightFunction_u_multisteps_2nd[i], Fivebeads_multisteps[i], optimizer, od_entropy_loss_ML_2nd, od_entropy_infer_ML, training_options) for i in range(cg_num_steps)]
    dtlogf_multisteps_2nd  = [ModelTrainer(WeightFunction_dtlogf_multisteps_2nd[i], Fivebeads_multisteps[i], optimizer, od_dtlogf_loss_2nd, od_entropy_infer_ML, training_options) for i in range(cg_num_steps)]
    # Training step by step

    print(f'rank {rank} starting training for coarse = {coarse_step}')
    sys.stdout.flush()

    if rank != 0:
        with contextlib.redirect_stdout(io.StringIO()):
            multistep_train(u_multisteps, dtlogf_multisteps, WeightFunction_u_multisteps, WeightFunction_dtlogf_multisteps, Fivebeads_multisteps, cg_data_train, cg_data_validate, coarse_step, order=1)
            multistep_train(u_multisteps_2nd, dtlogf_multisteps_2nd, WeightFunction_u_multisteps_2nd, WeightFunction_dtlogf_multisteps_2nd, Fivebeads_multisteps, cg_data_train, cg_data_validate, coarse_step, order=2)
    else:
        multistep_train(u_multisteps, dtlogf_multisteps, WeightFunction_u_multisteps, WeightFunction_dtlogf_multisteps, Fivebeads_multisteps, cg_data_train, cg_data_validate, coarse_step, order=1)
        multistep_train(u_multisteps_2nd, dtlogf_multisteps_2nd, WeightFunction_u_multisteps_2nd, WeightFunction_dtlogf_multisteps_2nd, Fivebeads_multisteps, cg_data_train, cg_data_validate, coarse_step, order=2)

    print(f'rank {rank} done with training coarse = {coarse_step}, calculating EP', flush=True)

    # Compute entropy production per trajectory
    cg_step_begin = 0
    cg_step_end = cg_num_steps
    #first order
    nn_diss_path_step_udx = theo_model.path_udx_step_nn( WeightFunction_u_multisteps, WeightFunction_dtlogf_multisteps, cg_data_validate, params, cg_step_begin, cg_step_end)
    nn_diss_path_step_dtlogf = theo_model.path_dtlogf_step_nn( WeightFunction_u_multisteps, WeightFunction_dtlogf_multisteps, cg_data_validate, params, cg_step_begin, cg_step_end)
    
    nn_path_udx_cum = nn_diss_path_step_udx.cumsum(axis=1)
    nn_path_dtlogf_cum = nn_diss_path_step_dtlogf.cumsum(axis=1)

    nn_path_udx_cum_cpu = nn_path_udx_cum.cpu().numpy()
    nn_path_dtlogf_cum_cpu = nn_path_dtlogf_cum.cpu().numpy()

    nn_path_diss_cum_cpu = np.empty ((*nn_diss_path_step_dtlogf.shape,2))

    nn_path_diss_cum_cpu[...,0] = nn_path_udx_cum_cpu
    nn_path_diss_cum_cpu[...,1] = nn_path_dtlogf_cum_cpu

    #second order
    nn_diss_path_step_udx_2nd = theo_model.path_udx_step_nn( WeightFunction_u_multisteps_2nd, WeightFunction_dtlogf_multisteps_2nd, cg_data_validate, params, cg_step_begin, cg_step_end)
    nn_diss_path_step_dtlogf_2nd = theo_model.path_dtlogf_step_nn( WeightFunction_u_multisteps_2nd, WeightFunction_dtlogf_multisteps_2nd, cg_data_validate, params, cg_step_begin, cg_step_end)

    nn_path_udx_cum_2nd = nn_diss_path_step_udx_2nd.cumsum(axis=1)
    nn_path_dtlogf_cum_2nd = nn_diss_path_step_dtlogf_2nd.cumsum(axis=1)

    nn_path_udx_cum_cpu_2nd = nn_path_udx_cum_2nd.cpu().numpy()
    nn_path_dtlogf_cum_cpu_2nd = nn_path_dtlogf_cum_2nd.cpu().numpy()

    nn_path_diss_cum_cpu_2nd = np.empty ((*nn_diss_path_step_dtlogf_2nd.shape,2))
    nn_path_diss_cum_cpu_2nd[...,0] = nn_path_udx_cum_cpu_2nd
    nn_path_diss_cum_cpu_2nd[...,1] = nn_path_dtlogf_cum_cpu_2nd

    # summary final data
    nn_final_diss_cum_cpu = nn_path_diss_cum_cpu[:,-1,:]
    nn_final_diss_cum_cpu_2nd = nn_path_diss_cum_cpu_2nd[:,-1,:]

    print(f'rank {rank} done with calculating EP, saving EP', flush=True)

    # Save data
    cg_ep_file_path = os.path.join(directory, f'nn_path_diss_cum{cname}_1st.npy')
    np.save(cg_ep_file_path, nn_path_diss_cum_cpu)

    cg_ep_file_path = os.path.join(directory, f'nn_path_diss_cum{cname}_2nd.npy')
    np.save(cg_ep_file_path, nn_path_diss_cum_cpu_2nd)

    print(f'rank {rank} done with saving EP, saving models', flush=True)
    first_order_directory = directory + f'/1storder'
    os.makedirs(first_order_directory, exist_ok=True)
    data_save(first_order_directory, params, u_multisteps, dtlogf_multisteps, WeightFunction_u_multisteps, WeightFunction_dtlogf_multisteps, cg_num_steps, coarse_step)

    second_order_directory = directory + f'/2ndorder'
    os.makedirs(second_order_directory, exist_ok=True)
    data_save(second_order_directory, params, u_multisteps_2nd, dtlogf_multisteps_2nd, WeightFunction_u_multisteps_2nd, WeightFunction_dtlogf_multisteps_2nd, cg_num_steps, coarse_step)

    print(f'rank {rank} entering gather', flush=True)
    timestamps = comm.gather([timestamp, rank], root=0)
    nn_final_diss_cum_cpu = comm.gather(nn_final_diss_cum_cpu, root=0)
    nn_final_diss_cum_cpu_2nd = comm.gather(nn_final_diss_cum_cpu_2nd, root=0)
    

    if rank ==0:
        for item in timestamps :
            index_dict[f'ID{cname}'].append(item[0])
            index_dict[f'index{cname}'].append(item[1])
        
        nn_final_diss_path = base_dir + f'nn_final_diss_cum{cname}'
        np.savez(nn_final_diss_path, first_order = nn_final_diss_cum_cpu, second_order = nn_final_diss_cum_cpu_2nd)

    print(f'rank {rank} done with saving, cleaning up', flush=True)
    #memory cleanup
    nn_final_diss_cum_cpu = None
    nn_final_diss_cum_cpu_2nd = None
    nn_path_diss_cum_cpu = None
    nn_path_diss_cum_cpu_2nd = None
    nn_path_dtlogf_cum_cpu = None
    nn_path_dtlogf_cum_cpu_2nd = None
    nn_path_udx_cum_cpu = None
    nn_path_udx_cum_cpu_2nd = None


# Save summary data
if rank == 0:
    filename = f"IDs.json"
    file_path = os.path.join(base_dir, filename)
    with open(file_path, 'w') as file:
            json.dump(index_dict, file, indent=4)

    
    









