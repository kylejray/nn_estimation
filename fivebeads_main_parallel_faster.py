from mpi4py import MPI

comm = MPI.COMM_WORLD
size = comm.Get_size()  # number of MPI prdocs
rank = comm.Get_rank()  # i.d. for local proc

import os
if rank != 0:
    os.environ["TQDM_DISABLE"] = 'True'

rank_pid = os.getpid()

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
from fivebeads_main_kyle import multistep_train, r2_score
from mpi4py import MPI

import contextlib
import io
import gc

#setting up option variables on all ranks
base_dir = ''
start_time = datetime.now()
params = {}
training_options = Namespace()
u_model_options = Namespace()
dtlogf_model_options = Namespace ()

dynamics = simulate_five_spring_overdamped
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
options_list = [u_model_options, dtlogf_model_options, training_options]


# Optimizer
optimizer = torch.optim.Adam

#loss functions
loss_1st =  [od_entropy_loss_ML, od_dtlogf_loss]
loss_2nd = [od_entropy_loss_ML_2nd, od_dtlogf_loss_2nd]
############################################

# synthetic data generation
if rank ==0:
    print(f'making data')
    sys.stdout.flush()

total_data_train= TrajectoryGenerator(simulate_five_spring_overdamped, params).batch(training_options.epoch_s)
total_data_validate= TrajectoryGenerator(simulate_five_spring_overdamped, params).batch(training_options.infer_s)

print(f'rank {rank} has data, making theo model', flush=True)

# convert numpy array to list for json serialization
params['init'] = params['init'].tolist()

# compute the theoretical cumsum of ep
theo_model = fivebeads.five_beads(params['init'], 2 * np.linspace(params['kBT'][0], params['kBT'][1], 5))
step_begin,step_end = 0, params['path_length']
theo_path_diss_cum=theo_model.diss_path_theo_defi_cum(total_data_validate.cpu().numpy(), step_begin, step_end)
theo_final_diss_cum = theo_path_diss_cum[:,-1]



# save the theoretical cumsum of ep and data

print(f'rank {rank} has theo, saving input data', flush=True)

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
    print(f'pid {rank_pid} rank {rank} initializing training for coarse = {coarse_step}', flush=True)
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

    for loss_order, loss, order_string in zip([1,2],[loss_1st,loss_2nd],['1st','2nd']):
        gc.collect()
        torch.mps.empty_cache()

        order_directory = directory + f'/{order_string}order'
        os.makedirs(order_directory, exist_ok=True)

        #print(torch.cuda.memory_summary())
        print(f'rank{rank}:{torch.mps.current_allocated_memory()/2**30:.3f}GB, {torch.mps.driver_allocated_memory()/2**30:.3f}GB', flush=True)
 
        # Training step by step
        print(f'rank {rank} starting training for coarse = {coarse_step}, order={loss_order}', flush=True)
        if rank != 0:
            with contextlib.redirect_stdout(io.StringIO()):
                trained_networks = multistep_train(options_list, optimizer, loss[0], loss[1], od_entropy_infer_ML, dynamics, params, cg_num_steps, cg_data_train, cg_data_validate, coarse_step, order_directory, order=loss_order)
        else:
            trained_networks = multistep_train(options_list, optimizer, loss[0], loss[1], od_entropy_infer_ML, dynamics, params, cg_num_steps, cg_data_train, cg_data_validate, coarse_step, order_directory, order=loss_order)

        print(f'rank {rank} done with training and saving coarse = {coarse_step}, order={loss_order}; calculating EP', flush=True)

        # Compute entropy production per trajectory
        cg_step_begin = 0
        cg_step_end = cg_num_steps
    
        nn_diss_path_step_udx = theo_model.path_udx_step_nn( *trained_networks, cg_data_validate, params, cg_step_begin, cg_step_end).detach()
        nn_diss_path_step_dtlogf = theo_model.path_dtlogf_step_nn( *trained_networks, cg_data_validate, params, cg_step_begin, cg_step_end).detach()

        nn_path_udx_cum = nn_diss_path_step_udx.cumsum(axis=1)
        nn_path_dtlogf_cum = nn_diss_path_step_dtlogf.cumsum(axis=1)

        nn_path_udx_cum_cpu = nn_path_udx_cum.cpu().numpy()
        nn_path_dtlogf_cum_cpu = nn_path_dtlogf_cum.cpu().numpy()

        nn_path_diss_cum_cpu = np.empty ((*nn_diss_path_step_dtlogf.shape,2))

        nn_path_diss_cum_cpu[...,0] = nn_path_udx_cum_cpu
        nn_path_diss_cum_cpu[...,1] = nn_path_dtlogf_cum_cpu

                # Save data
        cg_ep_file_path = os.path.join(directory, f'nn_path_diss_cum{cname}_{order_string}.npy')
        np.save(cg_ep_file_path, nn_path_diss_cum_cpu)

        # summary final data
        if loss_order == 1:
            nn_final_diss_cum_cpu = nn_path_diss_cum_cpu[:,-1,:]
        if loss_order == 2:
            nn_final_diss_cum_cpu_2nd = nn_path_diss_cum_cpu[:,-1,:]


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

    #print(f'rank {rank} done with saving, cleaning up', flush=True)
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

    
    









