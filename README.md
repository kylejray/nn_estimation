# MPI-based Stochastic Thermodynamics Inference

This project implements a parallel training pipeline for inferring thermodynamic quantities (e.g., stochastic entropy production) from trajectories using neural networks.

It uses:
- MPI (`mpi4py`) for parallelization
- PyTorch for model training
- YAML for experiment configuration



# Minimal example

To run the minimal example
```bash
mpirun -n x python fivebeads_main_parallel.py config.yaml 
```
where x is the number of the parralel training. For example, run with 4 processes:

```bash
mpirun -n 4 python main.py config.yaml
```

The script requires a YAML config file `config.yaml`, which looks like

```yaml
---
base_directory: './testing_mpi/'

simulation:
  dt: .01
  path_length: 100
  init: [0.56,-0.23,0.14,-0.12,0.09,0.87,-0.15,0.08,-0.19,0.92,-0.21,0.11,0.68,-0.17,0.79]
  kBT: [1, 2]
  mob:  1
  k: 1
  coarse: 1
  coarse_steps: [1,2,3]

# Training options for the model
training:
  n_epoch: 10
  epoch_s: 8_000
  n_iter: 2
  iter_s: 4_096 
  n_infer: 1
  infer_s: 2_000 
  lr: .0001
  wd: .00001  
  patience: 5
  min_delta: 0
# Model options

u_model:
  n_input: 5
  n_hidden: 32
  n_output: 5
  num_inner: 2

dtlogf_model:
  n_input: 5
  n_hidden: 32
  n_output: 1
  num_inner: 2
```


# Parameters

## `simulation`

Controls simulation parameters.

* **`dt`**: Simulation time step size
* **`path_length`**: Number of simulation steps per trajectory
* **`init`**: Initial state vector
  * Here: the right triangle part of the covariance of the initial Gaussian distribution $(\Sigma_{11}, ..., \Sigma_{15}, \Sigma_{22}, ..., \Sigma_{25},\Sigma_{33}, ...,\Sigma_{35}, ..., \Sigma_{55})$
* **`kBT`**: Boltzmann constant * Temperature 
* **`mob`**: Mobility coefficient (Not used)
* **`k`**: Spring constant
* **`coarse`**: 
* **`coarse_steps`**: List of coarse-graining levels



Here, for example, the following data are sent to neural networks:

* 1 → full trajectory
* 2 → every 2 steps
* 3 → every 3 steps



## `training`

Controls neural network training.

### Core parameters

* **`n_epoch`**: Number of epochs generated
* **`epoch_s`**: Samples per epoch generated
* **`n_iter`**: Iterations per training stage
* **`iter_s`**: Batch size
* **`n_infer`**: Number of inference runs (not used)
* **`infer_s`**: Validation dataset size

### Optimization

* **`lr`**: Learning rate
* **`wd`**: Weight decay

### Early stopping

* **`patience`**: Stop if no improvement after N steps
* **`min_delta`**: Minimum improvement threshold


##  `u_model`

Neural network for **local entropy production**

```yaml
u_model:
  n_input: 5
  n_hidden: 32
  n_output: 5
  num_inner: 2
```

* **`n_input`**: Input dimension
* **`n_hidden`**: Hidden size
* **`n_output`**: Output dimension
* **`num_inner`**: Number of hidden layers



## `dtlogf_model`

Neural network for **temporal score function**

```yaml
dtlogf_model:
  n_input: 5
  n_hidden: 32
  n_output: 1
  num_inner: 2
```

# Results
After running the experiment, the results directory will look like:
```bash
results/<timestamp>/
├── 1883_rank1/
├── 1889_rank0/
├── config.yaml
├── IDs.json
├── nn_final_diss_cum_coarse_01.npz
├── nn_final_diss_cum_coarse_02.npz
├── nn_final_diss_cum_coarse_03.npz
└── theo_final_diss.npy
```



##  Neural Network Results

```bash
nn_final_diss_cum_coarse_01.npz
nn_final_diss_cum_coarse_02.npz
nn_final_diss_cum_coarse_03.npz
```

Each file corresponds to a **coarse-graining level**.

### Contents:

```python
np.load(file)
```

Returns:

* `first_order`: cumulative stochastic entropy production (1st-order estimator)
* `second_order`: cumulative stochastic entropy production (2nd-order estimator)

More details can be found in the jupyter notebook `nn_od_example_notebook.ipynb`.

# Citation

If you use this code, please cite:

Lyu, J., Ray, K. J., & Crutchfield, J. P. (2025).  
*Learning Stochastic Thermodynamics Directly from Correlation and Trajectory-Fluctuation Currents*.  
arXiv:2504.19007 (Accepted by PRE)

https://arxiv.org/abs/2504.19007
