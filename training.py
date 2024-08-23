import torch
from torch import nn
from tqdm import tqdm
import matplotlib.pyplot as plt

class ModelBuilder(nn.Module):
    '''
    abstract class that lays out the framework for defining new models
    '''
    def __init__(self, options, default_options):
        super().__init__()
        self.default_options = default_options
        self.options = options
        self.verify_options()
        self.generate_network()
    
    def verify_options(self):
        '''
        makes sure all necessary options are there
        it checks the dictionary self.default_options
        to make sure all values are present and are int valued
        only put necessary options in the dictionary
        '''
        for k, v in self.default_options.items():
            try:
                if type(getattr(self.options, k)) != int:
                    ('option {k} not integer, using default value:{v}')
                    setattr(self.options, k, v)
            except AttributeError:
                print(f'option {k} missing, using default value:{v}')
                setattr(self.options, k, v)
        return
    
    #THESE TWO METHODS ARE WHAT YOU DEFINE WHEN YOU MAKE A NEW MODEL
    def generate_network(self):
        '''
    
        this generates the model architecture
        at some point it should define at least one nn.Module
        that is used in forward
        '''
        return 

    def forward(self, input):
        '''
        normal forward method, should be designed in conjunction with generate_network
        '''
        return 


class ModelTrainer:
    '''
    this is the main class, 
    it brings together a TrajectoryGenerator and a nn.Model instance
    and trains/infers from simulated trajectories
    '''
    def __init__(self, model, traj_generator, optimizer, loss_function, inference, training_options):
        '''
        model is nn.module network
        traj_generator is defined in experiments.py
        optimizer is a torch optimizer
        training options should have 
            wd, 
            lr, 
            n_iter, 
            n_epoch, how many epochs
            epoch_s, size of each epoch 
            n_iter, how many iterations each epoch
            iter_s, size of each iteration (should be < epoch_s)
            n_infer, how many batches to infer over
            infer_s, size of each inference bath

        loss function should take in (inputs, outputs, physical_params)
            in is the trajectory input, 
            out is the nn output, 
            params is a namespace containing other info to calculate the loss
        '''
        self.model = model
        self.traj_generator = traj_generator
        self.physical_params = traj_generator.params
        option_keys = ['wd','lr','n_epoch','epoch_s','n_iter','iter_s', 'n_infer', 'infer_s']
        option_vals = [1E-5, 1E-4, 10, 5_000, 10, 2_000, 1, 5_000]
        self.default_options = {k:v for k,v in zip(option_keys,option_vals)}
        self.training_options = training_options
        self.verify_options()
        self.loss_function = loss_function
        self.inference = inference
        self.optimizer = optimizer(model.parameters(),training_options.lr, weight_decay = training_options.wd)
        self.all_loss = []
        self.epoch_avg_loss = []

    def train(self, plot=False):

        for _ in tqdm( range(self.training_options.n_epoch)):
            traj_batch = self.traj_generator.batch(self.training_options.epoch_s)
            epoch_loss = self.training_epoch(traj_batch)

            self.all_loss.extend(epoch_loss)
            self.epoch_avg_loss.append( (sum(epoch_loss)/len(epoch_loss)) )
        
        if plot:
            self.plot_training_loss()

        return
    
    def training_epoch(self, traj_batch):
        loss = []
        for _ in range(self.training_options.n_iter):
            idx = torch.randperm(traj_batch.shape[0])[:self.training_options.iter_s]
            loss.append( self.training_step(traj_batch[idx]) )
        return loss

    def training_step(self, trajs):
        self.model.train()

        loss = self.loss_function(trajs, self.model(trajs), self.traj_generator)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()
    
    def plot_training_loss(self, ax = None):
            
            if ax is None:
                fig, ax = plt.subplots(1,2, sharey=True)
            else:
                assert len(ax) == 2, 'input ax must have length 2'

            L = len(self.epoch_avg_loss)
            skip = 1
            if L > 30:
                skip = int(L / 30)
                L = 30
            sz = 4 + 6 * (30 - L) / 30
            fig, ax = plt.subplots(1,2, figsize=(10,5));
            ax[0].plot(range(len(self.epoch_avg_loss))[::skip], self.epoch_avg_loss[::skip], markersize=sz, marker='o')
            ax[0].set_xlabel('epoch')
            ax[0].set_ylabel('epoch avg loss')

            ax[1].plot(self.all_loss)
            ax[1].set_xlabel('training iteration')
            ax[1].set_ylabel('all loss')
            plt.show()
            return 
    
    def infer(self, return_trajectories = False):
        self.model.eval()
        out = []
        if return_trajectories:
            test_traj = []
        for _ in range(self.training_options.n_infer):
            trajectories = self.traj_generator.batch(self.training_options.infer_s)
            with torch.no_grad():
                value = self.inference(trajectories, self.model(trajectories), self.traj_generator)
            out.append( value )
            if return_trajectories:
                test_traj.append(trajectories)
        
        if return_trajectories:
            return out, test_traj
        return out
    
    def verify_options(self):
        '''
        makes sure all necessary options are there
        it checks the dictionary self.default_options
        to make sure all values are present
        '''
        for k, v in self.default_options.items():
            try:
                getattr(self.training_options, k)
            except AttributeError:
                print(f'option {k} missing, using default value:{v}')
                setattr(self.training_options, k, v)
        return


class TrajectoryGenerator:
    def __init__(self, get_traj, params):
        '''
        this class is basically an interface from a simulation to the model
        
        get_traj must take inputs inputs of (N, params)
            N is how many trajectories to simulate
            params is be a dictionary that tells it how to simualate
                it must have at least 'dt' and 'coarse' as keys
                it also probably has 'num_steps','gamma','kBT' etc...

        get_traj returns a ndarray of trajectories
            the array has dims [N_traj , steps , coords_values]
            coord_values are typically assumed to be in the order [x_1,x_2..., v_1,v_2,...]
        '''
        self.get_traj = get_traj
        self.params = params
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 
        self.include_time = False
        self.infer_velocity = False
        self.position_only = False
    
    def batch(self, N):
        trajectories = torch.from_numpy( self.get_traj(N, self.params) )
        self.params['Dt'] = self.params['dt'] * self.params['coarse']
        if self.infer_velocity:
            trajectories = self.estimate_velocity(trajectories)
        if self.include_time:
            trajectories = self.add_time_vector(trajectories)
        if self.position_only:
            trajectories = trajectories[...,0]
        return trajectories.to(self.device).float()

    def estimate_velocity(self, trajs):
        trajs[:,:-1,1] = trajs[...,0].diff(axis=1)/self.params['Dt']
        trajs = trajs[:,:-1,]
        return trajs
    
    def add_time_vector(self, trajs):
        T = (trajs.shape[1]-1) * self.params['Dt']
        times = torch.linspace(0,T,trajs.shape[1])[None,:,None].expand(trajs.shape[0],-1,-1)
        return torch.cat([trajs, times], axis=-1)