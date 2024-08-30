from training import ModelBuilder
import torch
from torch import nn


# simple network for rate based estimation
# also serves as an example for making new models
class SingleTimeStep(ModelBuilder):
    def __init__(self, options):
        default_options = {'n_input':2, 'n_output':1, 'n_hidden':512, 'num_inner':2}
        super().__init__(options, default_options)

    def generate_network(self):
        opt = self.options
        self.h = fully_connected_linear(
            opt.n_input, opt.n_output, opt.n_hidden, opt.num_inner,
            )
        return 

    def forward(self, s):
        return self.h(s)

# here is using two linear networks to also collapse all trajectories to one number
# potentially for testing the TUT or the oneshot method
class FullTrajectory(ModelBuilder):
    def __init__(self, options):
        default_options = {'n_input':2, 'n_output':1, 'n_hidden':512, 'num_inner':2, 'traj_length':10}
        super().__init__(options, default_options)
    
    def generate_network(self):
        opt = self.options

        self.h0 = fully_connected_linear(
            opt.n_input, opt.n_output, opt.n_hidden, opt.num_inner,
            )

        self.h1 = fully_connected_linear(
            opt.traj_length, opt.n_output, opt.n_hidden, opt.num_inner,
            )
        return

    def forward(self, s):
        w_coords = self.h0(s).squeeze()
        w = self.h1(w_coords)
        return h1


def fully_connected_linear(n_input, n_output, n_hidden, num_inner):
    '''
    helper function to make fully connected linear/relu layers
    '''
    linear_layers = [nn.Linear(n_input, n_hidden),
                    *[nn.Linear(n_hidden, n_hidden) for i in range(num_inner)],
                    nn.Linear(n_hidden, n_output),
                    ]
    relu_layers = [nn.ReLU(inplace=True) for i in range(len(linear_layers)-1)]
    layers = [None]*(2*len(linear_layers)-1)
    layers[::2] = linear_layers
    layers[1::2] = relu_layers
    return nn.Sequential(*layers)

class TimeOddCurrent(FullTrajectory):
    '''
    return a time odd function of a full trajectory
    1D only right now
    '''

    def __init__(self, options):
        super().__init__(options)
    
    def one_pass(self, s):
        h0 = self.h0(s).squeeze()
        return self.h1(h0)
        
    def forward(self, s):
        s_rev = torch.fliplr(s)
        s_rev[...,1] *= -1
        return self.one_pass(s) - self.one_pass(s_rev)




