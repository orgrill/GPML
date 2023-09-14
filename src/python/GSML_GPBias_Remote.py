# -*- coding: utf-8 -*-
"""
Designed to load training objects and test objects built from Matlab,
has a flag to determine if we want to do training
"""

import torch
import gpytorch
import gc


class GPRegressionModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(GPRegressionModel, self).__init__(train_x, train_y, likelihood)
        # SKI requires a grid size hyperparameter. This util can help with that
        #grid_size = gpytorch.utils.grid.choose_grid_size(train_x)
        #grid_size = gpytorch.utils.grid.choose_grid_size(train_x[:,1], kronecker_structure=False)
        #grid_size = gpytorch.utils.grid.choose_grid_size(train_x)
        grid_bounds = [(xmin[0][0], xmax[0][0]), (xmin[0][1], xmax[0][1])]
        grid_bounds_1D = [(xmin[0][0], xmax[0][0])] # Needs to include all the ranges for elevation and time!
        
        #self.mean_module = gpytorch.means.ZeroMean()
        #self.mean_module = gpytorch.means.LinearMean(input_size=torch.tensor(2))
        self.mean_module = gpytorch.means.ConstantMean(input_size=torch.tensor(2))
        #
        # self.covar_module = gpytorch.kernels.AdditiveStructureKernel(
        # # #self.covar_module = gpytorch.kernels.ProductStructureKernel(
        #      gpytorch.kernels.GridInterpolationKernel(
        # #         #gpytorch.kernels.SpectralMixtureKernel(num_mixtures=10), grid_size=16000, grid_bounds = grid_bounds, num_dims=1
        # #         #gpytorch.kernels.SpectralMixtureKernel(num_mixtures=6), grid_size=100000, grid_bounds = grid_bounds, num_dims=1
        #          gpytorch.kernels.PeriodicKernel(), grid_size=100000, grid_bounds = grid_bounds, num_dims=1
        # ), num_dims=2
        # )
        #
        
        #Product Structure Kernel, Periodic
        #self.covar_module = gpytorch.kernels.ProductStructureKernel(
        #     gpytorch.kernels.GridInterpolationKernel(
        #          gpytorch.kernels.PeriodicKernel(), grid_size=5000, grid_bounds = grid_bounds_1D, num_dims=1), num_dims=2)
        
    # Product Structure Kernel, Spectral Mixture Kernels
        #self.covar_module = gpytorch.kernels.ProductStructureKernel(
        #gpytorch.kernels.GridInterpolationKernel(
        #     gpytorch.kernels.SpectralMixtureKernel(num_mixtures=10), grid_size = 100000, grid_bounds = grid_bounds, num_dims=1), num_dims = 2)
        
    # Product of a Periodic in a grid to a linear, this is pretty memory intensive
        #self.covar_module =  gpytorch.kernels.GridInterpolationKernel(
        #      gpytorch.kernels.PeriodicKernel(), grid_size=50000, grid_bounds = gb1, num_dims=1, active_dims = 0)*gpytorch.kernels.LinearKernel(active_dims = 1)        
        
    # Grid of a product of a periodic in time and spectral mixture in elevation
        #self.covar_module =  gpytorch.kernels.GridInterpolationKernel(
        #      (gpytorch.kernels.PeriodicKernel(ard_num_dims=2)*gpytorch.kernels.SpectralMixtureKernel(num_mixtures=10,active_dims=torch.tensor([1]))), grid_size=[100000, 50], grid_bounds = grid_bounds, num_dims=2)        
        
    # Grid of a product of a periodic in time and rbf in elevation
    #    self.covar_module =  gpytorch.kernels.GridInterpolationKernel(
    #          gpytorch.kernels.RBFKernel(active_dims=1)*gpytorch.kernels.PeriodicKernel(), grid_size=[50000, 100], grid_bounds = grid_bounds, num_dims=2)        

    # 2D version of what worked well in 1D
        #self.covar_module =  gpytorch.kernels.GridInterpolationKernel(
        #        gpytorch.kernels.ScaleKernel(gpytorch.kernels.PeriodicKernel()*gpytorch.kernels.RQKernel(active_dims = 0))+(gpytorch.kernels.RQKernel(active_dims = 1)), grid_size=[10000, 50], grid_bounds = grid_bounds, num_dims=2)
    # 2D version of what worked well in 1D
        self.covar_module =  gpytorch.kernels.GridInterpolationKernel(
                gpytorch.kernels.ScaleKernel(gpytorch.kernels.PeriodicKernel()*gpytorch.kernels.RQKernel(active_dims = 0))*gpytorch.kernels.ScaleKernel(gpytorch.kernels.RQKernel()), grid_size=[10000, 100], grid_bounds = grid_bounds, num_dims=2)    
    # Product of two grids periodic in time and spectral mixture in elevation    
    #    self.covar_module =  gpytorch.kernels.GridInterpolationKernel(
    #          gpytorch.kernels.PeriodicKernel(),grid_size=10000, grid_bounds = gb1, active_dims=torch.tensor([0]))*gpytorch.kernels.GridInterpolationKernel(gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel()),grid_size=100, grid_bounds = gb2, active_dims=torch.tensor([1])) 
        
        # I never got this to work
        #self.covar_module.base_kernel.base_kernel.initialize_from_data(test_x[:,1], train_y)
    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


gc.collect()
torch.cuda.empty_cache() 

def Run(train_x, train_y, test_x, TrainFlag):
    train_x = torch.tensor(train_x, dtype=torch.float)
    train_y = torch.tensor(train_y, dtype=torch.float).flatten()
    test_x = torch.tensor(test_x, dtype=torch.float)

    mean = train_x.mean(dim=-2, keepdim=True)
    std = train_x.std(dim=-2, keepdim=True) + 1e-6 # prevent dividing by 0
    train_x = (train_x - mean) / std
    test_x = (test_x - mean) / std

    # normalize labels
    meany, stdy = train_y.mean(),train_y.std()
    train_y = (train_y - meany) / stdy

    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    model = GPRegressionModel(train_x, train_y, likelihood).cuda()
    if TrainFlag == 1:
        hypers = {
        'likelihood.noise_covar.noise': torch.tensor(10),
        }
        model.initialize(**hypers)
    else:
        state_dict = torch.load('MyGSML_GPSKI.pth')
        model.load_state_dict(state_dict)
    
    likelihood.train()
    if torch.cuda.is_available():
        model = model.cuda()
        likelihood = likelihood.cuda()   

    if TrainFlag == 1:   
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.5)  # Includes GaussianLikelihood parameters
        # # "Loss" for GPs - the marginal log likelihood
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
        training_iterations = 15
        def train():
            for i in range(training_iterations):
                optimizer.zero_grad()
                #with  gpytorch.settings.max_root_decomposition_size(50):
                output = model(train_x.cuda())
                loss = -mll(output, train_y.cuda())
                loss.backward()
                optimizer.step()
                print('Iter %d/%d - Loss: %d  noise: %.3f ' % (i + 1, training_iterations, loss.item(),model.likelihood.noise.item()))
                torch.cuda.empty_cache()
        train()

    # Set model and likelihood into evaluation mode
    model.eval()
    likelihood.eval()
    
    if torch.cuda.is_available():
        test_x = test_x.cuda()
        
    #with gpytorch.beta_features.checkpoint_kernel(1000), gpytorch.settings.max_preconditioner_size(10), torch.no_grad():
    #    with gpytorch.settings.max_root_decomposition_size(50), gpytorch.settings.fast_pred_var():
    with torch.no_grad(), gpytorch.settings.fast_pred_var():          
            #pred_labels = model(test_x)
            observed_pred = likelihood(model(test_x))
            pred_labels = observed_pred.mean
    #with torch.no_grad(), gpytorch.settings.fast_pred_var():
        #observed_pred = likelihood(model(test_x))
        #pred_labels = observed_pred.mean.view(n, n)
        #pred_labels = observed_pred.mean

    # Undo All the Z normalizing
    pred_labels = ((pred_labels.cpu() * stdy) + meany).numpy()
    train_y = ((train_y.cpu() * stdy) + meany).numpy()
    train_x = ((train_x.cpu() * std) + mean).numpy()
    test_x = ((test_x.cpu() * std) + mean).numpy()
    return train_x, train_y, test_x, pred_labels