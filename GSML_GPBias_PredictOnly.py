# -*- coding: utf-8 -*-
"""
Created on Fri Aug 12 14:25:30 2022
This actually does training as well, I needed to retrain after adjusting the grid!
@author: mkupilik
"""
def BiasSKI(train_x, train_y, test_x):
    import math
    import torch
    from torch.utils.data import Dataset
    import gpytorch
    import gc
    import json
    from json import JSONEncoder
    
    gc.collect()
    torch.cuda.empty_cache() 
    train_x = torch.tensor(train_x, dtype=torch.float)
    train_y = torch.tensor(train_y, dtype=torch.float)
    test_x = torch.tensor(test_x, dtype=torch.float)
    
    
    class GPRegressionModel(gpytorch.models.ExactGP):
        def __init__(self, train_x, train_y, likelihood):
            super(GPRegressionModel, self).__init__(train_x, train_y, likelihood)
            
            # SKI requires a grid size hyperparameter. This util can help with that
            #grid_size = gpytorch.utils.grid.choose_grid_size(train_x)
            #grid_size = gpytorch.utils.grid.choose_grid_size(train_x, kronecker_structure=False)
            #grid_bounds = torch.tensor([[0.2922, 4.7843], [.003, 2]])
            grid_bounds = torch.tensor(((0.003, 4.7843),))
            #grid_size = ([400, 10])
            #print(grid_size)
            #self.mean_module = gpytorch.means.ZeroMean()
            self.mean_module = gpytorch.means.LinearMean(input_size=torch.tensor(2))
            self.covar_module = gpytorch.kernels.AdditiveStructureKernel(
            #self.covar_module = gpytorch.kernels.ProductStructureKernel(
                gpytorch.kernels.GridInterpolationKernel(
                    #gpytorch.kernels.SpectralMixtureKernel(num_mixtures=10), grid_size=16000, grid_bounds = grid_bounds, num_dims=1
                    #gpytorch.kernels.SpectralMixtureKernel(num_mixtures=6), grid_size=100000, grid_bounds = grid_bounds, num_dims=1
                    gpytorch.kernels.PeriodicKernel(), grid_size=100000, grid_bounds = grid_bounds, num_dims=1
            ), num_dims=2
            )
            # self.covar_module = gpytorch.kernels.GridInterpolationKernel(
            #         gpytorch.kernels.SpectralMixtureKernel(num_mixtures=4,ard_num_dims = 2), grid_bounds = grid_bounds, grid_size=grid_size, num_dims=2
            
            # )
            #self.covar_module.base_kernel.base_kernel.initialize_from_data(test_x[:,1], train_y)
        def forward(self, x):
            mean_x = self.mean_module(x)
            covar_x = self.covar_module(x)
            return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    model = GPRegressionModel(train_x, train_y, likelihood).cuda()
    hypers = {
    # 'likelihood.noise_covar.noise': myparam[1],
    # 'covar_module.base_kernel.lengthscale': myparam[0],
    'likelihood.noise_covar.noise': torch.tensor(6),
    #'covar_module.base_kernel.': torch.tensor(.14),
    }

    model.initialize(**hypers)
    #78151 has 8 mixtures, grid size 100000 (tested on 10 future days)
    #77763 has 8 mixtures, grid size 100000
    # 73715 has 8 mixtures, grid size 16000
    #state_dict = torch.load('G:/Other computers/UAALaptop/UAA/Research/GeoSpatialML/Code/GPSKI73715.pth')
    # 84051 only has 6 mixtures, grid size 16000
    #state_dict = torch.load('G:/Other computers/UAALaptop/UAA/Research/GeoSpatialML/Code/GPSKI77763.pth')
    #model = GPRegressionModel(train_x, train_y, likelihood)
    #model.load_state_dict(state_dict)
   
    likelihood.train()
    if torch.cuda.is_available():
        model = model.cuda()
        likelihood = likelihood.cuda()   
    # Find optimal model hyperparameters
   
    model.train()
    
    # # Use the adam optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)  # Includes GaussianLikelihood parameters
    
    # # "Loss" for GPs - the marginal log likelihood
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
    training_iterations = 500
    def train():
        for i in range(training_iterations):
            optimizer.zero_grad()
            output = model(train_x.cuda())
            loss = -mll(output, train_y.cuda())
            loss.backward()
            optimizer.step()
            print('Iter %d/%d - Loss: %d  noise: %.3f ' % (i + 1, training_iterations, loss.item(),model.likelihood.noise.item()))
    #         # print('Iter %d  noise: %.3f' % (
    #         #     i + 1,  
    #         #     model.likelihood.noise.item()
    #         # ))
    train()
    # # Need to add code to save the model, so we can load it and use it for inference later
    # torch.save(model.state_dict(), 'G:/Other computers/UAALaptop/UAA/Research/GeoSpatialML/Code/MyGSML_GPSKI.pth')
    # Set model and likelihood into evaluation mode
    model.eval()
    likelihood.eval()
    
    if torch.cuda.is_available():
        test_x = test_x.cuda()
        
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        observed_pred = likelihood(model(test_x))
        #pred_labels = observed_pred.mean.view(n, n)
        pred_labels = observed_pred.mean
    train_x = train_x.cpu()
    train_y = train_y.cpu()
    test_x = test_x.cpu()
    pred_labels = pred_labels.cpu()
    
    # Here is how we can write the model to a text file
    class EncodeTensor(JSONEncoder,Dataset):
        def default(self, obj):
            if isinstance(obj, torch.Tensor):
                return obj.cpu().detach().numpy().tolist()
            return super(EncodeTensor, self).default(obj)

    with open('torch_weights.json', 'w') as json_file:
        json.dump(model.state_dict(), json_file,cls=EncodeTensor)
    
    with open('torch_weights.json', 'r') as f:
        data = json.load(f)

    # Convert the Python object to text
    text = str(data)

    # Write the text to a new file
    with open('GPyModel.txt', 'w') as f:
        f.write(text)    
    torch.save(model.state_dict(), 'G:/Other computers/UAALaptop/UAA/Research/GeoSpatialML/Code/MyGSML_GPSKI.pth')
    return train_x,train_y,test_x,pred_labels

