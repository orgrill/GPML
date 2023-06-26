# -*- coding: utf-8 -*-
"""
Created on Fri Aug 12 14:25:30 2022

@author: mkupilik
"""
def GPGrid(train_x, train_y, grid, test_x):
    import gpytorch
    import torch
    import math
    import gc
    # grid_bounds = [(0, 5), (0, 2)]
    # grid_size = 100
    # grid = torch.zeros(grid_size, len(grid_bounds))
    # for i in range(len(grid_bounds)):
    #     grid_diff = float(grid_bounds[i][1] - grid_bounds[i][0]) / (grid_size - 2)
    #     grid[:, i] = torch.linspace(grid_bounds[i][0] - grid_diff, grid_bounds[i][1] + grid_diff, grid_size)
    
    # train_x = gpytorch.utils.grid.create_data_from_grid(grid)
    # train_y = torch.sin((train_x[:, 0] + train_x[:, 1]) * (2 * math.pi)) + torch.randn_like(train_x[:, 0]).mul(0.01)
    gc.collect()
    torch.cuda.empty_cache() 
    if torch.cuda.is_available():
        train_x, train_y = train_x.cuda(), train_y.cuda()
    class GridGPRegressionModel(gpytorch.models.ExactGP):
        def __init__(self, grid, train_x, train_y, likelihood):
            super(GridGPRegressionModel, self).__init__(train_x, train_y, likelihood)
            num_dims = train_x.size(-1)
            self.mean_module = gpytorch.means.ConstantMean()
            #self.covar_module = gpytorch.kernels.GridKernel(gpytorch.kernels.RBFKernel(), grid=grid)
            self.covar_module = gpytorch.kernels.GridKernel(gpytorch.kernels.MaternKernel(), grid=grid)
        def forward(self, x):
            mean_x = self.mean_module(x)
            covar_x = self.covar_module(x)
            return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
    
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    model = GridGPRegressionModel(grid, train_x, train_y, likelihood)
    hypers = {
    # 'likelihood.noise_covar.noise': myparam[1],
    # 'covar_module.base_kernel.lengthscale': myparam[0],
    'likelihood.noise_covar.noise': torch.tensor(.0001),
    'covar_module.base_kernel.lengthscale': torch.tensor(.04),
    }

    model.initialize(**hypers)
    import os
    smoke_test = ('CI' in os.environ)
    training_iter = 25 if smoke_test else 2
    if torch.cuda.is_available():
        model = model.cuda()
        likelihood = likelihood.cuda()   
    
    # Find optimal model hyperparameters
    model.train()
    likelihood.train()
    
    # Use the adam optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=.3)  # Includes GaussianLikelihood parameters
    # "Loss" for GPs - the marginal log likelihood
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
    
    for i in range(training_iter):
        # Zero gradients from previous iteration
        optimizer.zero_grad()
        # Output from model
        output = model(train_x)
        # Calc loss and backprop gradients
        loss = -mll(output, train_y)
        loss.backward()
        #loss.backward(retain_graph=True)
        # print('Iter %d/%d - Loss: %.3f   lengthscale: %.3f   noise: %.3f' % (
        #     i + 1, training_iter, loss.item(),
        #     model.covar_module.base_kernel.lengthscale.item(),
        #     model.likelihood.noise.item()
        # ))
        optimizer.step()
      
    model.eval()
    likelihood.eval()
    # n = 20
    # test_x = torch.zeros(int(pow(n, 2)), 2)
    # for i in range(n):
    #     for j in range(n):
    #         test_x[i * n + j][0] = float(i) / (n-1)
    #         test_x[i * n + j][1] = float(j) / (n-1)
    if torch.cuda.is_available():
        test_x = test_x.cuda()
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        observed_pred = likelihood(model(test_x))
    
    # matplotlib.pyplot as plt
    # % matplotlib inline
    # n1 = len(torch.unique(test_x[:,0]))
    # n2 = len(torch.unique(test_x[:,1]))
    #pred_labels = observed_pred.mean.view(n1,n2)
    pred_labels = observed_pred.mean
    # Calc abosolute error
    train_x = train_x.cpu()
    train_y = train_y.cpu()
    test_x = test_x.cpu()
    pred_labels = pred_labels.cpu()
    #test_y_actual = torch.sin(((test_x[:, 0] + test_x[:, 1]) * (2 * math.pi))).view(n, n)
    #delta_y = torch.abs(pred_labels - test_y_actual).detach().numpy()
    #myparam = [model.covar_module.base_kernel.lengthscale.item(), model.likelihood.noise.item()]
    
    
    return train_x,train_y,test_x,pred_labels
# =============================================================================
# # Define a plotting function
# def ax_plot(f, ax, y_labels, title):
#     if smoke_test: return  # this is for running the notebook in our testing framework
#     im = ax.imshow(y_labels)
#     ax.set_title(title)
#     f.colorbar(im)
# 
# # Plot our predictive means
# f, observed_ax = plt.subplots(1, 1, figsize=(4, 3))
# ax_plot(f, observed_ax, pred_labels, 'Predicted Values (Likelihood)')
# 
# # Plot the true values
# f, observed_ax2 = plt.subplots(1, 1, figsize=(4, 3))
# ax_plot(f, observed_ax2, test_y_actual, 'Actual Values (Likelihood)')
# 
# # Plot the absolute errors
# f, observed_ax3 = plt.subplots(1, 1, figsize=(4, 3))
# ax_plot(f, observed_ax3, delta_y, 'Absolute Error Surface')
# 
# 
# =============================================================================
