# -*- coding: utf-8 -*-
"""
Created on Fri Aug 12 14:25:30 2022

@author: mkupilik
"""
def GPSKI(train_x, train_y, test_x):
    import math
    import torch
    import gpytorch
    #from matplotlib import pyplot as plt

    # Make plots inline
    #%matplotlib inline
    
    
    # We make an nxn grid of training points spaced every 1/(n-1) on [0,1]x[0,1]
    # n = 40
    # train_x = torch.zeros(pow(n, 2), 2)
    # for i in range(n):
    #     for j in range(n):
    #         train_x[i * n + j][0] = float(i) / (n-1)
    #         train_x[i * n + j][1] = float(j) / (n-1)
    # True function is sin( 2*pi*(x0+x1))
    #train_y = torch.sin((train_x[:, 0] + train_x[:, 1]) * (2 * math.pi)) + torch.randn_like(train_x[:, 0]).mul(0.01)
    if torch.cuda.is_available():
        train_x, train_y = train_x.cuda(), train_y.cuda()
    class GPRegressionModel(gpytorch.models.ExactGP):
        def __init__(self, train_x, train_y, likelihood):
            super(GPRegressionModel, self).__init__(train_x, train_y, likelihood)
    
            # SKI requires a grid size hyperparameter. This util can help with that
            #grid_size = gpytorch.utils.grid.choose_grid_size(train_x)
            grid_size = 13
            #print(grid_size)
            self.mean_module = gpytorch.means.ConstantMean()
            self.covar_module = gpytorch.kernels.ScaleKernel(
                gpytorch.kernels.GridInterpolationKernel(
                    #gpytorch.kernels.RBFKernel(), grid_size=grid_size, num_dims=2
                    gpytorch.kernels.MaternKernel(), grid_size=grid_size, num_dims=2
                )
            )
    
        def forward(self, x):
            mean_x = self.mean_module(x)
            covar_x = self.covar_module(x)
            return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    model = GPRegressionModel(train_x, train_y, likelihood)
    hypers = {
    # 'likelihood.noise_covar.noise': myparam[1],
    # 'covar_module.base_kernel.lengthscale': myparam[0],
    'likelihood.noise_covar.noise': torch.tensor(.01),
    #'covar_module.base_kernel.': torch.tensor(.14),
    }

    model.initialize(**hypers)
    likelihood.train()
    if torch.cuda.is_available():
        model = model.cuda()
        likelihood = likelihood.cuda()   
    # Find optimal model hyperparameters
    model.train()
    
    # Use the adam optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)  # Includes GaussianLikelihood parameters
    
    # "Loss" for GPs - the marginal log likelihood
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
    training_iterations = 100 
    def train():
        for i in range(training_iterations):
            optimizer.zero_grad()
            output = model(train_x)
            loss = -mll(output, train_y)
            loss.backward()
            optimizer.step()
            print('Iter %d/%d - Loss: %.3f   lengthscale: %.3f   noise: %.3f' % (
                i + 1,  training_iterations, loss.item(),
                model.covar_module.base_kernel.lengthscale.item(),
                model.likelihood.noise.item()
            ))
    #%time train()

    # Set model and likelihood into evaluation mode
    model.eval()
    likelihood.eval()
    
    # Generate nxn grid of test points spaced on a grid of size 1/(n-1) in [0,1]x[0,1]
    # n = 10
    # test_x = torch.zeros(int(pow(n, 2)), 2)
    # for i in range(n):
    #     for j in range(n):
    #         test_x[i * n + j][0] = float(i) / (n-1)
    #         test_x[i * n + j][1] = float(j) / (n-1)
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
    # Calc abosolute error
    # test_y_actual = torch.sin(((test_x[:, 0] + test_x[:, 1]) * (2 * math.pi))).view(n, n)
    # delta_y = torch.abs(pred_labels - test_y_actual).detach().numpy()
    
    # Define a plotting function
    #def ax_plot(f, ax, y_labels, title):
    #    if smoke_test: return  # this is for running the notebook in our testing framework
    #    im = ax.imshow(y_labels)
    #    ax.set_title(title)
    #    f.colorbar(im)
    
    ## Plot our predictive means
    #f, observed_ax = plt.subplots(1, 1, figsize=(4, 3))
    #ax_plot(f, observed_ax, pred_labels, 'Predicted Values (Likelihood)')
    
    ## Plot the true values
    #f, observed_ax2 = plt.subplots(1, 1, figsize=(4, 3))
    #ax_plot(f, observed_ax2, test_y_actual, 'Actual Values (Likelihood)')
    
    # Plot the absolute errors
    #f, observed_ax3 = plt.subplots(1, 1, figsize=(4, 3))
    #ax_plot(f, observed_ax3, delta_y, 'Absolute Error Surface')
    
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
