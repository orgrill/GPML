% This script should create the train_x, train_y, test_x data sets and save
% them as a csv to the shared folder with the ADSAIL machine
clearvars -except CWS_DEM
close all

% Path for external (non Mathworks) functions and data loading commands
addpath("src")
addpath("toolbox\ArcticMappingToolbox");
addpath("toolbox\kmz2struct");
addpath('toolbox\Gaussian Process Regression(GPR)\gpml-matlab-v4.2-2018-06-11\cov');


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%   Coordinate Transform       %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%   Topographical Downscaling       %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%
%  BuildTrainSet %
%%%%%%%%%%%%%%%%%%