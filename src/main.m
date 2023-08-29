cd("..")
addpath(fullfile("src", "TemperatureBiasCalculator"))


x = TemperatureBiasCalculator;
bias = x.CalculateBias();

% a = NextStepObject;
% a.DoNextStep(bias);