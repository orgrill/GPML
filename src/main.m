addpath("TemperatureBiasCalculator/")
% addpath("NextStepObject/")

x = TemperatureBiasCalculator;
bias = x.CalculateBias();

% a = NextStepObject;
% a.DoNextStep(bias);