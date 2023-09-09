function [] = main(Skip_Transform)
    addpath("src");
    addpath(fullfile("src", "TemperatureBiasCalculator"));

    % Load data
    CWS_Projection = load(fullfile("data", "CWS_Projection.mat")).CWS_Projection;
    AllStationData = load(fullfile("data", "AllStationData.mat")).AllStationData;
    if Skip_Transform
        CWS_SNAPData = load(fullfile("data","CWS_SNAP_Rect.mat")).CWS_SNAP_Rect;
    else
        CWS_SNAPData = load(fullfile("data","CWS_SNAP_TMAX.mat")).CWS_SNAPData;
    end

    % Construct a bias calculator 
    x = TemperatureBiasCalculator(AllStationData, CWS_Projection, CWS_SNAPData, Skip_Transform);
    rsme = x.CalculateBias();
end