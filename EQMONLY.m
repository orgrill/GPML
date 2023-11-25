% This script should create the train_x, train_y, test_x data sets and save
% them as a csv to the shared folder with the ADSAIL machine.

clearvars -except CWS_DEM
close all
% Path for external (non Mathworks) functions and data loading commands
addpath("toolbox\ArcticMappingToolbox");
addpath("toolbox\kmz2struct");
addpath("toolbox\GPR\gpml-matlab-v4.2-2018-06-11\cov");

clear classes 
% These two python functions are set up to perform interpolation, one is
% gridded, one is scattered
mod = py.importlib.import_module('GSML_GPGRID'); % If all the training data is on a grid
py.importlib.reload(mod);
mod3 = py.importlib.import_module('GSML_GPSK'); % For Interpolating non-gridded stations
py.importlib.reload(mod3);

% Loading data
%load('data\CWS_DEM.mat')
load("data\CWS_SNAP_TMAX.mat");
load("data\CWS_StationData.mat");
load ('data\AllStationData.mat');
load("data\CWS_Projection.mat");
load("data\CWS_SNAP_Rect.mat");
SubRegion = kmz2struct('data\CopperRiverWatershed.kmz');
%% Getting everything into the same coordinate frame

% GCM (SNAP) Data is in polar stereographic, lets move it to lat/long, then
% to our projection grid
[SNAP_xpolar, SNAP_ypolar] = ndgrid(CWS_SNAPData.xc,CWS_SNAPData.yc);
[SNAP_lat, SNAP_long] = psn2ll(SNAP_xpolar,SNAP_ypolar,'TrueLat',64,'meridian',-152,'EarthRadius',6370000); 
[SNAPx, SNAPy] = projfwd(CWS_Projection.ProjectedCRS,SNAP_lat,SNAP_long); % This is 20km resolution
states = shaperead('usastatehi.shp','UseGeoCoords',true);
alaska = states(2,:);
alaska = geoshape(alaska);
[AKx, AKy] = projfwd(CWS_Projection.ProjectedCRS,alaska.Latitude,alaska.Longitude);
AlaskaPoly = polyshape(AKx,AKy);

%% Rotate
% We also have to rotate everything, this is not a regridding, just a
% coordinate transform, we perform the GP regression in the rotated
% rectilinear axes, all the plotting is in the non-rotated frame.  We also
% only rotate the gridded data, it doesn't matter if the training data is
% scattered
vectorX = reshape(CWS_SNAP_Rect.xgrid,[],1);
vectorY = reshape(CWS_SNAP_Rect.ygrid,[],1);
SNAP_Rect = [vectorX vectorY];
theta =pi/2+atan(CWS_SNAP_Rect.lineParams(1)); 
rot = [cos(theta) -sin(theta); sin(theta) cos(theta)];
points=(SNAP_Rect)*rot;

% Even After rotation, there is still numeric error, discretize to get everything
% on an exact grid
Bin = discretize(points(:,1),size(SNAPx,1));
for i=1:size(SNAPx,1)
    pxg(i) = mean(points(Bin==i,1));
end
Bin = discretize(points(:,2),size(SNAPx,2));
for i=1:size(SNAPx,2)
    pyg(i) = mean(points(Bin==i,2));
end

% Now re-grid using the discretized values
[pxg, pyg] = meshgrid(pxg,pyg);
pxg = fliplr(pxg');
pyg = fliplr(pyg');
vectorX = reshape(pxg,[],1);
vectorY = reshape(pyg,[],1);
train_x = [vectorX vectorY];       % SNAP Coordinates Rotated
SNAP_Rect_Rot = [vectorX vectorY]; % SNAP Coordinates Rotated

% We want to downsample for every day at a very fine grid,
% currently we are only set up to downsample to the test stations
% This builds a grid that we use for the very fine downsampling
factor = 20;
downGrid = apxGrid('create',SNAP_Rect_Rot,1,[size(SNAPx,1)*factor size(SNAPx,2)*factor]); 
% Lets Go in 6 on each edge, for some reason it extends outside the
% training region
[Xdown, Ydown] = meshgrid(downGrid{1}(6:end-5),downGrid{2}(6:end-5));

%%  Clean data
WS_Data_Clean = GSML_CleanWeatherStation(WS_Data);
WS_Data = WS_Data_Clean;

% Weather station data is still in Lat/Long, move it to our projection grid
[StationLL, iA, iC] = unique([[WS_Data{3}], [WS_Data{4}] ],'rows');
[StationX, StationY] = projfwd(CWS_Projection.ProjectedCRS,StationLL(:,1),StationLL(:,2));

%% Calculate Elevation Parameters for Downscaled Grid
% Elevations at the downsampled Grid (This cuts the edges quite a bit)
% This requires you to load CWS_DEM, which is big, so I downsampled it
load('data\CWS_SNAP_Fine.mat')
CWS_SNAP_Down.Elevation = CWS_SNAP_Fine.Elevation;  % We get a bunch of NANs near the borders
clear CWS_SNAP_Fine;
%% Split into test and fit sets, set up and execute EQM and topographical downscaling
% Load the raw station data, we need to divide this into two parts, 80/20
seed = 42;
rng(seed);

FitSet = sort(randperm(size(AllStationData,1),round(.8*size(AllStationData,1))));
%test set is complement of FitSet to ensure training data  isn't used in
%training and vice versa. 
TestSet = sort(setdiff(1:length(StationX),FitSet));

% Fit the Topographical Downscaling parameters (White 2016)
addpath('src\TemperatureBiasCalculator\modules\')
[StationDownScaleParams] = GSML_Topo_Downscale(AllStationData, FitSet); %[T0; Beta; Gamma; zref]

% Topographic Downscaling for SNAP data
CWS_SNAP_Rect.t2ref = zeros(size(CWS_SNAP_Rect.t2max));
for i=1:length(CWS_SNAP_Rect.Days)
    CWS_SNAP_Rect.t2ref(:,:,i) = CWS_SNAP_Rect.t2max(:,:,i)-StationDownScaleParams(3)*(StationDownScaleParams(4)-CWS_SNAP_Rect.Elevations);
end

% Go through all the stations and build the temperatures from the fit
% set, move one to reference elevation, and keep one at non-reference
FitDays = [];
station_Lengths = cellfun(@length, AllStationData(FitSet,3), 'UniformOutput', false);
for i=1:size(station_Lengths,1)
    temps = cell2mat(AllStationData(FitSet(i),4))-StationDownScaleParams(3)*(StationDownScaleParams(4)-cell2mat(AllStationData(FitSet(i),2)).*.3048);
    Tstation_Ref{i} = temps; 
    Tstation_NonRef{i} = cell2mat(AllStationData(FitSet(i),4));
    FitDays = [FitDays; [AllStationData{FitSet(i),3}]];
end
FitDays = unique(FitDays);  % These are the days present in the fit stations

%% Reshapes Station Data so its by date, This is ugly and slow
%Splits the station data into a training and fit set, indexed by date
xStationFit = cell(length(FitDays),1);
yStationFit = cell(length(FitDays),1);
yStationFit_NonRef = cell(length(FitDays),1);
for j=1:length(FitDays)
    for i=1:size(station_Lengths,1)
        this_dayStation = find([AllStationData{FitSet(i),3}]==FitDays(j));
        if ~isempty(this_dayStation)
            yStationFit{j} = [yStationFit{j}; Tstation_Ref{i}(this_dayStation) ];
            xStationFit{j} = [xStationFit{j}; [AllStationData{FitSet(i),5}]];
            yStationFit_NonRef{j} = [yStationFit_NonRef{j}; Tstation_NonRef{i}(this_dayStation) ];
        end
    end
end
%load('xyStationFit.mat');

%% Switch to the Test Set
% Find all the days in the test set, then find all the days that are
% present in the fit set and in the test set
station_Lengths = cellfun(@length, AllStationData(TestSet,3), 'UniformOutput', false);
TestDays = [];
for i=1:size(station_Lengths,1)
    TestDays = [TestDays; [AllStationData{TestSet(i),3}]];
end
TestDays = unique(TestDays); % These are the days present in the test stations
logical_Index = ismember(CWS_SNAP_Rect.Days,TestDays);
SNAPDays_inTest = find(logical_Index);
logical_Index = ismember(FitDays,TestDays);
FitStationDays_inTest = find(logical_Index);

%% Have to do this before running interpolation at all the days in SNAP data(below) - or just load SNAP_Ref_All
% EliminateLeap = find(~(month(CWS_SNAP_Rect.Days)==2 & day(CWS_SNAP_Rect.Days)==29));
% Days = CWS_SNAP_Rect.Days(EliminateLeap);
% nanIndices = find(isnan(CWS_SNAP_Rect.t2ref));
% 
% %removing nan's and keeping shape of CWS_SNAP_Rect (21 x 24 x size of data w/out NaN's)
% originalSize = size(CWS_SNAP_Rect.t2ref);
% reshapedData = reshape(CWS_SNAP_Rect.t2ref, [], originalSize(3)); %reshape to 2D array (21*24) x 47843
% %identify and remove nans
% nanIndices = any(isnan(reshapedData), 1);
% reshapedData(:, nanIndices) = [];
% % reshape back to original size
% CWS_SNAP_Rect.t2ref = reshape(reshapedData, [originalSize(1), originalSize(2), size(reshapedData, 2)]);
%% Interpolate all the SNAP data at reference elevation to ALL the station locations
% at ALL the days (47,843) in the SNAP Data set, We should only have to do this once!
% all_x = [StationX StationY];
% TheseDaysIndex = 1:length(Days);
% SNAP_Ref_All = InterpolateSNAP(SNAP_Rect_Rot,CWS_SNAP_Rect.t2ref,all_x,TheseDaysIndex);
load("data\SNAP_Ref_All.mat");
%%
% We can move to non-ref by applying the reverse transform, I don't think I
% need to reinterpolate?
SNAP_NonRef_All = zeros(size(SNAP_Ref_All));
for i=1:size(SNAP_Ref_All,1)
    StationZ = mean(AllStationData{i,2}).*.3048; % Feet to Meters 
    SNAP_NonRef_All(i,:) = SNAP_Ref_All(i,:)-StationDownScaleParams(3)*(StationZ-StationDownScaleParams(4));
end
SNAP_NonRef_Fit = SNAP_NonRef_All(FitSet,:);
SNAP_NonRef_Test = SNAP_NonRef_All(TestSet,:);
SNAP_All_Day = CWS_SNAP_Rect.Days;
SNAP_Ref_Fit = SNAP_Ref_All(FitSet,:);
SNAP_Ref_Test = SNAP_Ref_All(TestSet,:);

% The Snap Data has nans at leap years, this cuts those from the data 
EliminateLeap = find(~(month(SNAP_All_Day)==2 & day(SNAP_All_Day)==29));
SNAP_All_Day = SNAP_All_Day(EliminateLeap);
SNAP_NonRef_Test = SNAP_NonRef_Test(EliminateLeap);
SNAP_Ref_Test = SNAP_Ref_Test(EliminateLeap);
SNAP_NonRef_Fit = SNAP_NonRef_Fit(EliminateLeap);
SNAP_Ref_Fit = SNAP_Ref_Fit(EliminateLeap);
% Choose a set of future dates to apply the EQM method to
FutureDates = SNAP_All_Day(SNAP_All_Day > FitDays(end));

%% Interpolate FitSet Station data at reference elevation to test set station locations
%This takes a decent amount of time, 5ish minutes
 test_x = [StationX(TestSet) StationY(TestSet)];
 Station_Ref_Test = InterpolateStation(xStationFit,yStationFit_NonRef,test_x,FitStationDays_inTest);
 Station_TestDays = FitDays(FitStationDays_inTest);
% Create a non-reference station data set by applying the transform

Station_NonRef_Test = zeros(size(Station_Ref_Test));
for i=1:size(Station_Ref_Test,1)
     StationZ = mean(AllStationData{TestSet(i),2}).*.3048; % Feet to Meters, elevation of the test stations 
     Station_NonRef_Test(i,:) = Station_Ref_Test(i,:)-StationDownScaleParams(3)*(StationZ-StationDownScaleParams(4));
end

%% Find the ECDF functions
% This is done using the reference data we have interpolated to the
% test stations, we want two ecdf functions for each month, one for the
% SNAP data and one for the Station data.
SNAPMonth = cell(12,3);
StationMonth = cell(12,3);
SNAP_DeBias_Ref = zeros(size(AllStationData,1),length(FutureDates));
% We have to refind the test days in the SNAP data since we got rid of leap
% days
logical_Index = ismember(SNAP_All_Day,TestDays);
SNAPDays_inTest = find(logical_Index);
SNAP_TestDay = SNAP_All_Day(SNAPDays_inTest);

logical_Index = ismember(SNAP_All_Day,FutureDates);
SNAPDays_inFuture = find(logical_Index);
SNAP_FutureDay = SNAP_All_Day(SNAPDays_inFuture);

for i=1:12
    numdays_thismonth = eomday(2023,i);  
    for j=1:numdays_thismonth
        TheseDays = month(SNAP_TestDay)==i & day(SNAP_TestDay)==j; 
        WhereinSNAP = SNAPDays_inTest(TheseDays);
        SNAPMonth{i,1} = [SNAPMonth{i}; reshape(SNAP_Ref_Test(WhereinSNAP),[],1)];
        WhereinStation = month(Station_TestDays)==i & day(Station_TestDays)==j;
        StationMonth{i,1} = [StationMonth{i}; reshape(Station_Ref_Test(WhereinStation),[],1)];
    end
    [StationMonth{i,2}, StationMonth{i,3}] = ecdf(StationMonth{i,1});
    [SNAPMonth{i,2}, SNAPMonth{i,3}] = ecdf(SNAPMonth{i,1});
     
    % Now that we have the ECDF functions, go through and apply them to the
    % SNAP Ref data
    for j=1:numdays_thismonth
        TheseDays = month(SNAP_All_Day)==i & day(SNAP_All_Day)==j;
        f1 = @(x) interp1(SNAPMonth{i,3}(2:end),SNAPMonth{i,2}(2:end),x,"nearest",'extrap'); % x,f
        f2 = @(x) interp1(StationMonth{i,2}(2:end),StationMonth{i,3}(2:end),x,'nearest','extrap'); % f,x
        SNAP_DeBias_Ref(:,TheseDays) = arrayfun(f1,SNAP_Ref_All(:,TheseDays));
        SNAP_DeBias_Ref(:,TheseDays) = arrayfun(f2,SNAP_DeBias_Ref(:,TheseDays));
    end
end

% Now Move the DeBias_Ref Temps back to actual elevation using the known
% elevation of each test station
SNAP_DeBias_NonRef = zeros(size(SNAP_DeBias_Ref));
for i=1:size(AllStationData,1)
    StationZ = mean(AllStationData{i,2}).*.3048;
    %SNAP_FutureDeBias_NonRef(i,:) = SNAP_FutureDeBias_Ref(i,:)-StationDownScaleParams(3)*(StationZ-StationDownScaleParams(4));
    SNAP_DeBias_NonRef(i,:) = SNAP_DeBias_Ref(i,:)-StationDownScaleParams(3)*(StationZ-StationDownScaleParams(4));
end
%%
WhereinAllTestDay1 = cell(length(TestSet),1);
WhereinAllTestDay3 = cell(length(TestSet),1);

for i=1:length(TestSet)
    TheseStationTmax = [AllStationData{TestSet(i), 4}];
    StationDay = [AllStationData{TestSet(i), 3}];
end

AllTestDays = union(StationDay,SNAP_FutureDay); % All the days we want to test
[~,WhereinSNAPNONRef{i}, WhereinAllTestDay1{i}] = intersect(SNAP_TestDay, AllTestDays); % Where the old SNAP_NONRef is
[~,WhereinStationDay{i}, WhereinAllTestDay3{i}] = intersect(StationDay,AllTestDays);    % Where the Station Data is
[StationANDSNAP, ~, ib] = intersect(WhereinAllTestDay1{i},WhereinAllTestDay3{i});

EQMBias = cell(length(TestSet), 1);

for i = 1:length(TestSet)
    TheseStationTmax = [AllStationData{TestSet(i), 4}];
    StationDay = [AllStationData{TestSet(i), 3}];
    
    [~, Instation2, InSnapDeBias] = intersect(StationDay, SNAP_TestDay);

    % Check if there are common elements between StationDay and SNAP_TestDay
    if ~isempty(Instation2) && ~isempty(InSnapDeBias)
        EQMBias{i} = TheseStationTmax(Instation2) - SNAP_DeBias_NonRef(i, InSnapDeBias)';
    else
        EQMBias{i} = [];
    end
end

%RMSE for each station individually
RMSE_per_station = zeros(length(TestSet), 1);
for i = 1:length(TestSet)
    if ~isempty(EQMBias{i})
        RMSE_per_station(i) = sqrt(mean(EQMBias{i}.^2));
    else
        RMSE_per_station(i) = NaN; 
    end
end

EQMRMSE = mean(RMSE_per_station)

%% Plots
plot(SNAP_All_Day, SNAP_NonRef_Test, 'LineWidth', 1.5);
plot(SNAP_All_Day, SNAP_DeBias_NonRef, 'LineWidth', 1.5);
plot(Station_TestDays, Station_NonRef_Test, 'LineWidth', 1.5);

title('Original SNAP vs. Bias-Corrected SNAP vs. Station Data');
legend('Original SNAP', 'Bias-Corrected SNAP', 'Station Data');
xlabel('Date');
ylabel('Temperature (°K)');

% %%
% % Plotting original SNAP data and station data
% figure;
% subplot(2,1,1);
% plot(SNAP_All_Day, SNAP_NonRef_Test, '-b', 'LineWidth', 1.5);
% hold on;
% plot(Station_TestDays, Station_NonRef_Test, '-r', 'LineWidth', 1.5);
% title('Original SNAP Data vs. Station Data');
% legend('SNAP Data', 'Station Data');
% xlabel('Date');
% ylabel('Temperature (°C)');
% 
% % Plotting bias-corrected SNAP data and station data
% subplot(2,1,2);
% plot(SNAP_All_Day, SNAP_DeBias_NonRef, '-g', 'LineWidth', 1.5);
% hold on;
% plot(Station_TestDays, Station_NonRef_Test, '-r', 'LineWidth', 1.5);
% title('Bias-Corrected SNAP Data vs. Station Data');
% legend('Bias-Corrected SNAP Data', 'Station Data');
% xlabel('Date');
% ylabel('Temperature (°C)');

%% Function Files

function [DownSampleOut] = InterpolateStation(xStationFit,yStationFit,test_x,TheseDays)
% This does the grid interpolation using GPyTorch, so the fit and test data
% can be scattered
    test_x = test_x./1e6;
    pyTestX = py.torch.from_numpy(py.numpy.array(test_x));
    DownSampleOut = zeros(size(test_x,1),length(TheseDays));
    % The training set is not constant every day
    for i=1:length(TheseDays)
       disp(['Downsampling Day ', num2str(i)])
       train_x = xStationFit{TheseDays(i)}./1e6; 
       train_y = yStationFit{TheseDays(i)};
       pyTrainingX = py.torch.from_numpy(py.numpy.array(train_x));
       pyTrainingY = py.torch.from_numpy(py.numpy.array(train_y));
                            
       outvars = py.GSML_GPSK.GPSKI(pyTrainingX,pyTrainingY, pyTestX);
       outcell = cell(outvars);
       %trainXout = double(outcell{1}.cpu().numpy);
       %trainYout = double(outcell{2}.cpu().numpy);
       %testXout = double(outcell{3}.cpu().numpy);
       testYout = double(outcell{4}.cpu().numpy);
       DownSampleOut(1:size(test_x,1),i)= testYout';
   end 
end