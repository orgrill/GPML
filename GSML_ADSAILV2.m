% This script should create the train_x, train_y, test_x data sets and save
% them as a csv to the shared folder with the ADSAIL machine



clearvars -except CWS_DEM
close all

% Path for external (non Mathworks) functions and data loading commands
addpath("toolbox\ArcticMappingToolbox");
addpath("toolbox\kmz2struct");
addpath('toolbox\Gaussian Process Regression(GPR)\gpml-matlab-v4.2-2018-06-11\cov');

% These two python functions are set up to perform interpolation, one is
% gridded, one is scattered
mod = py.importlib.import_module('GSML_GPGRID'); % If all the training data is on a grid
py.importlib.reload(mod);
% mod2 = py.importlib.import_module('GSML_GPBias_TrainPredict');  % Uses kernel interpolation on a grid
% py.importlib.reload(mod2);

% Loading data
load("data\CWS_SNAP_TMAX.mat");
%load('data\CWS_DEM.mat')
WS_Data = load("data\CWS_StationData.mat");
load("data\CWS_Projection.mat");
SubRegion = kmz2struct('CopperRiverWatershed.kmz');
%% Getting everything into the same coordinate frame
% GCM (SNAP) Data is in polar stereographic, lets move it to lat/long, then
% to our projection grid

[SNAP_xpolar, SNAP_ypolar] = ndgrid(CWS_SNAPData.xc,CWS_SNAPData.yc);
[SNAP_lat, SNAP_long] = psn2ll(SNAP_xpolar,SNAP_ypolar,'TrueLat',64,'meridian',-152,'EarthRadius',6370000); 
[SNAPx, SNAPy] = projfwd(CWS_Projection.ProjectedCRS,SNAP_lat,SNAP_long); % This is 20km resolution

% The lat/long moved into the new grid is not rectilinear
% The following function moves it and interpolates, created a new SNAP data
% object, requires an external function, CurveGrid2Rect
% [CWS_SNAP_Rect] = MovetoRectilinear_Interpolate(SNAPx,SNAPy,CWS_SNAPData);
% save('CWS_SNAP_Rect','CWS_SNAP_Rect');
% We only have to do this once, so it is best to save the result, then load
load("data\CWS_SNAP_Rect.mat")
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
train_x = [vectorX vectorY]; % SNAP Coordinates Rotated
SNAP_Rect_Rot = [vectorX vectorY]; % SNAP Coordinates Rotated
% These are the coordinates rotated back, just to check!
% rot = [cos(-theta) -sin(-theta); sin(-theta) cos(-theta)];
% rotXY=XY*(rot); 
% Xqr = reshape(rotXY(:,1), size(pxg,1), []);
% Yqr = reshape(rotXY(:,2), size(pyg,1), []);

factor = 20;
% Eventually we need to downsample for every day at a very fine grid,
% currently we are only set up to downsample to the test stations
% This builds a grid that we could use for the very fine downsampling
downGrid = apxGrid('create',SNAP_Rect_Rot,1,[size(SNAPx,1)*factor size(SNAPx,2)*factor]); 
% Lets Go in 6 on each edge, for some reason it extends outside the
% training region
[Xdown, Ydown] = meshgrid(downGrid{1}(6:end-5),downGrid{2}(6:end-5));
%test_x = [reshape(Xdown,[],1) reshape(Ydown,[],1)]; 
%% Call the Weather Cleaning Function
WS_Data_Clean = GSML_CleanWeatherStation(WS_Data);
WS_Data = WS_Data_Clean;
% Weather station data is still in Lat/Long, move it to our projection grid
[StationLL, iA, iC] = unique([[WS_Data{3}], [WS_Data{4}] ],'rows');
[StationX, StationY] = projfwd(CWS_Projection.ProjectedCRS,StationLL(:,1),StationLL(:,2));

%%
py.importlib.import_module('torch');
%% Calculate Elevation Parameters for Downscaled Grid
% % Elevations at the downsampled Grid (This cuts the edges quite a bit)
% This requires you to load CWS_DEM, which is big, so I downsampled it and
load("data\CWS_SNAP_Fine.mat")
CWS_SNAP_Down.Elevation = CWS_SNAP_Fine.Elevation;  % We get a bunch of NANs near the borders
clear CWS_SNAP_Fine;
% figure
% plot(AlaskaPoly,'FaceColor','none')
% hold on
% mapshow(CWS_SNAP_Down.Xgrid,CWS_SNAP_Down.Ygrid,CWS_SNAP_Down.Elevation,'DisplayType','surface','FaceAlpha',1);
%% Split into test and fit sets, set up and execute EQM and topgraphical downscaling
% Load the raw station data, we need to divide this into two parts, 80/20
load("data\AllStationData.mat")
% FitSet = sort(randperm(size(AllStationData,1),round(.8*size(AllStationData,1))));
% Load a particular test/train split, the interpolation is really slow
load ("data\FitSet.mat")
% Fit the Topographical Downscaling parameters (White 2016)
[StationDownScaleParams] = GSML_Topo_Downscale(AllStationData, FitSet); %[T0; Beta; Gamma; zref]
TestSet = sort(setdiff(1:length(StationX),FitSet));
% First move the coarse tmax data to reference
CWS_SNAP_Rect.t2ref = zeros(size(CWS_SNAP_Rect.t2max));
for i=1:length(CWS_SNAP_Rect.Days)
    CWS_SNAP_Rect.t2ref(:,:,i) = CWS_SNAP_Rect.t2max(:,:,i)-StationDownScaleParams(3)*(StationDownScaleParams(4)-CWS_SNAP_Rect.Elevations);
end
% Topographic Downscaling, we need to move GCM and Station data to a
% reference elevation
SNAP_temp_Ref = zeros(size(CWS_SNAP_Rect.t2max));
for i = 1:length(CWS_SNAP_Rect.Days)
    %SNAP_temp_Ref(:,:,i) = CWS_SNAP_Rect.t2max(:,:,i)-StationDownScaleParams(3)*(StationDownScaleParams(4).*ones(size(CWS_SNAP_Rect.xgrid))-CWS_SNAP_Rect.Elevations);
    SNAP_temp_Ref(:,:,i) = CWS_SNAP_Rect.t2ref(:,:,i);
end
station_Lengths = cellfun(@length, AllStationData(FitSet,3), 'UniformOutput', false);
for i=1:size(station_Lengths)
    temps = cell2mat(AllStationData(FitSet(i),4))-StationDownScaleParams(3)*(StationDownScaleParams(4)-cell2mat(AllStationData(FitSet(i),2)).*.3048);
    Tstation_Ref{i} = temps; 
    Tstation_NonRef{i} = cell2mat(AllStationData(FitSet(i),4));
end
FitDays = [];
for i=1:size(station_Lengths,1)
    FitDays = [FitDays; [AllStationData{FitSet(i),3}]];
end
FitDays = unique(FitDays);  % These are the days present in the fit stations
%% Downsample the NonRef SNAP Data to the fine grid
% This is for the spatial check, will probably only work for 1 day, need to
% change so it works for a range
TheseDaysIndex = find(CWS_SNAP_Rect.Days==datetime(2085,4,26));
CWS_SNAP_Down.Tmax = zeros(size(Xdown,1), size(Xdown,2), length(TheseDaysIndex));
CWS_SNAP_Down.Days = [CWS_SNAP_Rect.Days(TheseDaysIndex) ];
vectorX = reshape(Xdown,[],1);
vectorY = reshape(Ydown,[],1);
for i=1:length(TheseDaysIndex)%:size(CWS_SNAP_Rect.t2max,3)
     disp(['Downsampling Day ', num2str(i)])
     InterpOut = InterpolateSNAP(SNAP_Rect_Rot,CWS_SNAP_Rect.t2ref,[vectorX vectorY],TheseDaysIndex);
     CWS_SNAP_Down.Tmax(:,:,i) = reshape(InterpOut,size(Xdown));
     
end
%% Move everything Back, unrotate, and display in the original coordinate frame
rot = [cos(-theta) -sin(-theta); sin(-theta) cos(-theta)];
rotXY=[vectorX vectorY]*(rot); 
Xqr = reshape(rotXY(:,1), size(Xdown,1), []);
Yqr = reshape(rotXY(:,2), size(Xdown,1), []);
%
CWS_SNAP_Down.Xgrid = Xqr;
CWS_SNAP_Down.Ygrid = Yqr;
figure
plot(AlaskaPoly,'FaceColor','none')
hold on
mapshow(Xqr,Yqr,CWS_SNAP_Down.Tmax(:,:,1),'DisplayType','surface','FaceAlpha',1);

%% Reshapes Station Data so its by date, This is ugly and slow
% Splits the station data into a training and fit set, indexed by date
% xStationFit = cell(length(FitDays),1);
% yStationFit = cell(length(FitDays),1);
% yStationFit_NonRef = cell(length(FitDays),1);
% for j=1:length(FitDays)
%     for i=1:size(station_Lengths,1)
%         this_dayStation = find([AllStationData{FitSet(i),3}]==FitDays(j));
%         if ~isempty(this_dayStation)
%             yStationFit{j} = [yStationFit{j}; Tstation_Ref{i}(this_dayStation) ];
%             xStationFit{j} = [xStationFit{j}; [AllStationData{FitSet(i),5}]];
%             yStationFit_NonRef{j} = [yStationFit_NonRef{j}; Tstation_NonRef{i}(this_dayStation) ];
%         end
%     end
% end
load('xyStationFit.mat');

%% Switch to the Test Set
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

%% Interpolate the SNAP Temps NOT at reference elevation to the test stations
test_x = [StationX(TestSet) StationY(TestSet)];
% We want this at all the days up to 2020, this will be used to fit an ecdf
TheseDays = CWS_SNAP_Rect.Days(CWS_SNAP_Rect.Days<=datetime(2019,12,31));
%TheseDays = CWS_SNAP_Rect.Days(CWS_SNAP_Rect.Days<=datetime(1970,1,5));
%TheseDays = CWS_SNAP_Rect.Days(CWS_SNAP_Rect.Days>datetime(2022,3,22));
logical_Index = ismember(CWS_SNAP_Rect.Days,TheseDays);
TheseDaysIndex = find(logical_Index);
%SNAP_Ref_Test2 = InterpolateSNAP(SNAP_Rect_Rot,CWS_SNAP_Rect.t2ref,test_x,TheseDaysIndex);
load("data\SNAP_Ref_Test2.mat"); % This is the reference WRF, interpolated to the fine grid


load("data\SNAP_NONRef_FutureTest.mat"); % This is raw WRF data interpolated to stations, into the future
SNAP_NONRef_FutureTest = SNAP_NONRef_Test;
SNAP_NONRef_FutureTestDays = SNAP_NONRef_TestDay;
load("data\SNAP_NONRef_Test.mat")
SNAP_NONRef_TestDay = CWS_SNAP_Rect.Days(TheseDaysIndex);
% Get rid of the NAN leap days
EliminateLeap = find(~(month(SNAP_NONRef_TestDay)==2 & day(SNAP_NONRef_TestDay)==29));
SNAP_NONRef_TestDay = SNAP_NONRef_TestDay(EliminateLeap);
SNAP_NONRef_Test = SNAP_NONRef_Test(:,EliminateLeap);
SNAP_Ref_Test2 = SNAP_Ref_Test2(:,EliminateLeap);
%% We want the SNAP Data in the future at the reference elevation so we can apply the EQM to it, 
SNAP_Ref_FutureTest = zeros(size(SNAP_NONRef_FutureTest));
for i=1:length(TestSet)
    StationZ = mean(AllStationData{TestSet(i),2}).*.3048;
    SNAP_Ref_FutureTest(i,:) = SNAP_NONRef_FutureTest(i,:)-StationDownScaleParams(3)*(StationDownScaleParams(4)-StationZ);
end
SNAP_Ref_FutureTestDays = SNAP_NONRef_FutureTestDays;
%% Interpolate the SNAP Temps NOT at reference elevation to the fit locations
% These will be used to find the raw bias, which will be fed to the
% Gaussian Process
test_x = [StationX(FitSet) StationY(FitSet)];
% TheseDays = CWS_SNAP_Rect.Days(CWS_SNAP_Rect.Days<=datetime(2022,3,22));
% logical_Index = ismember(CWS_SNAP_Rect.Days,TheseDays);
% TheseDaysIndex = find(logical_Index);
% SNAP_Ref_Fit2 = InterpolateSNAP(SNAP_Rect_Rot,CWS_SNAP_Rect.t2ref,test_x,TheseDaysIndex);
load("data\SNAP_Ref_Fit.mat")

% SNAP_NONRef_FitDay = CWS_SNAP_Rect.Days(TheseDaysIndex);
load("data\SNAP_NONRef_Fit.mat")
% Get rid of the leap days
EliminateLeap = find(~(month(SNAP_NONRef_FitDay)==2 & day(SNAP_NONRef_FitDay)==29));
SNAP_NONRef_FitDay = SNAP_NONRef_FitDay(EliminateLeap);
SNAP_NONRef_Fit = SNAP_NONRef_Fit(:,EliminateLeap);
SNAP_Ref_Fit = SNAP_Ref_Fit(:,EliminateLeap);
%% Downsample the SNAP Temps at reference elevation to the Test Stations
% This is still slow, and is dependent on the Test/Fit split
load("data\SNAP_Ref_Stat.mat") % This also loads SNAP_Ref_Test
SNAP_Ref_TestDays = CWS_SNAP_Rect.Days(SNAPDays_inTest); 
EliminateLeap = find(~(month(SNAP_Ref_TestDays)==2 & day(SNAP_Ref_TestDays)==29));
SNAP_Ref_TestDays = SNAP_Ref_TestDays(EliminateLeap);
SNAP_Ref_Test = SNAP_Ref_Test(:,EliminateLeap);
        
%  figure
%  plot(CWS_SNAP_Rect.Days(SNAPDays_inTest),SNAP_Ref_Test(1,:));
%  hold on
%  plot(CWS_SNAP_Rect.Days(TheseDaysIndex),SNAP_NONRef_Test(1,:));
 %% Downsample Station data at reference elevation to test station locations
 load("data\Station_Ref_Test.mat")
 Station_Ref_TestDays = FitDays(FitStationDays_inTest);

%% Interpolate Non Ref Training Station Data to test station locations
%test_x = [StationX(TestSet) StationY(TestSet)];
% We want this at all the days up to 2020, this will be used to fit an ecdf
TheseDays = CWS_SNAP_Rect.Days(CWS_SNAP_Rect.Days<=datetime(2019,12,31));
%TheseDays = CWS_SNAP_Rect.Days(CWS_SNAP_Rect.Days<=datetime(1970,1,5));
logical_Index = ismember(TestDays,TheseDays);
TheseDaysIndex = find(logical_Index);
%Station_NONRef_Test = InterpolateStation(xStationFit,yStationFit_NonRef,test_x,TheseDaysIndex);
load("data\Station_NONRef_Test.mat")
Station_NONRef_TestDays = FitDays(TheseDaysIndex);
% figure
% plot(FitDays(FitStationDays_inTest),Station_Ref_Test(1,:));
% hold on
% plot(FitDays(TheseDaysIndex),Station_NONRef_Test(1,:));
%% Find the ECDF functions
% This is done using the non-reference data we have interpolated to the
% test stations, we want two ecdf functions for each month, one for the
% SNAP data and one for the Station data.
SNAPMonth = cell(12,3);
StationMonth = cell(12,3);
SNAP_DeBias_Ref = zeros(size(SNAP_Ref_Test));
SNAP_FutureDeBias_Ref = zeros(size(SNAP_Ref_FutureTest));
for i=1:12
    numdays_thismonth = eomday(2023,i);  % Choose a leap year, I think thats a NAN anyway
    for j=1:numdays_thismonth
        WhereinSNAP = month(SNAP_NONRef_TestDay)==i & day(SNAP_NONRef_TestDay)==j; 
        SNAPMonth{i,1} = [SNAPMonth{i}; reshape(SNAP_Ref_Test2(:,WhereinSNAP),[],1)];
        WhereinStation = month(Station_Ref_TestDays)==i & day(Station_Ref_TestDays)==j;
        StationMonth{i,1} = [StationMonth{i}; reshape(Station_Ref_Test(:,WhereinStation),[],1)];
    end
    [StationMonth{i,2}, StationMonth{i,3}] = ecdf(StationMonth{i,1});
    [SNAPMonth{i,2}, SNAPMonth{i,3}] = ecdf(SNAPMonth{i,1});
     
    % Now that we have the ECDF functions, go through and apply them to the
    % SNAP Ref data
    for j=1:numdays_thismonth
        WhereinSNAPREF = month(SNAP_NONRef_TestDay)==i & day(SNAP_NONRef_TestDay)==j; 
        WhereinSNAPFuture = month(SNAP_Ref_FutureTestDays)==i & day(SNAP_Ref_FutureTestDays)==j; 
        f1 = @(x) interp1(SNAPMonth{i,3}(2:end),SNAPMonth{i,2}(2:end),x,"nearest",'extrap'); % x,f
        f2 = @(x) interp1(StationMonth{i,2}(2:end),StationMonth{i,3}(2:end),x,'nearest','extrap'); % f,x
        SNAP_DeBias_Ref(:,WhereinSNAPREF) = arrayfun(f1,SNAP_Ref_Test2(:,WhereinSNAPREF));
        SNAP_DeBias_Ref(:,WhereinSNAPREF) = arrayfun(f2,SNAP_DeBias_Ref(:,WhereinSNAPREF));
       
        SNAP_FutureDeBias_Ref(:,WhereinSNAPFuture) = arrayfun(f1,SNAP_Ref_FutureTest(:,WhereinSNAPFuture));
        SNAP_FutureDeBias_Ref(:,WhereinSNAPFuture) = arrayfun(f2,SNAP_FutureDeBias_Ref(:,WhereinSNAPFuture));
    end
end

% Now Move the DeBias_Ref Temps back to actual elevation using the known
% elevation of each test station
SNAP_DeBias_NonRef = zeros(size(SNAP_DeBias_Ref));
SNAP_FutureDeBias_NonRef = zeros(size(SNAP_FutureDeBias_Ref));
for i=1:length(TestSet)
    StationZ = mean(AllStationData{TestSet(i),2}).*.3048;
    SNAP_DeBias_NonRef(i,:) = SNAP_DeBias_Ref(i,:)-StationDownScaleParams(3)*(StationZ-StationDownScaleParams(4));
    SNAP_FutureDeBias_NonRef(i,:) = SNAP_FutureDeBias_Ref(i,:)-StationDownScaleParams(3)*(StationZ-StationDownScaleParams(4));
end

%% We need a trainx, trainy set to pass into the pre-trained model

% Build the train set here, have to do all training
StationBiasTrain = [];
StationDatesTrain = [];
StationElevTrain = [];

for i = 1:size(AllStationData(FitSet),2)
    TheseStationTmax = [AllStationData{FitSet(i), 4}];
    
    StationDay =       [AllStationData{FitSet(i),3}]; % Days that we have station data
    %StationDay = StationDay(year(StationDay) >= 2000); 
    AllFitDays =  union(StationDay,SNAP_NONRef_FitDay); % All the days we want to train on (10950:end)
    TheseStationElev = [AllStationData{FitSet(i),2}];
    StationZ = mean(AllStationData{FitSet(i),2}).*.3048;
    [~,WhereinSNAPNONRef, WhereinAllFitDay1] = intersect(SNAP_NONRef_FitDay, AllFitDays);
    [~,WhereinStationDay, WhereinAllFitDay2] = intersect(StationDay,AllFitDays);    % Where the Station Data is
    
    [StationANDSNAP, ~, ib] = intersect(WhereinAllFitDay1,WhereinAllFitDay2);
    [~,Instation,InSnap] = intersect(StationDay,SNAP_NONRef_FitDay);
    SnapTemp = nan(length(AllFitDays),1);
    SnapTemp(WhereinAllFitDay1) = SNAP_Ref_Fit(i,WhereinSNAPNONRef); 
    SnapTemp = SnapTemp-StationDownScaleParams(3)*(StationDownScaleParams(4)-StationZ); % Compare the bias to a topo downscaled!
    TheseStationTmax=TheseStationTmax-StationDownScaleParams(3)*(StationZ-StationDownScaleParams(4)); % Compare the bias to a topo downscaled!
    StationBiasTrain = [StationBiasTrain; (TheseStationTmax(Instation)-SnapTemp(StationANDSNAP))];
    StationDatesTrain = [StationDatesTrain; StationDay(Instation)];
    StationElevTrain = [StationElevTrain; TheseStationElev(Instation)];

end
[~,TrainStationDateIndex] = ismember(StationDatesTrain,CWS_SNAP_Rect.Days);
train_x = [TrainStationDateIndex StationElevTrain];
train_y = StationBiasTrain;


%%
% Test all the station locations all the way into the future
StationBiasTest = [];
StationDatesTest = [];
StationElevTest = [];
WhereinSNAPNONRef = cell(length(TestSet),1);
WhereinStationDay = cell(length(TestSet),1);
WhereinAllTestDay1 = cell(length(TestSet),1);
WhereinAllTestDay2 = cell(length(TestSet),1);
WhereinAllTestDay3 = cell(length(TestSet),1);
WhereinSNAPFuture = cell(length(TestSet),1);
WhereinStationTest = cell(length(TestSet),1);
WhereinSNAPTest = cell(length(TestSet),1);
WhichFitStation = [];
KnownTestDates = [];
for i = 1:size(AllStationData(TestSet),2)
    TheseStationTmax = [AllStationData{TestSet(i), 4}];
    StationDay =       [AllStationData{TestSet(i),3}]; % Days that we have station data
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Here is where we choose the days to evaluate on, they are all at the
    % test locations
    %StationDay = StationDay(year(StationDay) >= 2000); 
    AllTestDays = union(StationDay,SNAP_NONRef_FutureTestDays); % All the days we want to test
    %AllTestDays = union(StationDay,SNAP_NONRef_FutureTestDays([1 end]));
    %AllTestDays = SNAP_NONRef_FutureTestDays;
    %AllTestDays = [StationDay];
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    ThisStationElev =  mean([AllStationData{TestSet(i),2}]); % Elevation of station i
    TheseStationTmax=TheseStationTmax-StationDownScaleParams(3)*(ThisStationElev-StationDownScaleParams(4)); % Compare the bias to a topo downscaled!
    [~,WhereinSNAPNONRef{i}, WhereinAllTestDay1{i}] = intersect(SNAP_NONRef_TestDay, AllTestDays); % Where the old SNAP_NONRef is
    [~,WhereinSNAPFuture{i}, WhereinAllTestDay2{i}] = intersect(SNAP_NONRef_FutureTestDays, AllTestDays);% Where the future SNAP is
    [~,WhereinStationDay{i}, WhereinAllTestDay3{i}] = intersect(StationDay,AllTestDays);    % Where the Station Data is
    [~,Instation,InSnap] = intersect(StationDay,SNAP_NONRef_TestDay);
    %[TheseKnownTestDates,WhereinStationTest{i}, WhereinSNAPTest{i}] = intersect(StationDay,SNAP_NONRef_TestDay);
    [StationANDSNAP, ~, ib] = intersect(WhereinAllTestDay1{i},WhereinAllTestDay3{i});
    SnapTemp = nan(length(AllTestDays),1);
    SnapTemp(WhereinAllTestDay1{i}) = SNAP_Ref_Test2(i,WhereinSNAPNONRef{i}); 
    SnapTemp(WhereinAllTestDay2{i}) = SNAP_NONRef_FutureTest(i,WhereinSNAPFuture{i});
    SnapTemp = SnapTemp-StationDownScaleParams(3)*(StationDownScaleParams(4)-StationZ); % Compare the bias to a topo downscaled!
    WhichFitStation  = [WhichFitStation; i.*ones(length(AllTestDays),1)]; 
    if ~isempty(StationANDSNAP)
        StationBiasTest =  [StationBiasTest; (TheseStationTmax(Instation)-SnapTemp(StationANDSNAP))];
    end
    %TheseStationTmax(WhereinStationDay{i})-GP_DeBias{i}(WhereinAllTestDay3{i});
    StationDatesTest = [StationDatesTest; AllTestDays];
    %KnownTestDates = [KnownTestDates; TheseKnownTestDates];
    StationElevTest =  [StationElevTest; ThisStationElev.*ones(length(AllTestDays),1)];
end
[ia,TestStationDateIndex] = ismember(StationDatesTest,CWS_SNAP_Rect.Days);
GTtest_y = StationBiasTest;
% train_x and train_y come from the file load
test_x_new = [TestStationDateIndex StationElevTest];

%% Perform Prediction on this new test set, using the existing trained model
TrainFlag = 1;
SubFitTest = [];
% Build and save the data for the GpyTorch inference
[~, SubFitTest] = FutureBias(train_x,train_y,test_x_new,SubFitTest,TrainFlag);

% Analysis Contingent on Loading the Python Data
system('python data\GSML_GPGRID.py');
load('PythonResults.mat')
trainXout = double(trainXout);
trainYout = double(trainYout);
testXout = double(testXout);
testYout = double(testYout);

%% RMSE Code  
GPBiasTestDays = StationDatesTest(ia);
GP_DeBias = cell(length(TestSet),1);
GP_DeBiasDays = cell(length(TestSet),1);
WhereinStationDay = cell(length(TestSet),1);
GT_Bias = cell(length(TestSet),1);
GPDeBiasError = cell(length(TestSet),1);
EQMBias = cell(length(TestSet),1);
for i=1:length(TestSet)
    TheseStationTmax = [AllStationData{TestSet(i), 4}];
    StationDay =       [AllStationData{TestSet(i),3}];
    %StationDay = StationDay(year(StationDay) >= 2000); 
    ThisGPBias = testYout(WhichFitStation==i);
    [StationANDSNAP, ~, ~] = intersect(WhereinAllTestDay1{i},WhereinAllTestDay3{i});
    [~,Instation,~] = intersect(StationDay,SNAP_NONRef_TestDay);
    [~,Instation2,InSnapDeBias] = intersect(StationDay,SNAP_Ref_TestDays);
    SnapTemp = nan(length(ThisGPBias),1);
    SnapTemp(WhereinAllTestDay1{i}) = SNAP_Ref_Test2(i,WhereinSNAPNONRef{i}); 
    SnapTemp(WhereinAllTestDay2{i}) = SNAP_NONRef_FutureTest(i,WhereinSNAPFuture{i});
    StationZ = mean(AllStationData{FitSet(i),2}).*.3048;
    
    SnapTemp = SnapTemp-StationDownScaleParams(3)*(StationDownScaleParams(4)-StationZ);
    GP_DeBias{i} = (SnapTemp+ThisGPBias')-StationDownScaleParams(3)*(StationZ-StationDownScaleParams(4));
    
    GP_DeBiasDays{i} = GPBiasTestDays(WhichFitStation==i);
    %(TheseStationTmax(WhereinStationTest{i})'-SNAP_NONRef_Test(i,WhereinSNAPTest{i}))'
    if ~isempty(StationANDSNAP)
        GPDeBiasError{i} = TheseStationTmax(Instation)-GP_DeBias{i}(StationANDSNAP);
        GT_Bias{i} = TheseStationTmax(Instation)-SnapTemp(StationANDSNAP);
        EQMBias{i} = TheseStationTmax(Instation2)-SNAP_DeBias_NonRef(i,InSnapDeBias)';
    end
    
end

ALLGTBias = cell2mat(GT_Bias);
EQM_All = cell2mat(EQMBias);
EQM_RMSE = sqrt(sum(EQM_All.^2)/length(EQM_All));
RawWRF_RMSE = sqrt(sum(ALLGTBias.^2)/length(ALLGTBias));
AllDeBiasError = cell2mat(GPDeBiasError);
AllDeBiasError = AllDeBiasError(~isnan(AllDeBiasError));
GPDeBiasRMSE = sqrt(sum(AllDeBiasError.^2)/length(AllDeBiasError));

% NewBias = nan(length(TestStationDateIndex),1);
% NewBias(ValidRange) = testYout;
% GP_DeBias = reshape(NewBias, size(Xdown,1), []);
% figure
% plot(AlaskaPoly,'FaceColor','none')
% hold on
% mapshow(CWS_SNAP_Down.Xgrid,CWS_SNAP_Down.Ygrid,SNAP_Ref_Down.Tmax+GP_DeBias,'DisplayType','surface','FaceAlpha',1);

%% Temporal Plots at a given test station
%Plot for an individual test station 
whichTestStation = 2;
whichStation = TestSet(whichTestStation);
figure
plot(AllStationData{whichStation,3},AllStationData{whichStation, 4})
hold on
%plot(Station_NONRef_TestDays,Station_NONRef_Test(whichTestStation ,:));
plot(SNAP_NONRef_TestDay,SNAP_NONRef_Test(whichTestStation ,:));
% plot(SNAP_NONRef_FutureTestDays,SNAP_NONRef_FutureTest(whichTestStation ,:))
%plot(SNAP_Ref_TestDays,SNAP_DeBias_NonRef(whichTestStation ,:));
plot([GP_DeBiasDays{whichTestStation}],[GP_DeBias{whichTestStation}],'color',"#77AC30")
%legend('Measured Station Data','Biased Interpolated SNAP Data','Biased Interpolated SNAP Data','DeBiased SNAP', 'DeBiased GP')
legend('Measured Station Data','Biased Interpolated WRF Data','GP (Bias Removed)')
% %thisvector = SNAP_NONRef_Test(whichTestStation,:);
% %ylim([min(thisvector(thisvector>0))-10 max(thisvector)+10])
title(['Bias Snapshot, Station ', num2str(StationLL(whichStation,1)), 'N ', num2str(StationLL(whichStation,2)), 'W'])
xlim([datetime(1990,1,1) datetime(1998,1,1)])
xlabel('Time')
ylabel('Temperature (K)')

% 3D scaled domain plots
figure
stem3(testXout(:,1),testXout(:,2),testYout,'.');

% % Future Comparison Plots at a test station
whichTestStation = 2;
whichStation = TestSet(whichTestStation);
TheseGPdays = find([GP_DeBiasDays{whichTestStation}]>SNAP_NONRef_FutureTestDays(1));
GPdays = [GP_DeBiasDays{whichTestStation}];
GPdays = GPdays(TheseGPdays);
GPTemps = [GP_DeBias{whichTestStation}];
GPTemps = GPTemps(TheseGPdays);
figure
    hold on
    plot(SNAP_NONRef_FutureTestDays,SNAP_NONRef_FutureTest(whichTestStation ,:),'color',"#D95319")
    plot(SNAP_NONRef_FutureTestDays,SNAP_FutureDeBias_NonRef(whichTestStation ,:),'color',"#EDB120");
    plot(GPdays,GPTemps,'color',"#77AC30")
    legend('Biased WRF Data','DeBiased WRF','GP (Bias Removed)')
    title(['Future Predictions, Station ', num2str(StationLL(whichStation,1)), 'N ', num2str(StationLL(whichStation,2)), 'W'])
    xlabel('Time')
    ylabel('Temperature (K)')

% % Some rough future Calculations for comparison
% AllTheseDays = union(SNAP_NONRef_FutureTestDays,GPdays);
% [~,SNAPDateIndex] = ismember(SNAP_NONRef_FutureTestDays,AllTheseDays);
% [~,GPDateIndex] = ismember(GPdays,AllTheseDays);
% RAWSNAPLine = polyfit(SNAPDateIndex,SNAP_NONRef_FutureTest(whichTestStation ,:),1);
% SNAPLine = polyfit(SNAPDateIndex,SNAP_FutureDeBias_NonRef(whichTestStation ,:),1);
% GPLine = polyfit(GPDateIndex,GPTemps,1);
% figure
%     hold on
%     plot(SNAP_NONRef_FutureTestDays,polyval(RAWSNAPLine,SNAPDateIndex),'color',"#D95319")
%     plot(SNAP_NONRef_FutureTestDays,polyval(SNAPLine,SNAPDateIndex),'color',"#EDB120");
%     plot(GPdays,polyval(GPLine,GPDateIndex),'color',"#77AC30")
%     legend('Biased WRF Data','DeBiased WRF','GP (Bias Removed)')
%     title(['Future Linear Fit, Station ', num2str(StationLL(whichStation,1)), 'N ', num2str(StationLL(whichStation,2)), 'W'])
%     xlabel('Time')
%     ylabel('Temperature (K)')
%     grid on

%% Spatial Plots at a given day
% Apply the EQM to the entire downsampled domain at a given day
% First find the WRF data moved to the reference temp
% CW
SNAP_Ref_Down.Tmax = zeros(size(CWS_SNAP_Down.Tmax));
for i=1:length(CWS_SNAP_Down.Days)
    SNAP_Ref_Down.Tmax(:,:,i) = CWS_SNAP_Down.Tmax(:,:,i);%-StationDownScaleParams(3)*(StationDownScaleParams(4)-CWS_SNAP_Down.Elevation);
end
SNAP_Ref_Down.Days = CWS_SNAP_Down.Days;
for i=1:12
    numdays_thismonth = eomday(2023,i);  
    % Now that we have the ECDF functions, go through and apply them
    for j=1:numdays_thismonth
        WhereinSNAPREF = month(SNAP_Ref_Down.Days)==i & day(SNAP_Ref_Down.Days)==j;
        if (WhereinSNAPREF)
            f1 = @(x) interp1(SNAPMonth{i,3}(2:end),SNAPMonth{i,2}(2:end),x,"nearest",'extrap'); % x,f
            f2 = @(x) interp1(StationMonth{i,2}(2:end),StationMonth{i,3}(2:end),x,'nearest','extrap'); % f,x
            SNAP_DeBiasRef_Down.Tmax(:,:,WhereinSNAPREF) = arrayfun(f1,SNAP_Ref_Down.Tmax(:,:,WhereinSNAPREF));
            SNAP_DeBiasRef_Down.Tmax(:,:,WhereinSNAPREF) = arrayfun(f2,SNAP_DeBiasRef_Down.Tmax(:,:,WhereinSNAPREF));
        end
    end
end

% Now Move the DeBias_Ref Temps back to actual elevation using the known
% elevation of each test station
SNAP_DeBiasNonRef_Down.Tmax = zeros(size(CWS_SNAP_Down.Tmax));
for i=1:length(CWS_SNAP_Down.Days)
    SNAP_DeBiasNonRef_Down.Tmax(:,:,i) = SNAP_DeBiasRef_Down.Tmax(:,:,i)-StationDownScaleParams(3)*(CWS_SNAP_Down.Elevation-StationDownScaleParams(4));
end
TestDays= [];
TestElevations = [];
for i=1:length(CWS_SNAP_Down.Days)
    TestDays = [TestDays; repmat(CWS_SNAP_Down.Days(i),1,numel(CWS_SNAP_Down.Elevation))'];
    TestElevations = [TestElevations; reshape(CWS_SNAP_Down.Elevation,[],1)];
end

[~,TestStationDateIndex] = ismember(TestDays,CWS_SNAP_Rect.Days); % This indexing needs to be constant
ValidRange = find(~isnan(TestElevations)); % Many NaNs in the elevation data
test_x_new = [TestStationDateIndex(ValidRange) TestElevations(ValidRange)];

% Perform Prediction on this new test set, using the existing trained model
TrainFlag = 0;
[~, SubFitTest] = FutureBias(train_x,train_y,test_x_new,SubFitTest,TrainFlag);
disp('Data Built for new test set, run the Python, press a key')
pause();
load('Z:\PythonResultsPredict.mat')
trainXout = double(trainXout);
trainYout = double(trainYout);
testXout = double(testXout);
testYout = double(testYout);

NewBias = nan(length(TestStationDateIndex),1);
NewBias(ValidRange) = testYout;
GP_DeBias = reshape(NewBias, size(Xdown,1), []);
%GP_DeBias = (SNAP_Ref_Down.Tmax+GP_DeBias)-StationDownScaleParams(3)*(CWS_SNAP_Down.Elevation-StationDownScaleParams(4));
GP_DeBias = (SNAP_Ref_Down.Tmax+GP_DeBias)-StationDownScaleParams(3)*(CWS_SNAP_Down.Elevation-StationDownScaleParams(4));
% Side by Side Figure Comparison
figure
% Make sure we have the same colorbar
bottom = min(min(min(GP_DeBias)),min(min(SNAP_DeBiasNonRef_Down.Tmax(:,:,1))));
top  = max(max(max(GP_DeBias)),max(max(SNAP_DeBiasNonRef_Down.Tmax(:,:,1))));
subplot(1,3,1)
% plot(AlaskaPoly,'FaceColor','none')
% hold on
mapshow(CWS_SNAP_Down.Xgrid,CWS_SNAP_Down.Ygrid,SNAP_Ref_Down.Tmax-StationDownScaleParams(3)*(CWS_SNAP_Down.Elevation-StationDownScaleParams(4)),'DisplayType','surface','FaceAlpha',1);
caxis manual
caxis([bottom top]);
colorbar
title('No Bias Removal, April 26, 2023')
subplot(1,3,2)
% plot(AlaskaPoly,'FaceColor','none')
% hold on
mapshow(CWS_SNAP_Down.Xgrid,CWS_SNAP_Down.Ygrid,GP_DeBias,'DisplayType','surface','FaceAlpha',1);
caxis manual
caxis([bottom top]);
colorbar
title('GP DeBiasing, April 26, 2023')
subplot(1,3,3)
% plot(AlaskaPoly,'FaceColor','none')
% hold on
mapshow(CWS_SNAP_Down.Xgrid,CWS_SNAP_Down.Ygrid,SNAP_DeBiasNonRef_Down.Tmax(:,:,1),'DisplayType','surface','FaceAlpha',1);
caxis manual
caxis([bottom top]);
colorbar
title('EQM DeBiasing, April 26, 2023')


%% Function Files
function [CWS_SNAP_Rect] = MovetoRectilinear_Interpolate(SNAPx,SNAPy,CWS_SNAPData)
    % Force the SNAP grid to be rectilinear so we can use gridded kernel
    % interpolation with our GP, this is essentially a regridding (a teeny one), so we should interpolate to find our values at the new grid.   
    [SNAP_Rect, CWS_SNAP_Rect.lineParams] = CurveGrid2Rect(SNAPx,SNAPy);
    vectorX = reshape(SNAPx,[],1);
    vectorY = reshape(SNAPy,[],1);
    CWS_SNAP_Rect.t2max = zeros(size(SNAPx,1),size(SNAPy,2),size(CWS_SNAPData.t2max,3));
    for i=1:size(CWS_SNAPData.t2max,3) % Interpolate Each day of temp data to the new grid
           vectorT = reshape(double(CWS_SNAPData.t2max(:,:,i)),[],1);
           F = scatteredInterpolant(vectorX,vectorY,vectorT);
           CWS_SNAP_Rect.t2max(:,:,i) = reshape(F(SNAP_Rect(:,1),SNAP_Rect(:,2)),size(SNAPx,2),size(SNAPx,1))';
    end
    CWS_SNAP_Rect.xgrid = reshape(SNAP_Rect(:,1),size(SNAPx,2),size(SNAPx,1))';
    CWS_SNAP_Rect.ygrid = reshape(SNAP_Rect(:,2),size(SNAPx,2),size(SNAPx,1))';
    CWS_SNAP_Rect.Days = CWS_SNAPData.Days;
end

function [DownSampleOut] = InterpolateSNAP(train_x,SNAPTemps,test_x,TheseDays)
% This Function Downsamples the SNAP data to a set of locations, the xtrain
% must be gridded! This is called on the local machine
    % Here is some manual scaling for the grid
    train_x = train_x./1e6;
    test_x = test_x./1e6;
    % Use GpyTorch to krig the temp data 
    % Move everything to numpy, then to pytorch
    pyTrainingX = py.torch.from_numpy(py.numpy.array(train_x));
    pyGrid = py.torch.from_numpy(py.numpy.array(train_x));
    pyTestX = py.torch.from_numpy(py.numpy.array(test_x));
    % We need the days in the TestSet
    DownSampleOut = zeros(size(test_x,1),length(TheseDays));
    for i=1:length(TheseDays) %:size(CWS_SNAP_Rect.t2max,3)
         disp(['Downsampling Day ', num2str(i)])
         train_y = reshape(SNAPTemps(:,:,TheseDays(i)),[],1);
         pyTrainingY = py.torch.from_numpy(py.numpy.array(train_y));
         outvars = py.GSML_GPGRID.GPGrid(pyTrainingX,pyTrainingY, pyGrid, pyTestX);
         outcell = cell(outvars);
         %trainXout = double(outcell{1}.cpu().numpy);
         %trainYout = double(outcell{2}.cpu().numpy);
         %testXout = double(outcell{3}.cpu().numpy);
         testYout = double(outcell{4}.cpu().numpy);
         %testPred = double(outcell{4}.cpu().numpy);
         %myparam = py.torch.from_numpy(py.numpy.array(double(outcell{5})));
         DownSampleOut(1:size(test_x,1),i) = testYout';
    end
end

function [DownSampleOut,SubFitSet] = FutureBias(train_x,train_y,test_x,SubFitSet,TrainFlag)
    if TrainFlag == 1    
        NumberTrainPoints = 75000;
        SubFitSet = sort(randperm(size(train_x,1),NumberTrainPoints));
    end
    train_x = [train_x(SubFitSet,1) (train_x(SubFitSet,2).*.3048)];
    train_y = train_y(SubFitSet);
    test_x(:,1) = test_x(:,1);
    test_x(:,2) = (test_x(:,2).*.3048);
    save('Z:\PythonTrainTest','train_x','train_y','test_x','TrainFlag');
    DownSampleOut = [];
end