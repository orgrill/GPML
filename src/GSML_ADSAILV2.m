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
%load('data\xyStationFit.mat');


%%%%%%%%%%%%%%%%%%
%  BuildTrainSet %
%%%%%%%%%%%%%%%%%%

%% Perform Prediction on this new test set, using the existing trained model
TrainFlag = 1;
SubFitTest = [];
% Build and save the data for the GpyTorch inference
[~, SubFitTest] = FutureBias(train_x,train_y,test_x_new,SubFitTest,TrainFlag);

% Analysis Contingent on Loading the Python Data

%%% Not sure how to do this part correctly %%%
%%% probably won't work because the py file is working in the Z: path on
%%% the adsail machine
system('GSML_GPBias_Remote.py');
load(fullfile("data","PythonResults.mat"))
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
% mapshow(CWS_SNAP_Downscaled.Xgrid,CWS_SNAP_Downscaled.Ygrid,SNAP_Ref_Down.Tmax+GP_DeBias,'DisplayType','surface','FaceAlpha',1);

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
SNAP_Ref_Down.Tmax = zeros(size(CWS_SNAP_Downscaled.Tmax));
for i=1:length(CWS_SNAP_Downscaled.Days)
    SNAP_Ref_Down.Tmax(:,:,i) = CWS_SNAP_Downscaled.Tmax(:,:,i);%-StationDownScaleParams(3)*(StationDownScaleParams(4)-CWS_SNAP_Downscaled.Elevation);
end
SNAP_Ref_Down.Days = CWS_SNAP_Downscaled.Days;
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
SNAP_DeBiasNonRef_Down.Tmax = zeros(size(CWS_SNAP_Downscaled.Tmax));
for i=1:length(CWS_SNAP_Downscaled.Days)
    SNAP_DeBiasNonRef_Down.Tmax(:,:,i) = SNAP_DeBiasRef_Down.Tmax(:,:,i)-StationDownScaleParams(3)*(CWS_SNAP_Downscaled.Elevation-StationDownScaleParams(4));
end
TestDays= [];
TestElevations = [];
for i=1:length(CWS_SNAP_Downscaled.Days)
    TestDays = [TestDays; repmat(CWS_SNAP_Downscaled.Days(i),1,numel(CWS_SNAP_Downscaled.Elevation))'];
    TestElevations = [TestElevations; reshape(CWS_SNAP_Downscaled.Elevation,[],1)];
end

[~,TestStationDateIndex] = ismember(TestDays,CWS_SNAP_Rect.Days); % This indexing needs to be constant
ValidRange = find(~isnan(TestElevations)); % Many NaNs in the elevation data
test_x_new = [TestStationDateIndex(ValidRange) TestElevations(ValidRange)];

% Perform Prediction on this new test set, using the existing trained model
TrainFlag = 0;
[~, SubFitTest] = FutureBias(train_x,train_y,test_x_new,SubFitTest,TrainFlag);
disp('Data Built for new test set, run the Python, press a key')
pause();
load(fullfile("data", "PythonResultsPredict.mat")
trainXout = double(trainXout);
trainYout = double(trainYout);
testXout = double(testXout);
testYout = double(testYout);

NewBias = nan(length(TestStationDateIndex),1);
NewBias(ValidRange) = testYout;
GP_DeBias = reshape(NewBias, size(Xdown,1), []);
%GP_DeBias = (SNAP_Ref_Down.Tmax+GP_DeBias)-StationDownScaleParams(3)*(CWS_SNAP_Downscaled.Elevation-StationDownScaleParams(4));
GP_DeBias = (SNAP_Ref_Down.Tmax+GP_DeBias)-StationDownScaleParams(3)*(CWS_SNAP_Downscaled.Elevation-StationDownScaleParams(4));
% Side by Side Figure Comparison
figure
% Make sure we have the same colorbar
bottom = min(min(min(GP_DeBias)),min(min(SNAP_DeBiasNonRef_Down.Tmax(:,:,1))));
top  = max(max(max(GP_DeBias)),max(max(SNAP_DeBiasNonRef_Down.Tmax(:,:,1))));
subplot(1,3,1)
% plot(AlaskaPoly,'FaceColor','none')
% hold on
mapshow(CWS_SNAP_Downscaled.Xgrid,CWS_SNAP_Downscaled.Ygrid,SNAP_Ref_Down.Tmax-StationDownScaleParams(3)*(CWS_SNAP_Downscaled.Elevation-StationDownScaleParams(4)),'DisplayType','surface','FaceAlpha',1);
caxis manual
caxis([bottom top]);
colorbar
title('No Bias Removal, April 26, 2023')
subplot(1,3,2)
% plot(AlaskaPoly,'FaceColor','none')
% hold on
mapshow(CWS_SNAP_Downscaled.Xgrid,CWS_SNAP_Downscaled.Ygrid,GP_DeBias,'DisplayType','surface','FaceAlpha',1);
caxis manual
caxis([bottom top]);
colorbar
title('GP DeBiasing, April 26, 2023')
subplot(1,3,3)
% plot(AlaskaPoly,'FaceColor','none')
% hold on
mapshow(CWS_SNAP_Downscaled.Xgrid,CWS_SNAP_Downscaled.Ygrid,SNAP_DeBiasNonRef_Down.Tmax(:,:,1),'DisplayType','surface','FaceAlpha',1);
caxis manual
caxis([bottom top]);
colorbar
title('EQM DeBiasing, April 26, 2023')