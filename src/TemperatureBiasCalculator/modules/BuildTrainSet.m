function [train_x, train_y, train_x_new] = BuildTrainSet(AllStationData, FitSet, SNAP_Ref_Fit, SNAP_NONRef_FitDay, CWS_SNAP_Rect)
    % Build the train set here, have to do all training
    StationBiasTrain = [];
    StationDatesTrain = [];
    StationElevTrain = [];

    for i = 1:size(AllStationData(FitSet),2)
        TheseStationTmax = [AllStationData{FitSet(i), 4}];
        
        StationDay = [AllStationData{FitSet(i),3}]; % Days that we have station data
        %StationDay = StationDay(year(StationDay) >= 2000); 
        AllFitDays = union(StationDay,SNAP_NONRef_FitDay); % All the days we want to train on (10950:end)
        TheseStationElev = [AllStationData{FitSet(i),2}];
        StationZ = mean(AllStationData{FitSet(i),2}).*.3048;
        [~, WhereinSNAPNONRef,  WhereinAllFitDay1] = intersect(SNAP_NONRef_FitDay, AllFitDays);
        [~, WhereinStationDay, WhereinAllFitDay2] = intersect(StationDay, AllFitDays);    % Where the Station Data is
        
        [StationANDSNAP,  ~,  ib] = intersect(WhereinAllFitDay1, WhereinAllFitDay2);
        [~, Instation, InSnap] = intersect(StationDay, SNAP_NONRef_FitDay);
        SnapTemp = nan(length(AllFitDays), 1);
        SnapTemp(WhereinAllFitDay1) = SNAP_Ref_Fit(i, WhereinSNAPNONRef); 
        SnapTemp = SnapTemp-StationDownScaleParams(3)*(StationDownScaleParams(4)-StationZ); % Compare the bias to a topo downscaled!
        TheseStationTmax = TheseStationTmax - StationDownScaleParams(3) * (StationZ-StationDownScaleParams(4)); % Compare the bias to a topo downscaled!
        StationBiasTrain = [StationBiasTrain; (TheseStationTmax(Instation)-SnapTemp(StationANDSNAP))];
        StationDatesTrain = [StationDatesTrain; StationDay(Instation)];
        StationElevTrain = [StationElevTrain; TheseStationElev(Instation)];

    end
    [~, TrainStationDateIndex] = ismember(StationDatesTrain, CWS_SNAP_Rect.Days);
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
    [ia,TestStationDateIndex] = ismember(StationDatesTest, CWS_SNAP_Rect.Days);
    % GTtest_y = StationBiasTest;
    % train_x and train_y come from the file load
    test_x_new = [TestStationDateIndex StationElevTest];

    return 
end