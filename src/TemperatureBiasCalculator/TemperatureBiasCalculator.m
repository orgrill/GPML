classdef TemperatureBiasCalculator
    properties (SetAccess = immutable)
        % Input Data
        AllStationData  
        CWS_Projection
        SNAP_Data

        % Options
        Skip_Transform
    end

    methods
        function self = TemperatureBiasCalculator(AllStationData, CWS_Projection, SNAP_Data, Skip_Transform)
            self.AllStationData = AllStationData;
            self.CWS_Projection = CWS_Projection;
            self.SNAP_Data = SNAP_Data;
            self.Skip_Transform = Skip_Transform;
        end

        function result = CalculateBias(self) 
            addpath(fullfile("src", "TemperatureBiasCalculator","modules"));

            % TODO: Unused?
            %SubRegion = kmz2struct(fullfile("data", "CopperRiverWatershed.kmz"));

            % TODO: What is this
            % load('xyStationFit.mat');

            disp("---- Transforming Coordinates ----")
            SNAP_Rect = CoordinateTransform(self.CWS_Projection.ProjectedCRS, self.SNAP_Data, self.Skip_Transform);

            %{
                Split into test and fit sets, set up and execute EQM and topgraphical downscaling
                Load the raw station data, we need to divide this into two parts, 80/20
            %}
            disp("---- Selecting Test and Fit sets ----")
            FitSet = sort(randperm(size(AllStationData, 1), round(.8 * size(AllStationData, 1))));
            TestSet = sort(setdiff(1:length(AllStationData), FitSet));

            disp("---- Applying Topographical Downscaling ----")
            % Fit the Topographical Downscaling parameters (White 2016)
            StationDownScaleParams = GSML_Topo_Downscale(AllStationData, FitSet); %[T0; Beta; Gamma; zref]
            SNAP_Downscaled = TopographicalDownscaling(self.AllStationData, SNAP_Rect);

            
            %%%%%%%%%%%%%%%%%%%%%%%%%%

            FitDays = TemperatureBiasCalculator.GetFitSetConfig()
            TestDays = TemperatureBiasCalculator.GetTestSetConfig();

            %% Downsample Station data at reference elevation to test station locations
            % TODO: Where does this data come from? Can we compute it manually instead of loading from file?
            Station_Ref_Test = load(fullfile("data","Station_Ref_Test.mat")).Station_Ref_Test;
            % TODO: Why does Ref depend on TestDays (NONRef?) Is this a typo?
            Station_Ref_TestDays = FitDays(find(ismember(FitDays, TestDays)));

            % TODO: This is unused?
            % Station_NONRef_Test = load(fullfile("data", "Station_NONRef_Test.mat"), "Station_NONRef_Test");
            Station_NONRef_TestDays = FitDays(find(ismember(TestDays, selected_days)));


            [train_x, train_y, train_x_new] = BuildTrainSet(self.AllStationData, FitSet, SNAP_Ref_Fit, SNAP_NONRef_FitDays, SNAP_Rect);

            testResult = TemperateBiasCalculator.RunTest(testconfig);

            %%%%%%%%%%%%%%%%%%%%%%%%%%

            %{
                TODO: Replace this by not loading the files

                SNAP_NONRef_Fit_File = load(fullfile("data", "SNAP_NONRef_Fit.mat"));
                SNAP_NONRef_FitDay = SNAP_NONRef_Fit_File.SNAP_NONRef_FitDay
                SNAP_NONRef_Fit = SNAP_NONRef_Fit_File.SNAP_NONRef_Fit

                % Get rid of the leap days
                EliminateLeap = find(~(month(SNAP_NONRef_FitDay)==2 & day(SNAP_NONRef_FitDay)==29));
                SNAP_NONRef_FitDay = SNAP_NONRef_FitDay(EliminateLeap);
                SNAP_NONRef_Fit = SNAP_NONRef_Fit(:, EliminateLeap);
                SNAP_Ref_Fit = load(fullfile("data", "SNAP_Ref_Fit.mat")).SNAP_Ref_Fit(:, EliminateLeap);
            %}

            return 
        end
    end

    methods (Static)
        function FitDays = GetFitSetConfig(FitSet)
            FitDays = [];
            station_Lengths = cellfun(@length, AllStationData(FitSet, 3), 'UniformOutput', false);
            for i = 1:size(station_Lengths, 1)
                FitDays = [FitDays; [AllStationData{FitSet(i), 3}]];
            end
            FitDays = unique(FitDays);  % These are the days present in the fit stations
        end

        function TestDays = GetTestSetConfig(TestSet)
            %% Switch to the Test Set
            station_Lengths = cellfun(@length, AllStationData(TestSet, 3), 'UniformOutput', false);
            selected_days = SNAP_Rect.Days(SNAP_Rect.Days <= datetime(2019, 12, 31));

            % We want this at all the days up to 2020, this will be used to fit an ecdf
            SNAP_NONRef_TestDay = SNAP_Rect.Days(find(ismember(SNAP_Rect.Days, selected_days)));
            EliminateLeap = find(~(month(SNAP_NONRef_TestDay) == 2 & day(SNAP_NONRef_TestDay) == 29));
            SNAP_NONRef_TestDay = SNAP_NONRef_TestDay(EliminateLeap);

            % This is the reference WRF, interpolated to the fine grid
            % TODO: Where does this data come from? Can we compute it manually instead of loading from file?
            SNAP_Ref_Test2 = load(fullfile("data","SNAP_Ref_Test2.mat")).SNAP_Ref_Test2(:, EliminateLeap); 

            % TODO: Where does this data come from? Can we compute it manually instead of loading from file?
            SNAP_NONRef_Test = load(fullfile("data", "SNAP_NONRef_Test.mat")).SNAP_NONRef_Test(:, EliminateLeap);

            % This is raw WRF data interpolated to stations, into the future
            FutureTest = load(fullfile("data", "SNAP_NONRef_FutureTest.mat")); 
            SNAP_NONRef_FutureTest = FutureTest.SNAP_NONRef_Test(:, EliminateLeap);
            SNAP_NONRef_FutureTestDay = FutureTest.SNAP_NONRef_TestDay(EliminateLeap);

            %% We want the SNAP Data in the future at the reference elevation so we can apply the EQM to it,  
            SNAP_Ref_FutureTest = zeros(size(SNAP_NONRef_FutureTest));
            for i=1:length(TestSet)
                StationZ = mean(AllStationData{TestSet(i), 2}).*.3048;
                SNAP_Ref_FutureTest(i,:) = SNAP_NONRef_FutureTest(i, :)-StationDownScaleParams(3)*(StationDownScaleParams(4)-StationZ);
            end

            %{
                Interpolate the SNAP Temps NOT at reference elevation to the fit locations
                These will be used to find the raw bias, which will be fed to the Gaussian Process
            %}
            selected_days = SNAP_Rect.Days(SNAP_Rect.Days <= datetime(2022, 3, 22));
            selected_days_index = find(ismember(SNAP_Rect.Days, selected_days));
            test_x = [StationX(FitSet) StationY(FitSet)];
            SNAP_Ref_Fit = InterpolateSNAP(SNAP_Rect_Rot, SNAP_Rect.t2ref, test_x, selected_days_index);
            SNAP_NONRef_FitDays = SNAP_Rect.Days(selected_days_index);

            %% Downsample the SNAP Temps at reference elevation to the Test Stations
            % This is still slow, and is dependent on the Test/Fit split
            % TODO: Where does this data come from? Can we compute it manually instead of loading from file?
            %{
                SNAP_Ref_Stat_File = load(fullfile("data", "SNAP_Ref_Stat.mat")) 
                FitSet = SNAP_Ref_Stat_File.FitSet
                SNAP_Ref_Test = SNAP_Ref_Stat_File.SNAP_Ref_Test
                SNAPDays_inTest = SNAP_Ref_Stat_File.SNAPDays_inTest
                TestDays = SNAP_Ref_Stat_File.TestDays
                TestSet = SNAP_Ref_Stat_File.TestSet
            %}
            SNAP_Ref_TestDays = SNAP_Rect.Days(find(ismember(SNAP_Rect.Days, TestDays))); 
            EliminateLeap = find(~(month(SNAP_Ref_TestDays) == 2 & day(SNAP_Ref_TestDays) == 29));
            SNAP_Ref_TestDays = SNAP_Ref_TestDays(EliminateLeap);
            SNAP_Ref_Test = SNAP_Ref_Test(:, EliminateLeap);
                    
            %% Interpolate Non Ref Training Station Data to test station locations
            % We want this at all the days up to 2020, this will be used to fit an ecdf
            %test_x = [StationX(TestSet) StationY(TestSet)];
            % We want this at all the days up to 2020, this will be used to fit an ecdf
            selected_days = SNAP_Rect.Days(SNAP_Rect.Days<=datetime(2019,12,31));
            %selected_days = SNAP_Rect.Days(SNAP_Rect.Days<=datetime(1970,1,5));
            logical_Index = ismember(TestDays,selected_days);
            selected_days_index = find(logical_Index);

            %{
            Find the ECDF functions
                This is done using the non-reference data we have interpolated to the
                test stations, we want two ecdf functions for each month, one for the
                SNAP data and one for the Station data.
            %}
        end

        function [train_x, train_y, train_x_new] = BuildTrainSet(AllStationData, FitSet, SNAP_Ref_Fit, SNAP_NONRef_FitDays, SNAP_Rect)
            % Build the train set here, have to do all training
            StationBiasTrain = [];
            StationDatesTrain = [];
            StationElevTrain = [];

            for i = 1:size(AllStationData(FitSet), 2)
                TheseStationTmax = [AllStationData{FitSet(i), 4}];
                
                StationDay = [AllStationData{FitSet(i),3}]; % Days that we have station data
                %StationDay = StationDay(year(StationDay) >= 2000); 
                AllFitDays = union(StationDay, SNAP_NONRef_FitDays); % All the days we want to train on (10950:end)
                TheseStationElev = [AllStationData{FitSet(i), 2}];
                StationZ = mean(AllStationData{FitSet(i), 2}).*.3048;
                [~, WhereinSNAPNONRef,  WhereinAllFitDay1] = intersect(SNAP_NONRef_FitDays, AllFitDays);
                [~, WhereinStationDay, WhereinAllFitDay2] = intersect(StationDay, AllFitDays);    % Where the Station Data is
                
                [StationANDSNAP,  ~,  ib] = intersect(WhereinAllFitDay1, WhereinAllFitDay2);
                [~, Instation, InSnap] = intersect(StationDay, SNAP_NONRef_FitDays);
                SnapTemp = nan(length(AllFitDays), 1);
                SnapTemp(WhereinAllFitDay1) = SNAP_Ref_Fit(i, WhereinSNAPNONRef); 
                SnapTemp = SnapTemp-StationDownScaleParams(3)*(StationDownScaleParams(4)-StationZ); % Compare the bias to a topo downscaled!
                TheseStationTmax = TheseStationTmax - StationDownScaleParams(3) * (StationZ-StationDownScaleParams(4)); % Compare the bias to a topo downscaled!
                StationBiasTrain = [StationBiasTrain; (TheseStationTmax(Instation)-SnapTemp(StationANDSNAP))];
                StationDatesTrain = [StationDatesTrain; StationDay(Instation)];
                StationElevTrain = [StationElevTrain; TheseStationElev(Instation)];

            end
            [~, TrainStationDateIndex] = ismember(StationDatesTrain, SNAP_Rect.Days);
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
            [ia,TestStationDateIndex] = ismember(StationDatesTest, SNAP_Rect.Days);
            % GTtest_y = StationBiasTest;
            % train_x and train_y come from the file load
            test_x_new = [TestStationDateIndex StationElevTest];
            return 
        end

        function result = RunTest(AllStationData, SNAP_Rect, FitSet, TestSet, StationDownScaleParams)
            SNAPMonth = cell(12,3);
            StationMonth = cell(12,3);
            SNAP_DeBias_Ref = zeros(size(SNAP_Ref_Test));
            SNAP_FutureDeBias_Ref = zeros(size(SNAP_Ref_FutureTest));
            for i = 1:12
                numdays_thismonth = eomday(2023, i);  % Choose a leap year, I think thats a NAN anyway
                for j = 1:numdays_thismonth
                    WhereinSNAP = month(SNAP_NONRef_TestDay) == i & day(SNAP_NONRef_TestDay) == j; 
                    SNAPMonth{i, 1} = [SNAPMonth{i}; reshape(SNAP_Ref_Test2(:, WhereinSNAP), [], 1)];
                    WhereinStation = month(Station_Ref_TestDays) == i & day(Station_Ref_TestDays) == j;
                    StationMonth{i, 1} = [StationMonth{i}; reshape(Station_Ref_Test(:, WhereinStation), [], 1)];
                end
                [StationMonth{i, 2}, StationMonth{i, 3}] = ecdf(StationMonth{i, 1});
                [SNAPMonth{i, 2},  SNAPMonth{i, 3}] = ecdf(SNAPMonth{i, 1});
                
                % Now that we have the ECDF functions, go through and apply them to the
                % SNAP Ref data
                for j = 1:numdays_thismonth
                    WhereinSNAPREF = month(SNAP_NONRef_TestDay) == i & day(SNAP_NONRef_TestDay) == j; 
                    WhereinSNAPFuture = month(SNAP_NONRef_FutureTestDay) == i & day(SNAP_NONRef_FutureTestDay) == j; 
                    f1 = @(x) interp1(SNAPMonth{i, 3}(2:end), SNAPMonth{i, 2}(2:end), x, "nearest",'extrap'); % x,f
                    f2 = @(x) interp1(StationMonth{i, 2}(2:end), StationMonth{i, 3}(2:end), x, 'nearest', 'extrap'); % f,x
                    SNAP_DeBias_Ref(:, WhereinSNAPREF) = arrayfun(f1, SNAP_Ref_Test2(:, WhereinSNAPREF));
                    SNAP_DeBias_Ref(:, WhereinSNAPREF) = arrayfun(f2, SNAP_DeBias_Ref(:, WhereinSNAPREF));
                
                    SNAP_FutureDeBias_Ref(:, WhereinSNAPFuture) = arrayfun(f1, SNAP_Ref_FutureTest(:, WhereinSNAPFuture));
                    SNAP_FutureDeBias_Ref(:, WhereinSNAPFuture) = arrayfun(f2, SNAP_FutureDeBias_Ref(:, WhereinSNAPFuture));
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

            return
        end


        %%%%%%%%%%%%%%%%%%%
        % TODO: Unneeded? %
        %%%%%%%%%%%%%%%%%%%
        function result = PythonInterpolation()
            py.importlib.import_module('torch');

            % These two python functions are set up to perform interpolation, one is
            % gridded, one is scattered
            result = py.importlib.import_module('src.GSML_GPGRID'); % If all the training data is on a grid
            py.importlib.reload(mod);

            % result = py.importlib.import_module('GSML_GPBias_TrainPredict');  % Uses kernel interpolation on a grid
            % py.importlib.reload(mod2);
        end

        function result = TODO()
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

            [~,TestStationDateIndex] = ismember(TestDays,SNAP_Rect.Days); % This indexing needs to be constant
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
        end
    end
end