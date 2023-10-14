classdef GaussianLearningMethod
    methods
        function rmse = Run(self, AllStationData, SNAP_Rect, TestSet, FitSet, StationDownScaleParams, SNAP_NONRef_TestDays, ... 
                            SNAP_NONRef_FutureTest, WhichFitStation, StationDatesTest, StationElevTest)
            % Get SNAP_Ref_Test
            SNAP_Ref_Test = load(fullfile("data", "SNAP_Ref_Test2.mat")).SNAP_Ref_Test2(:, EliminateLeap); 

            %{
                Interpolate the SNAP Temps NOT at reference elevation to the fit locations
                These will be used to find the raw bias, which will be fed to the Gaussian Process
            %}
            selected_days = SNAP_Rect.Days(SNAP_Rect.Days <= datetime(2022, 3, 22));
            selected_days_index = find(ismember(SNAP_Rect.Days, selected_days));
            SNAP_NONRef_FitDays = SNAP_Rect.Days(selected_days_index);

            %% This builds everything needed to run a GP for TRAINING 
            % Build the train set here, have to do all training then use trained model to run on TEST data
            % TEST data obtained in next step
            StationBiasTrain = [];
            StationDatesTrain = [];
            StationElevTrain = [];

            for i = 1:size(AllStationData(FitSet),2)
                TheseStationTmax = [AllStationData{FitSet(i), 4}];
                
                StationDay =       [AllStationData{FitSet(i), 3}]; % Days that we have station data
                %StationDay = StationDay(year(StationDay) >= 2000); 
                AllFitDays =  union(StationDay, SNAP_NONRef_FitDays); % All the days we want to train on (10950:end)
                TheseStationElev = [AllStationData{FitSet(i),2}];
                StationZ = mean(AllStationData{FitSet(i), 2}).*.3048;
                [~,WhereinSNAPNONRef, WhereinAllFitDay1] = intersect(SNAP_NONRef_FitDays, AllFitDays);
                [~,WhereinStationDay, WhereinAllFitDay2] = intersect(StationDay,AllFitDays);    % Where the Station Data is
                
                [StationANDSNAP, ~, ib] = intersect(WhereinAllFitDay1, WhereinAllFitDay2);
                [~, Instation, InSnap] = intersect(StationDay, SNAP_NONRef_FitDays);
                SnapTemp = nan(length(AllFitDays), 1);

                % TODO: Where does this data come from?
                SNAP_Ref_Fit = load(fullfile("data", "SNAP_Ref_Fit.mat")).SNAP_Ref_Fit; 
                SnapTemp(WhereinAllFitDay1) = SNAP_Ref_Fit(i, WhereinSNAPNONRef); 
                SnapTemp = SnapTemp - StationDownScaleParams(3) * (StationDownScaleParams(4) - StationZ); % Compare the bias to a topo downscaled!
                TheseStationTmax=TheseStationTmax - StationDownScaleParams(3) * (StationZ - StationDownScaleParams(4)); % Compare the bias to a topo downscaled!
                StationBiasTrain = [StationBiasTrain; (TheseStationTmax(Instation)-SnapTemp(StationANDSNAP))];
                StationDatesTrain = [StationDatesTrain; StationDay(Instation)];
                StationElevTrain = [StationElevTrain; TheseStationElev(Instation)];

            end
            [~,TrainStationDateIndex] = ismember(StationDatesTrain, SNAP_Rect.Days);
            train_x = [TrainStationDateIndex StationElevTrain];
            train_y = StationBiasTrain;  

            % Set up Test set
            [ia,TestStationDateIndex] = ismember(StationDatesTest, SNAP_Rect.Days);
            GPBiasTestDays = StationDatesTest(ia);
            test_x_new = [TestStationDateIndex StationElevTest];

            % The GP will run on a TRAIN and TEST set
            % Perform Prediction on this new test set, using the existing trained model
            TrainFlag = 1;
            SubFitTest = [];
            % Build and save the data for the GpyTorch inference
            [~, SubFitTest] = FutureBias(train_x, train_y, test_x_new, SubFitTest, TrainFlag);

            disp("Calling GSML_GPBias_Remote.py")
            test_x = []; % matlab insists on variables being defined before they are defined
            train_x, train_y, test_x, pred_labels = pyrunfile( ...
                "GSML_GPBias_Remote.py", ...
                ["train_x", "train_y", "test_x", "pred_labels"], ...
                train_x = train_x, train_y = train_y, test_x = test_x_new, TrainFlag = TrainFlag);

            train_x = double(train_x);
            train_y = double(train_y);
            test_x = double(test_x);
            pred_labels = double(pred_labels);

            % RMSE code 
            GP_DeBias = cell(length(TestSet),1);
            GP_DeBiasDays = cell(length(TestSet),1);
            WhereinStationDay = cell(length(TestSet),1);
            GT_Bias = cell(length(TestSet),1);
            GPDeBiasError = cell(length(TestSet),1);

            for i=1:length(TestSet)
                TheseStationTmax = [AllStationData{TestSet(i), 4}];
                StationDay = [AllStationData{TestSet(i), 3}];
                ThisGPBias = pred_labels(WhichFitStation==i);
                [StationANDSNAP, ~, ~] = intersect(WhereinAllTestDay1{i}, WhereinAllTestDay3{i});
                [~, Instation, ~] = intersect(StationDay, SNAP_NONRef_TestDays);
                %References ref & non-ref data to make sure days align, but SnapTemp only used in GP RMSE calcs
                StationZ = mean(AllStationData{FitSet(i), 2}).*.3048; % converts elevation to meters
                SnapTemp = nan(length(ThisGPBias), 1);
                SnapTemp(WhereinAllTestDay1{i}) = SNAP_Ref_Test(i, WhereinSNAPNONRef{i}); 
                SnapTemp(WhereinAllTestDay2{i}) = SNAP_NONRef_FutureTest(i, WhereinSNAPFuture{i});
                SnapTemp = SnapTemp-StationDownScaleParams(3) * (StationDownScaleParams(4) - StationZ);
                GP_DeBias{i} = (SnapTemp+ThisGPBias') - StationDownScaleParams(3) * (StationZ-StationDownScaleParams(4));
                
                GP_DeBiasDays{i} = GPBiasTestDays(WhichFitStation==i);
                %(TheseStationTmax(WhereinStationTest{i})'-SNAP_NONRef_Test(i,WhereinSNAPTest{i}))'
                if ~isempty(StationANDSNAP)
                    GPDeBiasError{i} = TheseStationTmax(Instation)-GP_DeBias{i}(StationANDSNAP);
                    GT_Bias{i} = TheseStationTmax(Instation) - SnapTemp(StationANDSNAP);
                end
                
            end

            ALLGTBias = cell2mat(GT_Bias);
            RawWRF_RMSE = sqrt(sum(ALLGTBias.^2) / length(ALLGTBias));
            AllDeBiasError = cell2mat(GPDeBiasError);
            AllDeBiasError = AllDeBiasError(~isnan(AllDeBiasError));
            GPDeBiasRMSE = sqrt(sum(AllDeBiasError.^2) / length(AllDeBiasError));

            rmse = RawWRF_RMSE
            return 
        end
    end
end