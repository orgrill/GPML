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
            self.Skip_Transform = Skip_Transform
        end

        function result = CalculateBias(self) 
            addpath(fullfile("src", "TemperatureBiasCalculator","modules"));

            disp("---- Transforming Coordinates ----")
            [Xdown, Ydown, SNAP_Rect, SNAP_Rect_Rot] = CoordinateTransform(self.CWS_Projection.ProjectedCRS, self.SNAP_Data, self.Skip_Transform);

            disp("---- Applying Topographical Downscaling ----")
            [FitDays, FitSet, TestSet, StationDownScaleParams] = TopographicalDownscaling(self.AllStationData, SNAP_Rect, SNAP_Rect_Rot, Xdown, Ydown);

            disp("---- Drawing the rest of the owl ----")
            x = self.Owl(self.AllStationData, SNAP_Rect, FitDays, FitSet, TestSet, StationDownScaleParams);

            %%%%%%%%%%%%%%%%%%%%%%%%%%
            %%%%%%%%%%%%%%%%%%%%%%%%%%
            SNAP_NONRef_Fit_File = load(fullfile("data", "SNAP_NONRef_Fit.mat"));
            SNAP_NONRef_FitDay = SNAP_NONRef_Fit_File.SNAP_NONRef_FitDay
            SNAP_NONRef_Fit = SNAP_NONRef_Fit_File.SNAP_NONRef_Fit

            % Get rid of the leap days
            EliminateLeap = find(~(month(SNAP_NONRef_FitDay)==2 & day(SNAP_NONRef_FitDay)==29));
            SNAP_NONRef_FitDay = SNAP_NONRef_FitDay(EliminateLeap);
            SNAP_NONRef_Fit = SNAP_NONRef_Fit(:, EliminateLeap);
            SNAP_Ref_Fit = load(fullfile("data", "SNAP_Ref_Fit.mat")).SNAP_Ref_Fit(:, EliminateLeap);

            [train_x, train_y, train_x_new] = BuildTrainSet(self.AllStationData, FitSet, SNAP_Ref_Fit, SNAP_NONRef_FitDay, SNAP_Rect);

            % TODO: Take rest of code from GSML_ADSAILV2 

            return % result
        end

        function SNAP_NONRef_FitDay = Owl(self, AllStationData, CWS_SNAP_Rect, FitDays, FitSet, TestSet, StationDownScaleParams)
            %% Switch to the Test Set
            station_Lengths = cellfun(@length, AllStationData(TestSet, 3), 'UniformOutput', false);
            TheseDays = CWS_SNAP_Rect.Days(CWS_SNAP_Rect.Days <= datetime(2019, 12, 31));

            % We want this at all the days up to 2020, this will be used to fit an ecdf
            SNAP_NONRef_TestDay = CWS_SNAP_Rect.Days(find(ismember(CWS_SNAP_Rect.Days, TheseDays)));
            EliminateLeap = find(~(month(SNAP_NONRef_TestDay) == 2 & day(SNAP_NONRef_TestDay) == 29));
            SNAP_NONRef_TestDay = SNAP_NONRef_TestDay(EliminateLeap);

            % This is the reference WRF, interpolated to the fine grid
            SNAP_Ref_Test2 = load(fullfile("data","SNAP_Ref_Test2.mat")).SNAP_Ref_Test2(:, EliminateLeap); 
            % TODO: This is unused?
            %SNAP_NONRef_Test = load(fullfile("data", "SNAP_NONRef_Test.mat")).SNAP_NONRef_Test(:, EliminateLeap);

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

            %% Interpolate the SNAP Temps NOT at reference elevation to the fit locations
            % These will be used to find the raw bias, which will be fed to the
            % Gaussian Process

            %% Downsample the SNAP Temps at reference elevation to the Test Stations
            % This is still slow, and is dependent on the Test/Fit split
            SNAP_Ref_Stat_File = load(fullfile("data", "SNAP_Ref_Stat.mat")) 
            FitSet = SNAP_Ref_Stat_File.FitSet
            SNAP_Ref_Test = SNAP_Ref_Stat_File.SNAP_Ref_Test
            SNAPDays_inTest = SNAP_Ref_Stat_File.SNAPDays_inTest
            TestDays = SNAP_Ref_Stat_File.TestDays
            TestSet = SNAP_Ref_Stat_File.TestSet

            SNAP_Ref_TestDays = CWS_SNAP_Rect.Days(find(ismember(CWS_SNAP_Rect.Days, TestDays))); 
            EliminateLeap = find(~(month(SNAP_Ref_TestDays) == 2 & day(SNAP_Ref_TestDays) == 29));
            SNAP_Ref_TestDays = SNAP_Ref_TestDays(EliminateLeap);
            SNAP_Ref_Test = SNAP_Ref_Test(:, EliminateLeap);
                    
            %% Downsample Station data at reference elevation to test station locations
            Station_Ref_Test = load(fullfile("data","Station_Ref_Test.mat")).Station_Ref_Test;
            Station_Ref_TestDays = FitDays(find(ismember(FitDays, TestDays)));

            %% Interpolate Non Ref Training Station Data to test station locations
            % We want this at all the days up to 2020, this will be used to fit an ecdf

            % TODO: This is unused?
            % Station_NONRef_Test = load(fullfile("data", "Station_NONRef_Test.mat"), "Station_NONRef_Test");
            Station_NONRef_TestDays = FitDays(find(ismember(TestDays, TheseDays)));

            %{
            Find the ECDF functions
                This is done using the non-reference data we have interpolated to the
                test stations, we want two ecdf functions for each month, one for the
                SNAP data and one for the Station data.
            %}
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
    end
end