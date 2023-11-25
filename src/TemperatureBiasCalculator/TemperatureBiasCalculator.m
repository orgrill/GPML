classdef TemperatureBiasCalculator
    properties (SetAccess = immutable)
        % Input Data
        AllStationData  
        CWS_Projection
        SNAP_Data

        % Options
        Skip_Transform
    end

    properties (Access = private)
        % Cleaned + Transformed weather station data
        SNAP_Rect
        SNAP_Rect_Rotated
        SNAP_Downscaled

        StationDownScaleParams

        FitSet  % Station data selection used for Gauss (train on fit set)
        FitSet_Days  % Datetimes that are present in the FitSet

        TestSet  % Station data selection used for Gauss (predict on test set)
        TestSet_Days % Datetimes that are present in the TestSet

        SNAP_Ref_FutureTest 
        SNAP_Ref_FutureTestDays
        SNAP_NONRef_FutureTest 
        SNAP_NONRef_FutureTestDays
        SNAP_NONRef_Test
        SNAP_NONRef_TestDays

        WhereinSNAPNONRef
        WhereinStationDay
        WhereinAllTestDay1 
        WhereinAllTestDay2 
        WhereinAllTestDay3 
        WhereinSNAPFuture 

        StationDatesTest
        StationElevTest
        WhichFitStation
    end

    methods
        function self = TemperatureBiasCalculator(AllStationData, CWS_Projection, SNAP_Data, Skip_Transform)
            addpath(fullfile("src", "TemperatureBiasCalculator"));
            addpath(fullfile("src", "TemperatureBiasCalculator", "modules"));

            self.AllStationData = AllStationData;
            self.CWS_Projection = CWS_Projection;
            self.SNAP_Data = SNAP_Data;
            self.Skip_Transform = Skip_Transform;

            disp("--------- BiasCalculator Setup: Transforming Coordinates")
            [self.SNAP_Rect, self.SNAP_Rect_Rotated] = CoordinateTransform(self.CWS_Projection.ProjectedCRS, self.SNAP_Data, self.Skip_Transform);

            disp("--------- BiasCalculator Setup: Selecting Test and Fit sets")
            %{
                Split into test and fit sets, set up and execute EQM and topgraphical downscaling
                Load the raw station data, we need to divide this into two parts, 80/20
            %}
            self.FitSet = sort(randperm(size(self.AllStationData, 1), round(.8 * size(self.AllStationData, 1))));
            self.FitSet_Days = AllStationData(self.FitSet, 3);
            self.FitSet_Days = unique(vertcat(self.FitSet_Days{:, 1}));

            self.TestSet = sort(setdiff(1:length(self.AllStationData), self.FitSet));
            self.TestSet_Days = AllStationData(self.TestSet, 3);
            self.TestSet_Days = unique(vertcat(self.TestSet_Days{:, 1}));

            disp("--------- BiasCalculator Setup: Applying Topographical Downscaling")
            % Fit the Topographical Downscaling parameters (White 2016)
            self.StationDownScaleParams = GSML_Topo_Downscale(self.AllStationData, self.FitSet); %[T0; Beta; Gamma; zref]
            self.SNAP_Downscaled = TopographicalDownscaling(self.AllStationData, self.SNAP_Rect, self.SNAP_Rect_Rotated, self.StationDownScaleParams);

            % Perform the rest of the setup
            self = self.DataSetup();

            % TODO:
            %SubRegion = kmz2struct(fullfile("data", "CopperRiverWatershed.kmz"));

            % TODO: What is this
            % load('xyStationFit.mat');
        end

        function rmse = CalculateBias(self, methodType) 
            addpath(fullfile("src", "TemperatureBiasCalculator", "LearningMethods"));

            switch methodType
                case "EQM"
                    disp("--------- BiasCalculator: Calculating RMSE using EQM")
                    learningMethod = EQMLearningMethod();
                    rmse = learningMethod.Run( ...
                        self.AllStationData, self.CWS_Projection, self.SNAP_Rect, self.SNAP_Rect_Rotated, ...
                        self.FitSet, self.FitSet_Days, self.TestSet, self.TestSet_Days, self.StationDownScaleParams, ...
                        self.SNAP_Ref_FutureTest, self.SNAP_Ref_FutureTestDays, self.WhereinAllTestDay1, self.WhereinAllTestDay3);

                case "Gaussian"
                    disp("--------- BiasCalculator: Calculating RMSE using Gaussian Process")
                    learningMethod = GaussianLearningMethod();
                    rmse = learningMethod.Run( ...
                        self.AllStationData, self.SNAP_Rect, self.TestSet, self.FitSet, self.StationDownScaleParams, ...
                        self.SNAP_NONRef_Test, self.SNAP_NONRef_FutureTestDays, self.WhichFitStation, self.StationDatesTest, self.StationElevTest);
            end
            return 
        end
    end

    methods (Access = private)
        % Set up Additional info that is used by both LearningMethods
        function self = DataSetup(self)
            % This is raw WRF data interpolated to stations, into the future
            % TODO: Where does this data come from? Can we compute it manually instead of loading from file?
            NONRef_FutureTest = load(fullfile("data", "SNAP_NONRef_FutureTest.mat")); 
            EliminateLeap = find(~(month(NONRef_FutureTest.SNAP_NONRef_TestDay) == 2 & day(NONRef_FutureTest.SNAP_NONRef_TestDay) == 29));
            self.SNAP_NONRef_FutureTestDays = NONRef_FutureTest.SNAP_NONRef_TestDay(EliminateLeap);
            self.SNAP_NONRef_FutureTest = NONRef_FutureTest.SNAP_NONRef_Test(:, EliminateLeap);

            %% We want the SNAP Data in the future at the reference elevation so we can apply the EQM to it,  
            self.SNAP_Ref_FutureTest = zeros(size(self.SNAP_NONRef_FutureTest));
            for i = 1:length(self.TestSet)
                StationZ = mean(self.AllStationData{self.TestSet(i), 2}).* .3048;
                self.SNAP_Ref_FutureTest(i, :) = self.SNAP_NONRef_FutureTest(i, :) - self.StationDownScaleParams(3) * (self.StationDownScaleParams(4) - StationZ);
            end
            self.SNAP_Ref_FutureTestDays = self.TestSet_Days;

            % Get SNAP NONref Test Days
            selected_days = self.SNAP_Rect.Days(self.SNAP_Rect.Days <= datetime(2019, 12, 31));
            % We want this at all the days up to 2020, this will be used to fit an ecdf
            self.SNAP_NONRef_TestDays = self.SNAP_Rect.Days(find(ismember(self.SNAP_Rect.Days, selected_days)));
            EliminateLeap = find(~(month(self.SNAP_NONRef_TestDays) == 2 & day(self.SNAP_NONRef_TestDays) == 29));
            self.SNAP_NONRef_TestDays = self.SNAP_NONRef_TestDays(EliminateLeap);

            %% Select Test Set
            cell_allocated = cell(length(self.TestSet), 1);
            self.WhereinSNAPNONRef = cell_allocated;
            self.WhereinStationDay = cell_allocated;
            self.WhereinAllTestDay1 = cell_allocated;
            self.WhereinAllTestDay2 = cell_allocated;
            self.WhereinAllTestDay3 = cell_allocated;
            self.WhereinSNAPFuture = cell_allocated;

            self.StationDatesTest = [];
            self.StationElevTest = [];
            self.WhichFitStation = [];
            for i = 1:size(self.AllStationData(self.TestSet), 2)
                StationDay = [self.AllStationData{self.TestSet(i), 3}]; % Days that we have station data
                %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                % Here is where we choose the days to evaluate on, they are all at the
                % test locations
                %StationDay = StationDay(year(StationDay) >= 2000); 
                AllTestDays = union(StationDay, self.SNAP_NONRef_FutureTestDays); % All the days we want to test
                %AllTestDays = union(StationDay,SNAP_NONRef_FutureTestDays([1 end]));
                %AllTestDays = SNAP_NONRef_FutureTestDays;
                %AllTestDays = [StationDay];
                %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                [~,self.WhereinSNAPNONRef{i}, self.WhereinAllTestDay1{i}] = intersect(self.SNAP_NONRef_TestDays, AllTestDays); % Where the old SNAP_NONRef is
                [~,self.WhereinSNAPFuture{i}, self.WhereinAllTestDay2{i}] = intersect(self.SNAP_NONRef_FutureTestDays, AllTestDays); % Where the future SNAP is
                [~,self.WhereinStationDay{i}, self.WhereinAllTestDay3{i}] = intersect(StationDay, AllTestDays);    % Where the Station Data is

                ThisStationElev = mean([self.AllStationData{self.TestSet(i), 2}]); % Elevation of station i
                self.WhichFitStation = [self.WhichFitStation; i.*ones(length(AllTestDays), 1)]; 
                self.StationDatesTest = [self.StationDatesTest; AllTestDays];
                self.StationElevTest = [self.StationElevTest; ThisStationElev.*ones(length(AllTestDays), 1)];
            end
        end

    end

    methods (Static)
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