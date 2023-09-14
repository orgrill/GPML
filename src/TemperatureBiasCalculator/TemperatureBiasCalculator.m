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

        FitSet  % Station data selection used for EQM
        TestSet  % Station data selection used for Gauss

        StationDownScaleParams
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

            %{
                Split into test and fit sets, set up and execute EQM and topgraphical downscaling
                Load the raw station data, we need to divide this into two parts, 80/20
            %}
            disp("--------- BiasCalculator Setup: Selecting Test and Fit sets")
            self.FitSet = sort(randperm(size(self.AllStationData, 1), round(.8 * size(self.AllStationData, 1))));
            self.TestSet = sort(setdiff(1:length(self.AllStationData), self.FitSet));

            disp("--------- BiasCalculator Setup: Applying Topographical Downscaling")
            % Fit the Topographical Downscaling parameters (White 2016)
            self.StationDownScaleParams = GSML_Topo_Downscale(self.AllStationData, self.FitSet); %[T0; Beta; Gamma; zref]
            self.SNAP_Downscaled = TopographicalDownscaling(self.AllStationData, self.SNAP_Rect, self.SNAP_Rect_Rotated, self.StationDownScaleParams);

        end

        function rmse = CalculateBias(self, methodType) 
            addpath(fullfile("src", "TemperatureBiasCalculator", "LearningMethods"));
            % TODO: Unused?
            %SubRegion = kmz2struct(fullfile("data", "CopperRiverWatershed.kmz"));

            % TODO: What is this
            % load('xyStationFit.mat');

            switch methodType
                case "EQM"
                    disp("--------- BiasCalculator: Calculating RMSE using EQM")
                    learningMethod = EQMLearningMethod();

                    [FitDays, SNAP_Ref_Fit, SNAP_Ref_TestDays, SNAP_Ref_Test] = ...
                        learningMethod.GetConfig(self.AllStationData, self.SNAP_Rect, self.SNAP_Rect_Rotated, self.FitSet);

                    [train_x, train_y] = learningMethod.BuildTrainingSet(self.AllStationData, self.FitSet, SNAP_Ref_Fit, self.SNAP_Rect);

                    % TODO: Pass necessary parameters here
                    rmse = learningMethod.Run();

                case "Gaussian"
                    disp("--------- BiasCalculator: Calculating RMSE using Gaussian Process")
                    learningMethod = GaussianLearningMethod();
                    learningMethod.Run(self.AllStationData, self.SNAP_Rect, self.TestSet, self.FitSet, self.StationDownScaleParams);
            end

            %% Downsample Station data at reference elevation to test station locations
            % TODO: Where does this data come from? Can we compute it manually instead of loading from file?
            % TODO: Ref and NonRef both rely on the other methods parameters?
            %% Switch to the Test Set

            % TODO: Is this code useful for anything?
            station_lengths = cellfun(@length, self.AllStationData(self.TestSet, 3), 'UniformOutput', false);
            TestDays = [];
            for i = 1:size(station_lengths, 1)
                TestDays = [TestDays; [self.AllStationData{self.TestSet(i), 3}]];
            end
            TestDays = unique(TestDays); % These are the days present in the test stations

            %Yes need, used in run section of EQMLearningMethod.m code
            Station_Ref_Test = load(fullfile("data", "Station_Ref_Test.mat")).Station_Ref_Test;    
            %only used in plotting- will need eventually
            Station_NONRef_Test = load(fullfile("data", "Station_NONRef_Test.mat")).Station_NONRef_Test;
            %only used in plotting- will need eventually
            %Station_NONRef_TestDays = FitDays(find(ismember(TestDays, FitDays)));
            %Yes need, used in run of EQMLearningMethod.m
            % Station_Ref_TestDays = FitDays(find(ismember(FitDays, TestDays)));
            return 
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