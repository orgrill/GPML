classdef TemperatureBiasCalculator
    properties
        A
        B
    end

    methods
        function result = CalculateBias(self)
            % TODO: Replace load("folder/filename") with
            % load(fullfile("folder", "filename"))
            %CWS_Projection was nested, hence the .CWS_Projection at the end
            CWS_Projection = load(fullfile("data", "CWS_Projection.mat")).CWS_Projection;
            load(fullfile("data","CWS_SNAP_Rect.mat"));

            disp("Cleaning weather station data")
            WS_Data = GSML_CleanWeatherStation(CWS_Projection);

            % Weather station data is still in Lat/Long, move it to our projection grid
            [StationLL, ~, ~] = unique([[WS_Data{3}], [WS_Data{4}] ],'rows');
            [StationX, StationY] = projfwd(CWS_Projection.ProjectedCRS,StationLL(:,1),StationLL(:,2));

            disp("Transforming Coordinates")
            [Xdown, Ydown, SNAP_Rect_Rot] = CoordinateTransform(CWS_Projection.ProjectedCRS, CWS_SNAP_Rect);

            AllStationData = load(fullfile("data", "AllStationData.mat")).AllStationData;

            disp("Applying Topographical Downscaling")
            [CWS_SNAP_Downscaled, FitDays, FitSet, TestSet] = TopographicalDownscaling(AllStationData, CWS_SNAP_Rect, SNAP_Rect_Rot, StationX, Xdown, Ydown);

            disp("Drawing the rest of the owl")
            SNAP_NONRef_FitDay = self.Owl(AllStationData, CWS_SNAP_Rect, FitDays, FitSet, TestSet, StationX, StationY);

            %[train_x, train_y, train_x_new] = BuildTrainSet(AllStationData, SNAP_NONRef_FitDay);
        end

        function SNAP_NONRef_FitDay = Owl(self, AllStationData, CWS_SNAP_Rect, FitDays, FitSet, TestSet, StationX, StationY)
            %% Switch to the Test Set
            station_Lengths = cellfun(@length, AllStationData(TestSet, 3), 'UniformOutput', false);
            TestDays = [];
            for i = 1:size(station_Lengths, 1)
                TestDays = [TestDays; [AllStationData{TestSet(i), 3}]];
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

            SNAP_NONRef_TestDay = CWS_SNAP_Rect.Days(TheseDaysIndex);
            EliminateLeap = find(~(month(SNAP_NONRef_TestDay) == 2 & day(SNAP_NONRef_TestDay) == 29));
            SNAP_NONRef_TestDay = SNAP_NONRef_TestDay(EliminateLeap);

            SNAP_Ref_Test2 = load(fullfile("data","SNAP_Ref_Test2.mat")).SNAP_Ref_Test2; % This is the reference WRF, interpolated to the fine grid
            SNAP_NONRef_Test = load(fullfile("data", "SNAP_NONRef_Test.mat")).SNAP_NONRef_Test;
            FutureTest = load(fullfile("data", "SNAP_NONRef_FutureTest.mat")); % This is raw WRF data interpolated to stations, into the future
            SNAP_NONRef_FutureTest = FutureTest.SNAP_NONRef_Test
            SNAP_NONRef_FutureTestDay = FutureTest.SNAP_NONRef_TestDay

            % Get rid of the NAN leap days
            SNAP_Ref_Test2 = SNAP_Ref_Test2(:, EliminateLeap);
            SNAP_NONRef_Test = SNAP_NONRef_Test(:, EliminateLeap);
            SNAP_NONRef_FutureTest = SNAP_NONRef_FutureTest(:, EliminateLeap);

            %% We want the SNAP Data in the future at the reference elevation so we can apply the EQM to it,  
            SNAP_Ref_FutureTest = zeros(size(SNAP_NONRef_FutureTest));
            for i=1:length(TestSet)
                StationZ = mean(AllStationData{TestSet(i), 2}).*.3048;
                SNAP_Ref_FutureTest(i,:) = SNAP_NONRef_FutureTest(i, :)-StationDownScaleParams(3)*(StationDownScaleParams(4)-StationZ);
            end

            %% Interpolate the SNAP Temps NOT at reference elevation to the fit locations
            % These will be used to find the raw bias, which will be fed to the
            % Gaussian Process
            test_x = [StationX(FitSet) StationY(FitSet)];
            % TheseDays = CWS_SNAP_Rect.Days(CWS_SNAP_Rect.Days<=datetime(2022,3,22));
            % logical_Index = ismember(CWS_SNAP_Rect.Days,TheseDays);
            % TheseDaysIndex = find(logical_Index);
            % SNAP_Ref_Fit2 = InterpolateSNAP(SNAP_Rect_Rot,CWS_SNAP_Rect.t2ref,test_x,TheseDaysIndex);
            load(fullfile("data", "SNAP_Ref_Fit.mat"))

            % SNAP_NONRef_FitDay = CWS_SNAP_Rect.Days(TheseDaysIndex);
            load(fullfile("data", "SNAP_NONRef_Fit.mat"))
            % Get rid of the leap days
            EliminateLeap = find(~(month(SNAP_NONRef_FitDay)==2 & day(SNAP_NONRef_FitDay)==29));
            SNAP_NONRef_FitDay = SNAP_NONRef_FitDay(EliminateLeap);
            SNAP_NONRef_Fit = SNAP_NONRef_Fit(:, EliminateLeap);
            SNAP_Ref_Fit = SNAP_Ref_Fit(:, EliminateLeap);
            %% Downsample the SNAP Temps at reference elevation to the Test Stations
            % This is still slow, and is dependent on the Test/Fit split
            load(fullfile("data", "SNAP_Ref_Stat.mat")) % This also loads SNAP_Ref_Test
            SNAP_Ref_TestDays = CWS_SNAP_Rect.Days(SNAPDays_inTest); 
            EliminateLeap = find(~(month(SNAP_Ref_TestDays)==2 & day(SNAP_Ref_TestDays)==29));
            SNAP_Ref_TestDays = SNAP_Ref_TestDays(EliminateLeap);
            SNAP_Ref_Test = SNAP_Ref_Test(:,EliminateLeap);
                    
            %  figure
            %  plot(CWS_SNAP_Rect.Days(SNAPDays_inTest),SNAP_Ref_Test(1,:));
            %  hold on
            %  plot(CWS_SNAP_Rect.Days(TheseDaysIndex),SNAP_NONRef_Test(1,:));
            %% Downsample Station data at reference elevation to test station locations
            load(fullfile("data","Station_Ref_Test.mat"))
            Station_Ref_TestDays = FitDays(FitStationDays_inTest);

            %% Interpolate Non Ref Training Station Data to test station locations
            %test_x = [StationX(TestSet) StationY(TestSet)];
            % We want this at all the days up to 2020, this will be used to fit an ecdf
            TheseDays = CWS_SNAP_Rect.Days(CWS_SNAP_Rect.Days<=datetime(2019,12,31));
            logical_Index = ismember(TestDays,TheseDays);
            TheseDaysIndex = find(logical_Index);
            %Station_NONRef_Test = InterpolateStation(xStationFit,yStationFit_NonRef,test_x,TheseDaysIndex);
            load(fullfile("data", "Station_NONRef_Test.mat"))
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
                    WhereinSNAPFuture = month(SNAP_NONRef_FutureTestDay)==i & day(SNAP_NONRef_FutureTestDay)==j; 
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