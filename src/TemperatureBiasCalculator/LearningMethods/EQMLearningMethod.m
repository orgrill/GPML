classdef EQMLearningMethod
    methods
        function EQM_RMSE = Run(self, AllStationData, CWS_Projection, SNAP_Rect, SNAP_Rect_Rotated, FitSet, ...
                                FitSet_Days, TestSet, TestSet_Days, StationDownScaleParams, SNAP_Ref_FutureTest, ... 
                                SNAP_Ref_FutureTestDays, WhereinAllTestDay1, WhereinAllTestDay3)

            selected_days = SNAP_Rect.Days(SNAP_Rect.Days <= datetime(2019, 12, 31));
            selected_days_index = find(ismember(SNAP_Rect.Days, selected_days));

            [StationX, StationY] = projfwd(CWS_Projection.ProjectedCRS, cell2mat(AllStationData(:, 2)), cell2mat(AllStationData(:, 4)));
            test_x = [StationX(FitSet) StationY(FitSet)];

            % Get Station_Ref_Test
            Station_Ref_Test = load(fullfile("data", "Station_Ref_Test.mat")).Station_Ref_Test;
            shared_days = find(ismember(FitSet_Days, TestSet_Days));
            Station_Ref_TestDays = FitSet_Days(shared_days);

            % Get SNAP_Ref_Test
            shared_days = find(ismember(SNAP_Rect.Days, TestSet_Days));
            SNAP_Ref_TestDays = SNAP_Rect.Days(shared_days); 
            EliminateLeap = find(~(month(SNAP_Ref_TestDays) == 2 & day(SNAP_Ref_TestDays) == 29));
            SNAP_Ref_TestDays = SNAP_Ref_TestDays(EliminateLeap); 
            SNAP_Ref_Test = load(fullfile("data", "SNAP_Ref_Stat.mat")).SNAP_Ref_Test(:, EliminateLeap);

            % Get SNAP_NONRef_Test
            SNAP_NONRef_TestDays = SNAP_Rect.Days(find(ismember(SNAP_Rect.Days, selected_days)));
            SNAP_NONRef_TestDays = SNAP_NONRef_TestDays(EliminateLeap);
            SNAP_NONRef_Test = load(fullfile("data", "SNAP_NONRef_Test.mat")).SNAP_NONRef_Test(:, EliminateLeap);
            

            %% Find the ECDF functions
            % This is done using the non-reference data we have interpolated to the
            % test stations, we want two ecdf functions for each month, one for the
            % SNAP data and one for the Station data.
            
            SNAPMonth = cell(12, 3);
            StationMonth = cell(12, 3);
            SNAP_DeBias_Ref = zeros(size(SNAP_Ref_Test));
            SNAP_FutureDeBias_Ref = zeros(size(SNAP_Ref_FutureTest));

            for i=1:12
                numdays_thismonth = eomday(2023,i);  % Choose a leap year, I think thats a NAN anyway
                for j=1:numdays_thismonth
                    WhereinSNAP = month(SNAP_NONRef_TestDays) == i & day(SNAP_NONRef_TestDays) == j; 
                    SNAPMonth{i, 1} = [SNAPMonth{i}; reshape(SNAP_Ref_Test(:, WhereinSNAP), [], 1)];
                    WhereinStation = month(Station_Ref_TestDays) == i & day(Station_Ref_TestDays) == j;
                    StationMonth{i, 1} = [StationMonth{i}; reshape(Station_Ref_Test(:, WhereinStation), [], 1)];
                end
                [StationMonth{i,2}, StationMonth{i,3}] = ecdf(StationMonth{i, 1});
                [SNAPMonth{i, 2}, SNAPMonth{i, 3}] = ecdf(SNAPMonth{i, 1});
                
                % Now that we have the ECDF functions, go through and apply them to the
                % SNAP Ref data
                for j=1:numdays_thismonth
                    WhereinSNAPREF = month(SNAP_NONRef_TestDays) == i & day(SNAP_NONRef_TestDays) == j; 
                    WhereinSNAPFuture = month(SNAP_Ref_FutureTestDays) == i & day(SNAP_Ref_FutureTestDays) == j; 
                    f1 = @(x) interp1(SNAPMonth{i, 3}(2:end), SNAPMonth{i, 2}(2:end), x,"nearest", 'extrap'); % x,f
                    f2 = @(x) interp1(StationMonth{i, 2}(2:end), StationMonth{i, 3}(2:end), x, 'nearest', 'extrap'); % f,x
                    SNAP_DeBias_Ref(:, WhereinSNAPREF) = arrayfun(f1, SNAP_Ref_Test(:, WhereinSNAPREF));
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
                StationZ = mean(AllStationData{TestSet(i), 2}).*.3048;
                SNAP_DeBias_NonRef(i, :) = SNAP_DeBias_Ref(i, :) - StationDownScaleParams(3) * (StationZ-StationDownScaleParams(4));
                SNAP_FutureDeBias_NonRef(i, :) = SNAP_FutureDeBias_Ref(i, :) - StationDownScaleParams(3) * (StationZ-StationDownScaleParams(4));
            end


            EQMBias = cell(length(TestSet),1);
            for i=1:length(TestSet)
                TheseStationTmax = [AllStationData{TestSet(i), 4}];
                StationDay = [AllStationData{TestSet(i), 3}];
                [StationANDSNAP, ~, ~] = intersect(WhereinAllTestDay1{i}, WhereinAllTestDay3{i});
                [~, Instation2, InSnapDeBias] = intersect(StationDay, SNAP_NONRef_TestDays);
                StationANDSNAP
                if ~isempty(StationANDSNAP)
                    EQMBias{i} = TheseStationTmax(Instation2) - SNAP_DeBias_NonRef(i, InSnapDeBias)';
                end
                
            end

            % TODO: EQMBias contains some NaNs at this point, causing the RMSE to become NaN
            EQM_All = cell2mat(EQMBias);
            EQM_RMSE = sqrt(sum(EQM_All.^2)/length(EQM_All));
        end
    end
end