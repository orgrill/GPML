classdef EQMLearningMethod < LearningMethod
    methods
        function [FitDays, SNAP_Ref_Fit, SNAP_Ref_TestDays, SNAP_Ref_Test] = GetConfig(self, AllStationData, SNAP_Rect, SNAP_Rect_Rotated, FitSet)
            FitDays = [];
            station_lengths = cellfun(@length, AllStationData(FitSet, 3), 'UniformOutput', false);
            for i = 1:size(station_lengths, 1)
                FitDays = [FitDays; [AllStationData{FitSet(i), 3}]];
            end
            FitDays = unique(FitDays);  % These are the days present in the fit stations

            selected_days = SNAP_Rect.Days(SNAP_Rect.Days <= datetime(2022, 3, 22));
            selected_days_index = find(ismember(SNAP_Rect.Days, selected_days));

            [StationLL, iA, iC] = unique(AllStationData(:, [2, 3]), 'rows');
            [StationX, StationY] = projfwd(CWS_Projection.ProjectedCRS, StationLL(:, 1), StationLL(:, 2));
            test_x = [StationX(FitSet) StationY(FitSet)];
            SNAP_Ref_Fit = InterpolateSNAP(SNAP_Rect_Rotated, SNAP_Rect.t2ref, test_x, selected_days_index);

            SNAP_Ref_TestDays = SNAP_Rect.Days(find(ismember(SNAP_Rect.Days, TestDays))); 
            EliminateLeap = find(~(month(SNAP_Ref_TestDays) == 2 & day(SNAP_Ref_TestDays) == 29));
            SNAP_Ref_TestDays = SNAP_Ref_TestDays(EliminateLeap);

            %TODO: How is this calculated? Can't find it in original (Other than loading from file)
            % This is the reference WRF, interpolated to the fine grid
            % TODO: Where does this data come from? Can we compute it manually instead of loading from file?
            SNAP_Ref_Test = load(fullfile("data", "SNAP_Ref_Test2.mat")).SNAP_Ref_Test2(:, EliminateLeap); 
        end


        function [train_x, train_y] = BuildTrainingSet(AllStationData, FitSet, SNAP_Ref_Fit, SNAP_Rect)
            %TODO: find inputs needed to run the EQM ?? Not train/test split like regression
        end

        function rmse = Run(self, AllStationData, SNAP_Rect, FitSet, SNAP_Ref_Test, SNAP_Ref_FutureTest, StationDownScaleParams)
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

            %%%%%%%%%% Everything in run prior to this is the calculations for EQM method %%%%%%%%%%%%

            % Now Move the DeBias_Ref Temps back to actual elevation using the known
            % elevation of each test station
            SNAP_DeBias_NonRef = zeros(size(SNAP_DeBias_Ref));
            SNAP_FutureDeBias_NonRef = zeros(size(SNAP_FutureDeBias_Ref));
            for i=1:length(TestSet)
                StationZ = mean(AllStationData{TestSet(i),2}).*.3048;
                SNAP_DeBias_NonRef(i,:) = SNAP_DeBias_Ref(i,:)-StationDownScaleParams(3)*(StationZ-StationDownScaleParams(4));
                SNAP_FutureDeBias_NonRef(i,:) = SNAP_FutureDeBias_Ref(i,:)-StationDownScaleParams(3)*(StationZ-StationDownScaleParams(4));
            end

            % RMSE code - TODO: Double-check that everything needed for EQM is here
            %                  - Will need to add WhereinAllTestDay1 & 3, best way to do so?
        
            EQMBias = cell(length(TestSet),1);

            for i=1:length(TestSet)
                TheseStationTmax = [AllStationData{TestSet(i), 4}];
                StationDay =       [AllStationData{TestSet(i),3}];
                [StationANDSNAP, ~, ~] = intersect(WhereinAllTestDay1{i},WhereinAllTestDay3{i});
                [~,Instation2,InSnapDeBias] = intersect(StationDay,SNAP_Ref_TestDays);
                if ~isempty(StationANDSNAP)
                    EQMBias{i} = TheseStationTmax(Instation2)-SNAP_DeBias_NonRef(i,InSnapDeBias)';
                end
                
            end

            EQM_All = cell2mat(EQMBias);
            EQM_RMSE = sqrt(sum(EQM_All.^2)/length(EQM_All));
        end
    end
end
