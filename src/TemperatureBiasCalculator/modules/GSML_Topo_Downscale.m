function result = GSML_Topo_Downscale(AllStationData, FitSet)
% GSML_Topographical_Downscaling
% Do all the linear regression fitting for a random set of weather stations from
% AllStationData
    station_Lengths = cellfun(@length, AllStationData(FitSet,3), 'UniformOutput', false);
    Tstation_LT = zeros(length(station_Lengths),1);
    stationXcoord = zeros(length(station_Lengths),1);
    stationElev = zeros(length(station_Lengths),1);
    %thisrow = 1;
    for i=1:size(station_Lengths)
        Tstation_LT(i) = mean(cell2mat(AllStationData(FitSet(i),4)));
        stationXcoord(i) = AllStationData{FitSet(i),5}(1);
        stationElev(i) = mean(cell2mat(AllStationData(FitSet(i),2)).*.3048); % Feet to meters
        %stationXcoord(thisrow:thisrow+station_Lengths{i}-1) = AllStationData{FitSet(i),5}(1);
        %thisrow = thisrow+station_Lengths{i};
    end 
    %stationElev = cell2mat(AllStationData(FitSet,2));
    T0BetaGamma = [ones(size(Tstation_LT)) stationXcoord stationElev]\Tstation_LT;
    result = [T0BetaGamma; mean(stationElev)];
end