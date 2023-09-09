function [CWS_SNAP_Downscaled, FitDays, FitSet, TestSet, StationDownScaleParams] = TopographicalDownscaling(AllStationData, SNAP_Rect, SNAP_Rect_Rot, Xdown, Ydown)
    %%
    %% Calculate Elevation Parameters for Downscaled Grid
    % % Elevations at the downsampled Grid (This cuts the edges quite a bit)
    % This requires you to load CWS_DEM, which is big, so I downsampled it and
    CWS_SNAP_Downscaled.Elevation = load(fullfile("data", "CWS_SNAP_Fine.mat")).CWS_SNAP_Fine.Elevation;
    % TODO: How was Elevations calculated in CWS_SNAP_Rect.m? It's in the file, but there's no code in CoordinateTransform that would explain it

    %% Split into test and fit sets, set up and execute EQM and topgraphical downscaling
    % Load the raw station data, we need to divide this into two parts, 80/20
    FitSet = sort(randperm(size(AllStationData,1),round(.8*size(AllStationData,1))));
    % Load a particular test/train split, the interpolation is really slow
    % FitSet = load(fullfile("data", "FitSet.mat")).FitSet;
    
    % Fit the Topographical Downscaling parameters (White 2016)
    [StationDownScaleParams] = GSML_Topo_Downscale(AllStationData, FitSet); %[T0; Beta; Gamma; zref]
    TestSet = sort(setdiff(1:length(AllStationData), FitSet));

    % First move the coarse tmax data to reference
    % Topographic Downscaling, we need to move GCM and Station data to a
    % reference elevation
    SNAP_temp_ref = zeros(size(SNAP_Rect.t2max));
    for i = 1:length(SNAP_Rect.Days)
        SNAP_temp_ref(:, :, i) = SNAP_Rect.t2max(:, :, i)-StationDownScaleParams(3)*(StationDownScaleParams(4)-SNAP_Rect.Elevations);
    end

    FitDays = [];
    station_Lengths = cellfun(@length, AllStationData(FitSet,3), 'UniformOutput', false);
    for i = 1:size(station_Lengths, 1)
        FitDays = [FitDays; [AllStationData{FitSet(i), 3}]];
    end
    FitDays = unique(FitDays);  % These are the days present in the fit stations

    %% Downsample the NonRef SNAP Data to the fine grid
    % This is for the spatial check, will probably only work for 1 day, need to
    % change so it works for a range
    TheseDaysIndex = find(SNAP_Rect.Days == datetime(2085, 4, 26));
    CWS_SNAP_Downscaled.Tmax = zeros(size(Xdown, 1), size(Xdown, 2), length(TheseDaysIndex));
    CWS_SNAP_Downscaled.Days = [SNAP_Rect.Days(TheseDaysIndex)];
    vectorX = reshape(Xdown, [], 1);
    vectorY = reshape(Ydown, [], 1);
    for i = 1:length(TheseDaysIndex)
        disp(['Downsampling Iteration ', num2str(i)])
        InterpOut = InterpolateSNAP(SNAP_Rect_Rot, SNAP_temp_ref, [vectorX vectorY], TheseDaysIndex);
        CWS_SNAP_Downscaled.Tmax(:, :, i) = reshape(InterpOut, size(Xdown));
    end

    %% Move everything Back, unrotate, and display in the original coordinate frame
    theta = pi/2 + atan(SNAP_Rect.lineParams(1));
    rot = [cos(-theta) -sin(-theta); sin(-theta) cos(-theta)];
    rotXY = [vectorX vectorY]*(rot); 
    Xqr = reshape(rotXY(:, 1),  size(Xdown, 1),  []);
    Yqr = reshape(rotXY(:, 2),  size(Xdown, 1),  []);
    %
    CWS_SNAP_Downscaled.Xgrid = Xqr;
    CWS_SNAP_Downscaled.Ygrid = Yqr;
    figure
    
    hold on
    mapshow(Xqr, Yqr, CWS_SNAP_Downscaled.Tmax(:, :, 1), 'DisplayType', 'surface', 'FaceAlpha', 1);

    return 
end