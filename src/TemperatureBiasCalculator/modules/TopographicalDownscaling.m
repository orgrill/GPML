function SNAP_Downscaled = TopographicalDownscaling(AllStationData, SNAP_Rect, SNAP_Rect_Rot, StationDownScaleParams)
    %%
    %% Calculate Elevation Parameters for Downscaled Grid
    % % Elevations at the downsampled Grid (This cuts the edges quite a bit)
    % This requires you to load CWS_DEM, which is big, so I downsampled it and
    SNAP_Downscaled.Elevation = load(fullfile("data", "CWS_SNAP_Fine.mat")).CWS_SNAP_Fine.Elevation;
    % TODO: How was Elevations calculated in CWS_SNAP_Rect.m? It's in the file, but there's no code in CoordinateTransform that would explain it

    % First move the coarse tmax data to reference
    % Topographic Downscaling, we need to move GCM and Station data to a
    % reference elevation
    SNAP_temp_ref = zeros(size(SNAP_Rect.t2max));
    for i = 1:length(SNAP_Rect.Days)
        SNAP_temp_ref(:, :, i) = SNAP_Rect.t2max(:, :, i)-StationDownScaleParams(3)*(StationDownScaleParams(4)-SNAP_Rect.Elevations);
    end

    % Eventually we need to downsample for every day at a very fine grid,
    % currently we are only set up to downsample to the test stations
    % This builds a grid that we could use for the very fine downsampling
    factor = 20;
    downGrid = apxGrid('create',SNAP_Rect_Rot,1,[size(SNAP_Rect.xgrid,1)*factor size(SNAP_Rect.xgrid,2)*factor]); 

    % Lets Go in 6 on each edge, for some reason it extends outside the
    % training region
    [Xdown, Ydown] = meshgrid(downGrid{1}(6:end-5),downGrid{2}(6:end-5));
    %test_x = [reshape(Xdown,[],1) reshape(Ydown,[],1)]; 

    %% Downsample the NonRef SNAP Data to the fine grid
    % This is for the spatial check, will probably only work for 1 day, need to
    % change so it works for a range
    TheseDaysIndex = find(SNAP_Rect.Days == datetime(2085, 4, 26));
    SNAP_Downscaled.Tmax = zeros(size(Xdown, 1), size(Xdown, 2), length(TheseDaysIndex));
    SNAP_Downscaled.Days = [SNAP_Rect.Days(TheseDaysIndex)];
    vectorX = reshape(Xdown, [], 1);
    vectorY = reshape(Ydown, [], 1);
    for i = 1:length(TheseDaysIndex)
        disp(['Downsampling Iteration ', num2str(i)])
        InterpOut = InterpolateSNAP(SNAP_Rect_Rot, SNAP_temp_ref, [vectorX vectorY], TheseDaysIndex);
        SNAP_Downscaled.Tmax(:, :, i) = reshape(InterpOut, size(Xdown));
    end

    %% Move everything Back, unrotate, and display in the original coordinate frame
    theta = pi/2 + atan(SNAP_Rect.lineParams(1));
    rot = [cos(-theta) -sin(-theta); sin(-theta) cos(-theta)];
    rotXY = [vectorX vectorY]*(rot); 
    Xqr = reshape(rotXY(:, 1),  size(Xdown, 1),  []);
    Yqr = reshape(rotXY(:, 2),  size(Xdown, 1),  []);
    %
    SNAP_Downscaled.Xgrid = Xqr;
    SNAP_Downscaled.Ygrid = Yqr;
    figure
    
    hold on
    mapshow(Xqr, Yqr, SNAP_Downscaled.Tmax(:, :, 1), 'DisplayType', 'surface', 'FaceAlpha', 1);

    return 
end