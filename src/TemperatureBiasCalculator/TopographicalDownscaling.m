function CWS_SNAP_Downscaled = TopographicalDownscaling(AllStationData)
    %%
    %% Calculate Elevation Parameters for Downscaled Grid
    % % Elevations at the downsampled Grid (This cuts the edges quite a bit)
    % This requires you to load CWS_DEM, which is big, so I downsampled it and
    load("data\CWS_SNAP_Fine.mat")
    CWS_SNAP_Downscaled.Elevation = CWS_SNAP_Fine.Elevation;  % We get a bunch of NANs near the borders
    clear CWS_SNAP_Fine;
    % figure
    % plot(AlaskaPoly,'FaceColor','none')
    % hold on
    % mapshow(CWS_SNAP_Downscaled.Xgrid,CWS_SNAP_Downscaled.Ygrid,CWS_SNAP_Downscaled.Elevation,'DisplayType','surface','FaceAlpha',1);
    %% Split into test and fit sets, set up and execute EQM and topgraphical downscaling
    % Load the raw station data, we need to divide this into two parts, 80/20
    % FitSet = sort(randperm(size(AllStationData,1),round(.8*size(AllStationData,1))));
    % Load a particular test/train split, the interpolation is really slow
    load ("data\FitSet.mat")
    % Fit the Topographical Downscaling parameters (White 2016)
    [StationDownScaleParams] = GSML_Topo_Downscale(AllStationData, FitSet); %[T0; Beta; Gamma; zref]
    TestSet = sort(setdiff(1:length(StationX),FitSet));
    % First move the coarse tmax data to reference
    CWS_SNAP_Rect.t2ref = zeros(size(CWS_SNAP_Rect.t2max));
    for i=1:length(CWS_SNAP_Rect.Days)
        CWS_SNAP_Rect.t2ref(:,:,i) = CWS_SNAP_Rect.t2max(:,:,i)-StationDownScaleParams(3)*(StationDownScaleParams(4)-CWS_SNAP_Rect.Elevations);
    end
    % Topographic Downscaling, we need to move GCM and Station data to a
    % reference elevation
    SNAP_temp_Ref = zeros(size(CWS_SNAP_Rect.t2max));
    for i = 1:length(CWS_SNAP_Rect.Days)
        %SNAP_temp_Ref(:,:,i) = CWS_SNAP_Rect.t2max(:,:,i)-StationDownScaleParams(3)*(StationDownScaleParams(4).*ones(size(CWS_SNAP_Rect.xgrid))-CWS_SNAP_Rect.Elevations);
        SNAP_temp_Ref(:,:,i) = CWS_SNAP_Rect.t2ref(:,:,i);
    end
    station_Lengths = cellfun(@length, AllStationData(FitSet,3), 'UniformOutput', false);
    for i=1:size(station_Lengths)
        temps = cell2mat(AllStationData(FitSet(i),4))-StationDownScaleParams(3)*(StationDownScaleParams(4)-cell2mat(AllStationData(FitSet(i),2)).*.3048);
        Tstation_Ref{i} = temps; 
        Tstation_NonRef{i} = cell2mat(AllStationData(FitSet(i),4));
    end
    FitDays = [];
    for i=1:size(station_Lengths,1)
        FitDays = [FitDays; [AllStationData{FitSet(i),3}]];
    end
    FitDays = unique(FitDays);  % These are the days present in the fit stations
    %% Downsample the NonRef SNAP Data to the fine grid
    % This is for the spatial check, will probably only work for 1 day, need to
    % change so it works for a range
    TheseDaysIndex = find(CWS_SNAP_Rect.Days == datetime(2085, 4, 26));
    CWS_SNAP_Downscaled.Tmax = zeros(size(Xdown, 1), size(Xdown, 2), length(TheseDaysIndex));
    CWS_SNAP_Downscaled.Days = [CWS_SNAP_Rect.Days(TheseDaysIndex)];
    vectorX = reshape(Xdown, [], 1);
    vectorY = reshape(Ydown, [], 1);
    for i = 1:length(TheseDaysIndex)
        disp(['Downsampling Day ', num2str(i)])
        InterpOut = InterpolateSNAP(SNAP_Rect_Rot, CWS_SNAP_Rect.t2ref, [vectorX vectorY], TheseDaysIndex);
        CWS_SNAP_Downscaled.Tmax(:, :, i) = reshape(InterpOut, size(Xdown));
    end

    %% Move everything Back, unrotate, and display in the original coordinate frame
    rot = [cos(-theta) -sin(-theta); sin(-theta) cos(-theta)];
    rotXY=[vectorX vectorY]*(rot); 
    Xqr = reshape(rotXY(:,1), size(Xdown,1), []);
    Yqr = reshape(rotXY(:,2), size(Xdown,1), []);
    %
    CWS_SNAP_Downscaled.Xgrid = Xqr;
    CWS_SNAP_Downscaled.Ygrid = Yqr;
    figure
    plot(AlaskaPoly,'FaceColor','none')
    hold on
    mapshow(Xqr,Yqr,CWS_SNAP_Downscaled.Tmax(:,:,1),'DisplayType','surface','FaceAlpha',1);

    return %CWS_SNAP_Downscaled
end