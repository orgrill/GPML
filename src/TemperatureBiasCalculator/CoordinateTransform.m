function [Xdown, Ydown, SNAP_Rect_Rot] = CoordinateTransform(ProjectedCRS)
    % Loading data
    CWS_SNAPData = load("data\CWS_SNAP_TMAX.mat");
    %load('data\CWS_DEM.mat')
    WS_Data = load("data\CWS_StationData.mat");
    SubRegion = kmz2struct('data\CopperRiverWatershed.kmz');
    %% Getting everything into the same coordinate frame
    % GCM (SNAP) Data is in polar stereographic, lets move it to lat/long, then
    % to our projection grid

    [SNAP_xpolar, SNAP_ypolar] = ndgrid(CWS_SNAPData.xc,CWS_SNAPData.yc);
    [SNAP_lat, SNAP_long] = psn2ll(SNAP_xpolar,SNAP_ypolar,'TrueLat',64,'meridian',-152,'EarthRadius',6370000); 
    [SNAPx, SNAPy] = projfwd(ProjectedCRS,SNAP_lat,SNAP_long); % This is 20km resolution

    % The lat/long moved into the new grid is not rectilinear
    % The following function moves it and interpolates, created a new SNAP data
    % object, requires an external function, CurveGrid2Rect
    % [CWS_SNAP_Rect] = MovetoRectilinear_Interpolate(SNAPx,SNAPy,CWS_SNAPData);
    % save('CWS_SNAP_Rect','CWS_SNAP_Rect');
    % We only have to do this once, so it is best to save the result, then load
    load("data\CWS_SNAP_Rect.mat")
    states = shaperead('usastatehi.shp','UseGeoCoords',true);
    alaska = geoshape(states(2,:));
    [AKx, AKy] = projfwd(ProjectedCRS,alaska.Latitude,alaska.Longitude);
    AlaskaPoly = polyshape(AKx,AKy);
    %% Rotate
    % We also have to rotate everything, this is not a regridding, just a
    % coordinate transform, we perform the GP regression in the rotated
    % rectilinear axes, all the plotting is in the non-rotated frame.  We also
    % only rotate the gridded data, it doesn't matter if the training data is
    % scattered
    vectorX = reshape(CWS_SNAP_Rect.xgrid,[],1);
    vectorY = reshape(CWS_SNAP_Rect.ygrid,[],1);
    SNAP_Rect = [vectorX vectorY];
    theta =pi/2+atan(CWS_SNAP_Rect.lineParams(1)); 
    rot = [cos(theta) -sin(theta); sin(theta) cos(theta)];
    points=(SNAP_Rect)*rot;
    % Even After rotation, there is still numeric error, discretize to get everything
    % on an exact grid
    Bin = discretize(points(:,1),size(SNAPx,1));
    for i=1:size(SNAPx,1)
        pxg(i) = mean(points(Bin==i,1));
    end
    Bin = discretize(points(:,2),size(SNAPx,2));
    for i=1:size(SNAPx,2)
        pyg(i) = mean(points(Bin==i,2));
    end
    % Now re-grid using the discretized values
    [pxg, pyg] = meshgrid(pxg,pyg);
    pxg = fliplr(pxg');
    pyg = fliplr(pyg');
    vectorX = reshape(pxg,[],1);
    vectorY = reshape(pyg,[],1);
    train_x = [vectorX vectorY]; % SNAP Coordinates Rotated
    SNAP_Rect_Rot = [vectorX vectorY]; % SNAP Coordinates Rotated
    % These are the coordinates rotated back, just to check!
    % rot = [cos(-theta) -sin(-theta); sin(-theta) cos(-theta)];
    % rotXY=XY*(rot); 
    % Xqr = reshape(rotXY(:,1), size(pxg,1), []);
    % Yqr = reshape(rotXY(:,2), size(pyg,1), []);

    % Eventually we need to downsample for every day at a very fine grid,
    % currently we are only set up to downsample to the test stations
    % This builds a grid that we could use for the very fine downsampling
    factor = 20;
    downGrid = apxGrid('create',SNAP_Rect_Rot,1,[size(SNAPx,1)*factor size(SNAPx,2)*factor]); 

    % Lets Go in 6 on each edge, for some reason it extends outside the
    % training region
    [Xdown, Ydown] = meshgrid(downGrid{1}(6:end-5),downGrid{2}(6:end-5));
    %test_x = [reshape(Xdown,[],1) reshape(Ydown,[],1)]; 

    return %[Xdown, Ydown, SNAP_Rect_Rot]
end
