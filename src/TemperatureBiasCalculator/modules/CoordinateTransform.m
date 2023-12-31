function [SNAP_Rect, SNAP_Rect_Rot] = CoordinateTransform(ProjectedCRS, CWS_SNAPData, Skip_Transform)
    addpath(fullfile("toolbox", "kmz2struct"));
    addpath(fullfile("toolbox", "ArcticMappingToolbox"));
    addpath(fullfile("toolbox", "GPR", "gpml-matlab-v4.2-2018-06-11", "cov"));

    %% Getting everything into the same /coordinate frame
    % GCM (SNAP) Data is in polar stereographic, lets move it to lat/long, then
    % to our projection grid

    % The lat/long moved into the new grid is not rectilinear
    % The following function moves it and interpolates, created a new SNAP data
    % object, requires an external function, CurveGrid2Rect
    if Skip_Transform 
        SNAP_Rect = load(fullfile("data", "CWS_SNAP_Rect.mat")).CWS_SNAP_Rect;
    else
        [SNAP_xpolar, SNAP_ypolar] = ndgrid(CWS_SNAPData.xc, CWS_SNAPData.yc);
        [SNAP_lat, SNAP_long] = psn2ll(SNAP_xpolar, SNAP_ypolar, 'TrueLat', 64, 'meridian', -152, 'EarthRadius', 6370000); 
        [SNAPx, SNAPy] = projfwd(ProjectedCRS, SNAP_lat, SNAP_long); % This is 20km resolution
        SNAP_Rect = MovetoRectilinear_Interpolate(SNAPx, SNAPy, CWS_SNAPData);
    end
    
    states = shaperead('usastatehi.shp','UseGeoCoords',true);
    alaska = geoshape(states(2,:));
    [AKx, AKy] = projfwd(ProjectedCRS, alaska.Latitude, alaska.Longitude);
    AlaskaPoly = polyshape(AKx,AKy);
    plot(AlaskaPoly,'FaceColor','none');

    %{ 
    Rotate - 
        We also have to rotate everything, this is not a regridding, just a
        coordinate transform, we perform the GP regression in the rotated
        rectilinear axes, all the plotting is in the non-rotated frame.  We also
        only rotate the gridded data, it doesn't matter if the training data is scattered
    %}
    vectorX = reshape(SNAP_Rect.xgrid,[],1);
    vectorY = reshape(SNAP_Rect.ygrid,[],1);
    SNAP_Rect_Rot = [vectorX vectorY];
    theta =pi/2+atan(SNAP_Rect.lineParams(1)); 
    rot = [cos(theta) -sin(theta); sin(theta) cos(theta)];
    points=(SNAP_Rect_Rot)*rot;
    % Even After rotation, there is still numeric error, discretize to get everything
    % on an exact grid
    Bin = discretize(points(:,1),size(SNAP_Rect.xgrid,1));
    for i=1:size(SNAP_Rect.xgrid,1)
        pxg(i) = mean(points(Bin==i,1));
    end
    Bin = discretize(points(:,2),size(SNAP_Rect.xgrid,2));
    for i=1:size(SNAP_Rect.xgrid,2)
        pyg(i) = mean(points(Bin==i,2));
    end
    % Now re-grid using the discretized values
    [pxg, pyg] = meshgrid(pxg,pyg);
    pxg = fliplr(pxg');
    pyg = fliplr(pyg');
    vectorX = reshape(pxg,[],1);
    vectorY = reshape(pyg,[],1);
    SNAP_Rect_Rot = [vectorX vectorY]; % SNAP Coordinates Rotated
    % These are the coordinates rotated back, just to check!
    % rot = [cos(-theta) -sin(-theta); sin(-theta) cos(-theta)];
    % rotXY=XY*(rot); 
    % Xqr = reshape(rotXY(:,1), size(pxg,1), []);
    % Yqr = reshape(rotXY(:,2), size(pyg,1), []);
    return 
end