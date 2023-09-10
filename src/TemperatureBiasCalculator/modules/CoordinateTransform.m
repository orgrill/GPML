function SNAP_Rect = CoordinateTransform(ProjectedCRS, CWS_SNAPData, Skip_Transform)
    addpath(fullfile("toolbox", "kmz2struct"));
    addpath(fullfile("toolbox", "ArcticMappingToolbox"));
    addpath(fullfile("toolbox", "GPR", "gpml-matlab-v4.2-2018-06-11", "cov"));

    %% Getting everything into the same coordinate frame
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
    [AKx, AKy] = projfwd(ProjectedCRS,alaska.Latitude,alaska.Longitude);
    AlaskaPoly = polyshape(AKx,AKy);
    plot(AlaskaPoly,'FaceColor','none');

    return %[Xdown, Ydown, SNAP_Rect_Rot]
end
