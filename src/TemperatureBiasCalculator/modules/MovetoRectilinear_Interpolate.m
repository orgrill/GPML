%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%MovetoRectilinear_Interpolate & CurveGrid2Rect are both used to create the
%file CWS_SNAP_Rect, which is always the same so it was ran once and saved
%as a .mat file. No need to run this and the Curve2Grid functions
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function result = MovetoRectilinear_Interpolate(SNAPx, SNAPy, CWS_SNAPData)
    % Force the SNAP grid to be rectilinear so we can use gridded kernel
    % interpolation with our GP, this is essentially a regridding (a teeny one), so we should interpolate to find our values at the new grid.   
    [SNAP_Rect, result.lineParams] = CurveGrid2Rect(SNAPx, SNAPy);
    vectorX = reshape(SNAPx, [], 1);
    vectorY = reshape(SNAPy, [], 1);
    result.t2max = zeros(size(SNAPx, 1), size(SNAPy, 2), size(CWS_SNAPData.t2max, 3));
    for i=1:size(CWS_SNAPData.t2max,3) % Interpolate Each day of temp data to the new grid
           vectorT = reshape(double(CWS_SNAPData.t2max(:, :, i)), [], 1);
           F = scatteredInterpolant(vectorX, vectorY, vectorT);
           result.t2max(:, :, i) = reshape(F(SNAP_Rect(:, 1), SNAP_Rect(:, 2)), size(SNAPx, 2), size(SNAPx, 1))';
    end
    result.xgrid = reshape(SNAP_Rect(:, 1), size(SNAPx, 2), size(SNAPx, 1))';
    result.ygrid = reshape(SNAP_Rect(:, 2), size(SNAPx, 2), size(SNAPx, 1))';
    result.Days = CWS_SNAPData.Days;
end