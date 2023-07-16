function [DownSampleOut] = InterpolateSNAP(train_x,SNAPTemps,test_x,TheseDays)
% This Function Downsamples the SNAP data to a set of locations, the xtrain
% must be gridded! This is called on the local machine
    % Here is some manual scaling for the grid
    train_x = train_x./1e6;
    test_x = test_x./1e6;
    % Use GpyTorch to krig the temp data 
    % Move everything to numpy, then to pytorch
    pyTrainingX = py.torch.from_numpy(py.numpy.array(train_x));
    pyGrid = py.torch.from_numpy(py.numpy.array(train_x));
    pyTestX = py.torch.from_numpy(py.numpy.array(test_x));
    % We need the days in the TestSet
    DownSampleOut = zeros(size(test_x,1),length(TheseDays));
    addpath('src')
    for i=1:length(TheseDays) %:size(CWS_SNAP_Rect.t2max,3)
         disp(['Downsampling Day ', num2str(i)])
         train_y = reshape(SNAPTemps(:,:,TheseDays(i)),[],1);
         pyTrainingY = py.torch.from_numpy(py.numpy.array(train_y));
         outvars = py.GSML_GPGRID.GPGrid(pyTrainingX,pyTrainingY, pyGrid, pyTestX);
         outcell = cell(outvars);
         %trainXout = double(outcell{1}.cpu().numpy);
         %trainYout = double(outcell{2}.cpu().numpy);
         %testXout = double(outcell{3}.cpu().numpy);
         testYout = double(outcell{4}.cpu().numpy);
         %testPred = double(outcell{4}.cpu().numpy);
         %myparam = py.torch.from_numpy(py.numpy.array(double(outcell{5})));
         DownSampleOut(1:size(test_x,1),i) = testYout';
    end
end
