function [DownSampleOut,SubFitSet] = FutureBias(train_x,train_y,test_x,SubFitSet,TrainFlag)
    if TrainFlag == 1    
        NumberTrainPoints = 75000;
        SubFitSet = sort(randperm(size(train_x,1),NumberTrainPoints));
    end
    train_x = [train_x(SubFitSet,1) (train_x(SubFitSet,2).*.3048)];
    train_y = train_y(SubFitSet);
    test_x(:,1) = test_x(:,1);
    test_x(:,2) = (test_x(:,2).*.3048);
    save('Z:\PythonTrainTest','train_x','train_y','test_x','TrainFlag');
    DownSampleOut = [];
end