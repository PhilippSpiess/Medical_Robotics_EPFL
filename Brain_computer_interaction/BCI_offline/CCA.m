function [correctTrain_filtered_epoch,errorTrain_filtered_epoch,A] = CCA(correctTrain_epoch,errorTrain_epoch)
% Function which computes a CCA based spatial filter, directly using
% Matlab's "canoncorr" function. This function is applied on the erroneous
% signal to get the filter (matrix A) and then applied both to the
% erroneous and correct signals to make sure to filter the whole signal.
% INPUTS:
% correctTrain_epoch: epoched signal of the correct behaviour for the
% training set.
% errorTrain_epoched: epoched signal of the erroneous behavour for the
% training set.
% OUTPUTS: same as inputs, after applying the spatial filter.
e_trials = size(errorTrain_epoch,3);
c_trials = size(correctTrain_epoch,3);

X1 = [];
X2 = [];
errorTrain_filtered_epoch = [];
correctTrain_filtered_epoch = [];
% concatenation of each trial followed by transpose operation to have a n
% channels * (k trials * m samples) matrix.
for t = 1:e_trials
    X1 = cat(1,X1,errorTrain_epoch(:,:,t));
end

for t = 1:c_trials
    X2 = cat(1,X2,correctTrain_epoch(:,:,t));
end


%channels_list= [1 3 4 5 8 9 10];
X1 = X1';
X2 = X2';
X = [X1 X2];
% averaging erroneous signal over each trial.
avg1 = mean(errorTrain_epoch,3)';
avg2 = mean(correctTrain_epoch,3)';
Y1 = repmat(avg1,1,size(errorTrain_epoch,3));
Y2 = repmat(avg2,1,size(correctTrain_epoch,3));
Y = [Y1 Y2];
[A,B,~,U] = canoncorr(X',Y');

% X = X';
% averaging erroneous signal over each trial.
% avg = mean(errorTrain_epoch,3)';
% Y = repmat(avg,1,size(errorTrain_epoch,3));
% [A,B,~,U] = canoncorr(X',Y');


for t = 1:e_trials
    tmp = errorTrain_epoch(:,:,t)*A(:,1:4);
    errorTrain_filtered_epoch = cat(3,errorTrain_filtered_epoch,tmp);
end

for t = 1:c_trials
    tmp = correctTrain_epoch(:,:,t)*A(:,1:4);
    correctTrain_filtered_epoch = cat(3,correctTrain_filtered_epoch,tmp);
end


end

