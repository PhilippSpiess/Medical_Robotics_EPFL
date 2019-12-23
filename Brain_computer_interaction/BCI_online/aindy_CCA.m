function [coeff,data_tr] = aindy_CCA(rolling_data,idx_comp,sampleTrial)
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



% rot_tr = zeros(308,16,sum(tr_labels));
% norot_tr = zeros(308,16,length(tr_labels)-sum(tr_labels));
all_tr = zeros(308,16,length(idx_comp.tr_labels));

for tr_trial = 1:length(idx_comp.tr_index)
    tr_sample = rolling_data{1,idx_comp.tr_index(tr_trial)};
    tr_sample = tr_sample(idx_comp.tr_delay(tr_trial)+sampleTrial,:);
    all_tr(:,:, tr_trial) = tr_sample;
end
    
rot_tr = all_tr(:,:,idx_comp.tr_labels==1);
norot_tr = all_tr(:,:,idx_comp.tr_labels==0);
% rot_tr = permute(rot_tr,[2,3,1]);
% norot_tr = permute(norot_tr,[2,3,1]);

X1 = [];
X2 = [];
for t = 1:size(rot_tr,3)
    X1 = cat(1,X1,rot_tr(:,:,t));
end

for t = 1:size(norot_tr,3)
    X2 = cat(1,X2,norot_tr(:,:,t));
end

X1 = X1';
X2 = X2';
X = [X1 X2];
% averaging erroneous signal over each trial.
avg1 = mean(rot_tr,3)';
avg2 = mean(norot_tr,3)';
Y1 = repmat(avg1,1,size(rot_tr,3));
Y2 = repmat(avg2,1,size(norot_tr,3));
Y = [Y1 Y2];
[coeff,B,~,U] = canoncorr(X',Y');

% X = X';
% averaging erroneous signal over each trial.
% avg = mean(errorTrain_epoch,3)';
% Y = repmat(avg,1,size(errorTrain_epoch,3));
% [A,B,~,U] = canoncorr(X',Y');

data_tr = zeros(308,4,length(idx_comp.tr_index));
for tr_trial = 1:length(idx_comp.tr_index)
    data_tr(:,:, tr_trial) = all_tr(:,:, tr_trial)*coeff(:,1:4);
end

coeff=coeff(:,1:4);

end

