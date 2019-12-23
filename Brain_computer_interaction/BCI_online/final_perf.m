function [hyper_perf,thresh_vect,consec_vect] = final_perf(rolling_data, header, delay_data, partition, params)
% Computes mean MCC score for all hyperparameters tested based on data
% input and partition
% OUTPUTS
% Matrix of size n threshold values tested * m consecutive_sample values
% tested. For each combination mean MCC score on all folds
% vectors of threshold values and number of consecutive samples tested
%
% compute distance data point to hyper plane and map it to logistic
% function
% off-diagonal elements not reliable enough for clasification


thresh_vect = params.thresh_vect;
consec_vect = params.consec_vect;
nb_feat = params.nb_feat;


% Prepare results matrix
results_tab = zeros(length(thresh_vect), length(consec_vect),10);


% Where we expect to find ErrP
sampleTrial = floor(0.2*header.SampleRate:0.8*header.SampleRate);

% electrodes selected
% channels_list = [1 3 4 5 8 9 10];


for fold = 1:partition.NumTestSets
    message = ['Processing fold ',num2str(fold),'/',num2str(partition.NumTestSets), '...'];
    disp(message)

    % run through folds and determine MCC via TP,TN,FP,FN counts

%     TPTNFPFN = zeros(length(thresh_vect), length(consec_vect), 4);

    index = [1:partition.NumObservations];
    idx_comp.tr_index = index(partition.training(fold));
    idx_comp.te_index = index(partition.test(fold));

    % delay and labels for training and test data
    idx_comp.tr_delay = delay_data(partition.training(fold),1);
    idx_comp.tr_labels = delay_data(partition.training(fold),2);
    idx_comp.te_labels = delay_data(partition.test(fold),2);
    idx_comp.te_delay = delay_data(partition.test(fold),1);


    [tr_feat, coeff_cca, coeff_pca] = format_train_cca_pca(rolling_data,idx_comp,sampleTrial,nb_feat);


    % train classifier
    classifier = fitcdiscr(tr_feat, idx_comp.tr_labels, 'discrimType','quadratic');


    % MCC score for all trials in fold
    TPTNFPFN = online_perf(rolling_data,idx_comp, classifier, coeff_cca, coeff_pca, params);
    
    % compute MCC for each fold
    TP = TPTNFPFN(:,:,1);
    TN = TPTNFPFN(:,:,2);
    FP = TPTNFPFN(:,:,3);
    FN = TPTNFPFN(:,:,4);
    

    results_tab(:,:,fold) = (TP.*TN-FP.*FN)./sqrt((TP+FP).*(TP+FN).*(TN+FP).*(TN+FN));
    
end

hyper_perf = mean(results_tab,3);

end
