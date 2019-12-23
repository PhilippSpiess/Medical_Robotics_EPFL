function [tr_feat, coeff_cca, coeff_pca] = format_train_cca_pca(rolling_data,idx_comp,sampleTrial,nb_feat)
% Computes training features for classification from raw data
% OUTPUTS
% tr_feat: matrix of training samples * features
% coeff_cca: cca coefficients used for test data
% coeff_pca: pca coefficients used for test data


[coeff_cca,data_tr] = aindy_CCA(rolling_data,idx_comp,sampleTrial);

data_train = zeros(length(idx_comp.tr_delay),ceil(length(sampleTrial)/8)*size(coeff_cca,2));


for tr_trial = 1:length(idx_comp.tr_index)
    tr_sample = data_tr(:,:,tr_trial);


    tr_sample = transpose(downsample(tr_sample,8));

    for cca_feat = 1:size(tr_sample,1)
        st = (cca_feat-1)*size(tr_sample,2)+1;
        data_train(tr_trial,st:st+size(tr_sample,2)-1) = tr_sample(cca_feat,:);
    end
end


% compute pca coeffs on training data
[coeff_pca, score, ~] = pca(data_train);
tr_feat = score(:,1:nb_feat);


end