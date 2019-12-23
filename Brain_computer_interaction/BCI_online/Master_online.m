%% Initialization

clear all
close all
clc

addpath(genpath('matlabFunctions'))
addpath(genpath('a3_20190603'))
addpath(genpath('ad4_20192502_mock'))
addpath(genpath('b4_20192603'))


% get all file names from training data
files = dir(strcat(pwd,'\b4_20192603'));
%files = dir(strcat(pwd,'\a3_20190603'));
%files = dir(strcat(pwd,'\ad4_20192502_mock'));

my_names = {files.name};

% deletes non relevant folders in cell
my_names([1:3 14]) = [];


%% sliding window next week for online decoding (32Hz)

rolling_data = [];
delay_data = [];

for folder = 1:numel(my_names)
        
        %import data from trial
        [signal, header] = sload(strcat(char(my_names(folder)),'.gdf'));
        behavior = single(dlmread(strcat(char(my_names(folder)),'.txt')));
        
        eog = signal(:,17:19);
        eeg = all_filters(signal(:,1:16),eog);
        trig = signal(:,33);
        
        % epoching
        % -0.25 0.5 pour posterior
        [epoch_data, delay] = epoching2(eeg, trig, behavior,0.25,0);
        
        rolling_data = [rolling_data epoch_data];
        delay_data = [delay_data ; delay];
end


%% threshold
results_tab = cross_val_online(rolling_data, header, delay_data, 25);

%% continuous classification

results_contin = continuous_decoding(rolling_data, header, delay_data, 20, 5, 3);

max(mean(results_contin(:,2:11,:),2))
%0.6155 pr 0.95 6 consecutive

%% plot posterior probs

[testfold_labels, poterior_prob] = plot_posterior(rolling_data, header, delay_data, 20, -1,1);
axis_vect = [-1:0.0313:0.4];
prob_plot = [];
for j=1:10
    prob_plot = cat(1, prob_plot, poterior_prob(:,:,j));
end


rot_prob = prob_plot(testfold_labels==1,:);
norot_prob = prob_plot(testfold_labels==0,:);
figure;
imagesc(norot_prob)
xticks([0:8:64])
xticklabels({'-1','-0.75','-0.5','-0.25','0','0.25','0.5','0.75','1'})
figure;
imagesc(rot_prob)
xticks([0:8:64])
xticklabels({'-1','-0.75','-0.5','-0.25','0','0.25','0.5','0.75','1'})
figure;
plot(axis_vect, mean(rot_prob,1))
figure;
plot(axis_vect, mean(norot_prob,1))

%%

[hyper_perf,thresh_vect,consec_vect] = final_perf(rolling_data, header, delay_data, partition, params);

%% hyperparameter optimization and performances

% number of pca features
params.nb_feat = 20;

% minimal threshold, maximal threshold and number of values tested
thresh = [0.65,0.95,10];
params.thresh_vect = linspace(thresh(1),thresh(2),thresh(3));

% number of consecutive samples tested
params.consec_vect = [3:9];


% partitions for cross-val and performance measure
first_partition = cvpartition(delay_data(:,2),'kfold',10);
second_partition = cvpartition(delay_data(first_partition.training(1),2),'kfold',9);

% 10th block: unseen data for testing
tenblock_data = rolling_data(first_partition.test(1));
tenblock_delay = delay_data(first_partition.test(1),:);

% First 9 blocks for hyperparam optimization and training of classifier
delay_9 = delay_data(first_partition.training(1),:);
rolling_9 = rolling_data(first_partition.training(1));



[hyper_perf,thresh_vect,consec_vect] = final_perf(rolling_9, header, delay_9, second_partition, params);

[max_mcc,maxidx] = max(hyper_perf(:));
[threshidx,consecidx] = ind2sub(size(hyper_perf),maxidx);




%%
idx_tr.tr_index = [1:length(delay_9(:,1))];
idx_tr.tr_delay = delay_9(:,1);
idx_tr.tr_labels = delay_9(:,2);
[tr_feat, coeff_cca, coeff_pca] = format_train_cca_pca(rolling_9,idx_tr,sampleTrial,params.nb_feat);

% train classifier
classifier = fitcdiscr(tr_feat, idx_tr.tr_labels, 'discrimType','quadratic');


params_final.nb_feat = 20;
params_final.thresh_vect = thresh_vect(threshidx);
params_final.consec_vect = consec_vect(consecidx);
idx_tr.te_index = [1:length(tenblock_delay(:,1))];
idx_tr.te_delay = tenblock_delay(:,1);
idx_tr.te_labels = tenblock_delay(:,2);


TPTNFPFN = online_perf(tenblock_data,idx_tr, classifier, coeff_cca, coeff_pca, params_final);

TP = TPTNFPFN(:,:,1);
TN = TPTNFPFN(:,:,2);
FP = TPTNFPFN(:,:,3);
FN = TPTNFPFN(:,:,4);

MCC_delafolie = (TP*TN-FP*FN)/sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN));
%0.7218!! 0.8833 thresh and 6 consec
