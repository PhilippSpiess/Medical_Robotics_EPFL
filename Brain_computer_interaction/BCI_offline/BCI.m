clear all
close all
clc

%% load the data - spectral filter  

%Philipp dataset - COMPETITION DATASET
addpath(genpath('matlabFunctions'))
addpath(genpath('b4_20192603'))
sets = {'20192603164635','20192603165019','20192603165401','20192603165735','20192603170112','20192603170833','20192603171214','20192603171550','20192603171928','20192603172258'};

% % Aindy dataset
% addpath(genpath('matlabFunctions'))
% addpath(genpath('a3_20190603'))
% sets = {'20190603103539','20190603103959','20190603104405','20190603104817','20190603105219','20190603105616','20190603110042','20190603110442','20190603110853','20190603111247'};

% % Jan dataset
% addpath(genpath('matlabFunctions'))
% addpath(genpath('a5_20190803'))
% sets = {'20190803133736','20190803134138','20190803134522','20190803134900','20190803135233','20190803135625','20190803140020','20190803140415','20190803140752','20190803141135'};

% % Kaleem dataset
% addpath(genpath('matlabFunctions'))
% addpath(genpath('b2_20191303'))
% sets = {'20191303165056','20191303165439','20191303165821','20191303170158','20191303170532','20191303171058','20191303171437','20191303171817','20191303172208','20191303172542'};

all_feat_correct=[];
all_feat_error=[];

Xeeg=[];
Xeog=[];
Xtrig=[];
Xbehavior=[];

for i=1:numel(sets)

    name = strcat('b4_Asynchronous_',char(sets(i)))
%     name = strcat('a3_Asynchronous_',char(sets(i)))
    %name = strcat('a5_Asynchronous_',char(sets(i)))
    %name = strcat('b2_Asynchronous_',char(sets(i)))

    [signal, header] = sload(strcat(name,'.gdf'));
    behavior = single(dlmread(strcat(name,'.txt')));

    %SPECTRAL FILTER
    eeg = filter_eeg(signal(:,1:16));
    eog = signal(:,17:19);
    trig = signal(:,33);

    Xeeg= cat(1,Xeeg, eeg);
    Xeog=cat(1,Xeog, eog);
    Xtrig=cat(1,Xtrig, trig);
    Xbehavior=cat(1,Xbehavior, behavior);

end

%% EOG FILTER - (not used)
% b = filterEOG(Xeeg,Xeog);
% Xeeg = Xeeg-Xeog*b;

%% EPOCHING
[correct_epoch error_epoch] = epoch(Xeeg, header, Xtrig, Xbehavior, [0.2,0.8]);
all_epoch = [];
all_epoch = cat(3,all_epoch, correct_epoch);
all_epoch = cat(3,all_epoch, error_epoch);
labels = [];
labels = cat(1, labels, ones(1,size(correct_epoch,3)));
labels = cat(2, labels, zeros(1,size(error_epoch,3)));

%% 10 fold cross-validation

explain_pourcentage = linspace(55, 100, 10);
MCC = [];


for k_explain = 1:length(explain_pourcentage)
    
rng('default')
c = cvpartition(labels,'KFold',10);
TP=0;
TN=0;
FP=0;
FN=0;

for i = 1:c.NumTestSets
    trIdx = find(c.training(i));
    teIdx = find(c.test(i));
    
    X = all_epoch(:,:,trIdx);
    T = labels(trIdx);
    correct_epoch = X(:,:,find(T==1));
    error_epoch = X(:,:,find(T==0));
    
    Y = all_epoch(:,:,teIdx);
    Y_T = labels(teIdx);
    
    %CCA on the train_set
    [correcttrain_epoch, errortrain_epoch, A] = CCA(correct_epoch,error_epoch);
    [correct_feat error_feat] = construct_feat(correcttrain_epoch, errortrain_epoch);

    %CCA on test_set and construction of the test_features 
  
    
    test_set=[];
    e_trials = size(Y,3);
    for t = 1:e_trials
        tmp = Y(:,:,t)*A(:,1:4);
        test_set = cat(3,test_set,tmp);
    end
    test_set = downsample(test_set,8);
    test_feat = [];
    for m=1:size(test_set,3)
        elec = [];
        for j=1:4
            elec = cat(1, elec, test_set(:,j,m));
        end
        test_feat = cat(2, test_feat, elec);
    end
    test_feat = cat(1, zeros(1,size(test_set,3)), test_feat);
    
    correct_feat = cat(2, correct_feat, error_feat);
    train_feat = correct_feat;
  
    %TRAINING with train_feat and CLASSIFICATION with test_feat
    
 
    [TP_fold, TN_fold, FP_fold, FN_fold] = classifier(train_feat, T, test_feat, Y_T, explain_pourcentage(k_explain));
    TP=TP+TP_fold;
    TN=TN+TN_fold;
    FP=FP+FP_fold;
    FN=FN+FN_fold;
end
MCC(k_explain) = (TP*TN - FP*FN) / sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))

end
f = figure;
plot(explain_pourcentage, MCC, 'r');
xlabel('% of variance explained');
ylabel('MCC score');
title('MCC score as a function of the number the % of variance exlained by the first n number of principal components');


%% Performance

confusion_matrix = [TN FN ; FP TP];
Precision= TP/(TP+FP);
Recall= TP/(TP+FN);
Class_accuracy = 2*Precision*Recall / (Precision+Recall)
TNR = TN/(TN+FN);
TPR = TP/(TP+FP);

%% 20/40/60 degree decoding

%change the magnitude of detection in epoch.m
