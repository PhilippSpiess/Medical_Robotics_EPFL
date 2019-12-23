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
%my_names([1:5]) = [];
%my_names([1:4]) = [];


data_out = prep_data(my_names);

labels = data_out(:,1);

all_data = data_out(:,2:size(data_out,2));

%% Feature contruction

%[data_coeff,score,latent,tsquared,explained,mu] = pca(all_data);

%all_data = all_data*data_coeff;

rng('default')
labelspartition = cvpartition(labels,'kfold',10);

% Maximal number of PCs to test
Ntest = 100;
[testMCC, trainMCC, testTNR, trainTNR] = pca_cross_validation(Ntest, labelspartition, all_data, labels);

% x axis
nbfeature = 1:Ntest;

figure;
plot(nbfeature,mean(transpose(trainMCC(:,:))),'b','LineWidth',2)
hold on;
plot(nbfeature,mean(transpose(testMCC(:,:))),'r','LineWidth',2)
