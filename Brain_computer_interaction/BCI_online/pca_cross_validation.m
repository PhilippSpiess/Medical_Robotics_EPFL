function [testMCC, trainMCC, testTNR, trainTNR] = pca_cross_validation(Ntest, partition, trainData, trainLabels)
%returns array where each row corresponds to the error for each fold (= 1 column) for the
%number of features corresponding to the row (Ntest)
    
% initiate arrays
    testMCC = zeros(Ntest,partition.NumTestSets);
    trainMCC = zeros(Ntest,partition.NumTestSets);
    testTNR = zeros(Ntest,partition.NumTestSets);
    trainTNR = zeros(Ntest,partition.NumTestSets);

    for i = 1:partition.NumTestSets
        %iterate on folds
        
        for nbfeat = 1:Ntest

            data_train = trainData(partition.training(i),:);
            
            %apply PCA inside the loop
            [coeff, score, ~] = pca(data_train);
            features_train = score(:,1:nbfeat);
            
            % use same PCA coefficient for testing fold
            features_test = trainData(partition.test(i),:)*coeff;
            features_test = features_test(:,1:nbfeat);
            
            % use diagL classifier
            classifier = fitcdiscr(features_train,trainLabels(partition.training(i)),'discrimType','quadratic');
            
            % make predictions
            testprediction = predict(classifier,features_test);
            trainprediction = predict(classifier,features_train);
            
            %compute error: MCC score and TNR are chosen
            [teMCC, teTNR] = MCCscore(testprediction,trainLabels(partition.test(i)));
            [trMCC, trTNR] = MCCscore(trainprediction,trainLabels(partition.training(i)));
            
            % feed in the results in array
            testMCC(nbfeat,i) = teMCC;
            trainMCC(nbfeat,i) = trMCC;
            testTNR(nbfeat,i) = teTNR;
            trainTNR(nbfeat,i) = trTNR;
        end
    end
end
        