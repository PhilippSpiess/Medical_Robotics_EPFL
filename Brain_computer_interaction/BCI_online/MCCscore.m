function [MCC, TNR] = MCCscore(predict,labels)
    % computes class error, to use with uniform prior
    
    TP = sum(predict(labels==1));
    FP = sum(predict(labels==0));
    FN = length(predict(labels==1)) - sum(predict(labels==1));
    TN = length(predict(labels==0)) - sum(predict(labels==0));
    
    
    MCC = (TP*TN-FP*FN)/sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN));
    TNR = TN/(TN+FP);

end