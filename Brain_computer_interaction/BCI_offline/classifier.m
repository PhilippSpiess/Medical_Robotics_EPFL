function [TP, TN, FP, FN] = classifier(X,T,Y,yreal, explain_pourcentage)
    TP=0;
    TN=0;
    FP=0;
    FN=0;
    num_feat = 1;
    num_feat_fisher=125;
    X=X';
    Y=Y';
    
    %Fisher Score
     [Idx_fisher, Z] = rankfeatures(X', T');
     feat_fisher = Idx_fisher(1:num_feat_fisher);
%      bar(Z)
%      xlabel('features');
%      ylabel('Z value');
     %size(X)
     X=X(:,feat_fisher);
     Y=Y(:,feat_fisher);
    
    
    %PCA
    [coeff,score,explained] = pca(X);
    exp = 0;
    while exp < explain_pourcentage
        exp = exp + explained(num_feat);
        num_feat=num_feat+1;
    end
    X = X*coeff;
    X = X(:,1:num_feat);
    Y = Y*coeff;
    Y = Y(:,1:num_feat);
    
    classifier = fitcdiscr(X,T,'discrimType','quadratic');
    %classifier = fitcdiscr(X,T,'discrimType','linear');
    %classifier = fitcsvm(X,T);
    %classifier = fitcnb(X,T);
    ytest = predict(classifier,Y);

        %GRAPH OF PRINCIPAL COMPONENTS       
%         f = figure;
%         scatter(X(1:245,1), X(1:245,2), 'r');
%         hold on;
%         scatter(X(246:360,1), X(246:360,2), 'b');
%         xlabel('first dimension');
%         ylabel('second dimension');
%         title('Separation of the two first Principal components');

        %NEURAL NET 
%         net= patternnet(10);
%         net.trainparam.epochs=100;
%         net = train(net,X,T);
%         ytest = net(Y);
%         ytest = round(ytest);

    for j = 1:length(ytest)
        if yreal(j) == 1
            if ytest(j) == 1
                TN=TN+1;
            else
                FN=FN+1;
            end
        end
    end
    for l = 1:length(ytest)
        if yreal(l) == 0
            if ytest(l) == 0
                TP=TP+1;
            else
                FP=FP+1;
            end
        end
    end
end


