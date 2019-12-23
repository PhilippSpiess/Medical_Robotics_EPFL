function Errp_detec = sliding_window(test_trial, delay, label, classifier, coeff_cca, coeff_pca, params)
% classifies one trial for each threshold value for posterioir probabilities and number of consecutive
% sample
% INPUT
% trial to classify, delay, index for trial, trained classifier, cca and
% pca coefficients, threshold and consec parameters
% OUTPUT
% Matrix of size n threshold values tested * m consecutive_sample values
% tested for the input trial. For each combination, value between1 and 4:
% 1:TP, 2:TN, 3:FP, 4:FN


thresh_vect = params.thresh_vect;
consec_vect = params.consec_vect;
nb_feat = params.nb_feat;

windsize = ceil(0.6*512);


% initiate prediction
Errp_detec = zeros(length(thresh_vect), length(consec_vect));
consec_detect = zeros(1,length(thresh_vect));

% initiate starting point of window
window_idx = 1;

% run through all data points for trial: window moves by 2
% points every iteration (64Hz downsampling and 32Hz data
% collection)

while window_idx <= size(test_trial,1)-windsize

    % set window
    window = test_trial(window_idx:window_idx+windsize-1,:);
    window = window*coeff_cca;
    window = downsample(window,8);

    sample = [];

    % get sample format for prediction
    for cca_feat=1:size(coeff_cca,2)
        sample = cat(1, sample, window(:,cca_feat));
    end
    sample = transpose(sample);


    %apply pca on sample data
    sample = sample*coeff_pca;
    sample = sample(:,1:nb_feat);


    [~, proba, ~] = predict(classifier, sample);
    
    
    detect = proba(2)>thresh_vect;
    consec_detect = (consec_detect + detect).*detect;
    
    for i=1:length(consec_detect)
        % run through all threshold values
        dunno = consec_detect(i)>=consec_vect;
        if window_idx < floor(delay/8)-floor(0.3*64)
            % FP case
            dunno = 3*dunno;
        end
        for j=1:length(consec_vect)
            % run through all consec values
            if Errp_detec(i,j) == 0
                % no decision made yet
                Errp_detec(i,j) = dunno(j);
            end
        end
    end
    
    % change window start
    window_idx = window_idx + 16;
    
end

if label == 0
    Errp_detec(Errp_detec==0)=2;
    Errp_detec(Errp_detec==1)=3;
else
    Errp_detec(Errp_detec==0)=4;
end
    

end