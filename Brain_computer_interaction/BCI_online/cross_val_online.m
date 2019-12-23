function results_tab = cross_val_online(rolling_data, header, delay_data, nbfeat)
    
    % number of threshold values tested
    thresnb = 5;
    
    % Prepare results matrix
    results_tab = zeros(thresnb,11);
    results_tab(:,1) = transpose(linspace(0.85,0.99,thresnb));
    
    % determine size of rolling window
    windsize = ceil(0.6*64);
    
    partition = cvpartition(delay_data(:,2),'kfold',10);
    
    % Where we expect to find ErrP
    sampleTrial = floor(0.2*header.SampleRate:0.8*header.SampleRate);
    
    % electrodes selected
    channels_list = [1 3 4 5 8 9 10];

    
    row = 0;
    
    for thresh = linspace(0.85,0.99,thresnb)
        
        row = row + 1;
        
        for fold = 1:partition.NumTestSets
            % run through folds and determine MCC via TP,TN,FP,FN counts
            
            TP = 0;
            TN = 0;
            FP = 0;
            FN = 0;
            
            index = [1:partition.NumObservations];
            tr_index = index(partition.training(fold));
            te_index = index(partition.test(fold));

            %data_train = epoch_data{1,index(partition.training(fold))};
            %data_test = epoch_data{1,index(partition.test(fold))};
            
            % delay between start of trial and rotation for training data
            tr_delay = delay_data(partition.training(fold),1);

            % training data of size 360 trials by 273 features
            data_tr = zeros(length(tr_delay),ceil(length(sampleTrial)/8)*length(channels_list));
            
            for tr_trial = 1:length(tr_index)
                tr_sample = rolling_data{1,tr_index(tr_trial)};
                tr_sample = tr_sample(tr_delay(tr_trial)+sampleTrial,channels_list);
                
                % downsample to obtain tr_sample of size 39 features by 7
                % electrodes
                tr_sample = transpose(downsample(tr_sample,8));
                
                for elec_idx = 1:size(tr_sample,1)
                    st = (elec_idx-1)*size(tr_sample,2)+1;
                    data_tr(tr_trial,st:st+size(tr_sample,2)-1) = tr_sample(elec_idx,:);
                end
            end
            
            % compute pca coeffs on training data
            [coeff_pca, score, ~] = pca(data_tr);
            tr_feat = score(:,1:nbfeat);
            
            % get delays and labels for test and train data
            tr_labels = delay_data(partition.training(fold),2);
            te_labels = delay_data(partition.test(fold),2);
            te_delay = delay_data(partition.test(fold),1);
            
            % train classifier
            classifier = fitcdiscr(tr_feat, tr_labels, 'discrimType','diaglinear');
            
            for trial = 1:length(te_index)

                test_trial = rolling_data{1,te_index(trial)};
                test_trial = downsample(test_trial,8);

                % initiate prediction
                output = 0;
                label = te_labels(trial);
                trial_delay = te_delay(trial);
                
                % initiate starting point of window
                window_idx=1;
                
                % run through all data points for trial: window moves by 2
                % points every iteration (64Hz downsampling and 32Hz data
                % collection)
                while window_idx <= size(test_trial,1)-windsize
                    
                    % set window
                    window = test_trial(window_idx:window_idx+windsize-1,:);
                    sample = [];

                    
                    % get sample format for prediction
                    for elec=channels_list
                        sample = cat(1, sample, window(:,elec));
                    end
                    sample = transpose(sample);
                    
                    
                    %apply pca on sample data
                    sample = sample*coeff_pca;
                    sample = sample(:,1:nbfeat);

                    
                    [~, proba, ~] = predict(classifier, sample);
                    
                    % if ErrP detected
                    if proba(2) > thresh
                        
                        output = 1;
                        
                        if window_idx < floor(trial_delay/8)-floor(0.3*64)
                        % case where detection too early, i.e 0.3 sec before rotation: ErrP not possible
                            FP = FP + 1;
                        
                        else
                            if label == 0
                                FP = FP+1;
                            else
                                TP = TP+1;
                            end
                        end
                        % break loop
                        window_idx = window_idx + 1000;
                    end
                    
                    % change window start
                    window_idx = window_idx + 2;
                end
                if output == 0
                    if label == 0
                        TN = TN + 1;
                    else
                        FN = FN + 1;
                    end
                end
            end

            % compute MCC for each fold
            results_tab(row,fold+1) = (TP*TN-FP*FN)/sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN));
            %results_tab(row,fold+1) = FP;
        end
    end

end
