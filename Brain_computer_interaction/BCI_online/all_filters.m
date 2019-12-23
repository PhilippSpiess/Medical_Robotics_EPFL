function filtered_eeg = all_filters(raw_eeg, eog)
    % Applies EOG, CAR, Laplacian and Butterworth bandpass filters

    %EOG
    eog_coeff = filterEOG(raw_eeg, eog);
    eeg = raw_eeg - eog*eog_coeff;
    
    % CAR
    eeg = eeg-mean(eeg,2);
    
    %Laplacian
    %eeg = lapfilter(eeg);
    
    % Butterworth bandpass between 1 and 10 Hz
    fc1 = 1;
    fc2 = 10;
    %sampling freq
    fs = 512;
    [b10, a10] = butter(2,[fc1/(fs/2) fc2/(fs/2)],'bandpass');
    
    filtered_eeg = filter(b10,a10,eeg);

end