function eeg = filter_eeg(eeg)

    %spectral filter
    [b, a]=butter(2, 10/(512/2),'low');
    eeg = filtfilt(b ,a, eeg);
    [b, a]=butter(2, 1/(512/2),'high');
    eeg = filtfilt(b ,a, eeg);

    %spatial filter: CAR (not useful with CCA)
%     means=mean(eeg,2);
%     eeg(:,1) = eeg(:,1)-means;
%     eeg(:,3) = eeg(:,3)-means;
%     eeg(:,4) = eeg(:,4)-means;
%     eeg(:,5) = eeg(:,5)-means;
%     eeg(:,8) = eeg(:,8)-means;
%     eeg(:,9) = eeg(:,9)-means;
%     eeg(:,10) = eeg(:,10)-means;
    
end
