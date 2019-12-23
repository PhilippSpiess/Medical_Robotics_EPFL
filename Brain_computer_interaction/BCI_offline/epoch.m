function [correct_epoch error_epoch] = epoch(eeg, header, trig, behavior, timeperiod)

    sampleTrial = floor(timeperiod(1)*header.SampleRate:timeperiod(2)*header.SampleRate);

    onset = find(diff(trig)) + 1;
    onset = onset(trig(onset) == 3);

    onset_b = find(diff(behavior(:,2))) + 1;
    onset_b = onset_b(behavior(onset_b,2) == 3);
    mag = behavior(onset_b,3);

    corridx = find(mag==0);
    correct_eeg = onset(corridx);
    correct_epoch = [];

    for i = 1:length(correct_eeg)
        correct_epoch = cat(3, correct_epoch, eeg(correct_eeg(i) + sampleTrial , :));
    end
    
    %normal
    erridx = find(mag~=0);
    
    %for 20 degrees only:
    %erridx = find(abs(mag)<0.5 & mag~=0);
    
    %for 40 degrees only:
    %erridx = find(abs(mag)<1 & abs(mag)>0.5);
    
    %for 60 degrees only:
    %erridx = find(abs(mag)>1);
    
    %for 60 and 40 degrees:
    %erridx = find(abs(mag)>0.5);
    
    error_eeg = onset(erridx);
    error_epoch = [];

    for trial = 1:length(error_eeg)

        error_epoch = cat(3, error_epoch, eeg(error_eeg(trial) + sampleTrial , :));

    end
end
