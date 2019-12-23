function [norot_epoch, rot_epoch] = epoching(eeg, header, trig, behavior, timeperiod)
% epoching of eeg data and classification in samples with rotation (error) or
% without rotation (correct)

    % The sample corresponds to the time region of interest
    sampleTrial = floor(timeperiod(1)*header.SampleRate:timeperiod(2)*header.SampleRate);


    onset = find(diff(trig)) + 1;
    onset = onset(trig(onset) == 3);


    onset_b = find(diff(behavior(:,2))) + 1;
    onset_b = onset_b(behavior(onset_b,2) == 3);
    mag = behavior(onset_b,3);

    norotidx = find(mag==0);
    norot_eeg = onset(norotidx);
    norot_epoch = [];

    for trial = 1:length(norot_eeg)
        
        norot_epoch = cat(3, norot_epoch, eeg(norot_eeg(trial) + sampleTrial , :));
    end

    rotidx = find(mag~=0);
    rot_eeg = onset(rotidx);
    rot_epoch = [];

    for trial = 1:length(rot_eeg)

        rot_epoch = cat(3, rot_epoch, eeg(rot_eeg(trial) + sampleTrial , :));
    end
end
