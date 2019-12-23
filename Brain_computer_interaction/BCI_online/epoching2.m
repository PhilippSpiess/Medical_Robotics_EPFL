function [epoch_data, delay] = epoching2(eeg, trig, behavior)
% epoching of eeg data and classification in samples with rotation (error) or
% without rotation (correct)
    too_early = 512*0.25;
    
    onset = find(diff(trig)) + 1;
    start_t = onset(trig(onset) == 1)+0.25*512;
    end_t = onset(trig(onset) == 0);
    on_rot = onset(trig(onset) == 3);
    
    onset_b = find(diff(behavior(:,2))) + 1;
    rot_b = onset_b(behavior(onset_b,2) == 3);
    %end_b = onset_b(behavior(onset_b,2) == 7);
    mag = behavior(rot_b,3);


    norotidx = find(mag==0);
    norot_st = start_t(norotidx);
    norot_end = end_t(norotidx);
    norot_epoch = {};

    for trial = 1:length(norot_st)
        
        %norot_epoch = cat(3, norot_epoch, eeg(norot_st(trial):norot_end(trial), :));
        norot_epoch(trial) = {eeg(norot_st(trial)+too_early:norot_end(trial), :)};
    end

    rotidx = find(mag~=0);
    rot_st = start_t(rotidx);
    rot_end = end_t(rotidx);
    rot_epoch = {};

    for trial = 1:length(rot_st)

        %rot_epoch = cat(3, rot_epoch, eeg(rot_st(trial):rot_end(trial), :));
        rot_epoch(trial) = {eeg(rot_st(trial)+too_early:rot_end(trial), :)};

    end
    
    delay = cat(1,on_rot(norotidx) - too_early - norot_st,on_rot(rotidx) - too_early - rot_st);
        
    delay(:,2) = cat(1,zeros(length(norot_epoch),1),ones(length(rot_epoch),1));
        
    delay(:,3) = end_t - start_t;
    delay(:,4) = delay(:,3) - delay(:,1);
     
    epoch_data = [norot_epoch rot_epoch];
    
end
