function all_feat = construct_feat(norot_epoch, rot_epoch)
    % generates array from filtered and epoched eeg data
    
    % downsample to 64 Hz
    rot_samp = downsample(rot_epoch,8);
    norot_samp = downsample(norot_epoch,8);
    
    % relevant channels to use
    channels_list = [1 3 4 5 8 9 10];

    
    rot_feat = [];
    for i=1:size(rot_epoch,3)
        elec = [];
        % iterate through all intersting electrodes
        for j=channels_list
            elec = cat(1, elec, rot_samp(:,j,i));
        end
        rot_feat = cat(2, rot_feat, elec);
    end

    rot_feat = cat(1, ones(1,size(rot_epoch,3)), rot_feat);


    
    norot_feat = [];
    for i=1:size(norot_epoch,3)
        elec = [];
        % iterate through all interesting electrodes
        for j=channels_list
            elec = cat(1, elec, norot_samp(:,j,i));
        end
        norot_feat = cat(2, norot_feat, elec);
    end
    norot_feat = cat(1, zeros(1,size(norot_epoch,3)), norot_feat);
    
    all_feat = transpose(cat(2, norot_feat, rot_feat));
    
end