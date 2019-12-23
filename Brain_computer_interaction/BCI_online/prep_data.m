function data_out = prep_data(tag)
    % cleans the raw eeg data by applying filters, epoching and
    % concatenating all the data
    % tag: file names without extension = folder names
    % data_out: matrix where each rows corresponds to one sample and each
    % column to one feature. The first column corresponds to the labels.

    data_out = [];

    for i=1:numel(tag)
        
        %import data from trial
        [signal, header] = sload(strcat(char(tag(i)),'.gdf'));
        behavior = single(dlmread(strcat(char(tag(i)),'.txt')));
        
        eog = signal(:,17:19);
        eeg = all_filters(signal(:,1:16),eog);
        trig = signal(:,33);
        
        % epoching
        [norot_epoch, rot_epoch] = epoching(eeg, header, trig, behavior, [0.2,0.8]);
        
        % clean data
        all_feat = construct_feat(norot_epoch, rot_epoch);
        
        % concatenate all arrays
        data_out = cat(1,data_out,all_feat);

    end
    
end