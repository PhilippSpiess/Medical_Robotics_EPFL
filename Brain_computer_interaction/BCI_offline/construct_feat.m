function [correct_feat error_feat] = construct_feat(correct_epoch, error_epoch)

    error_samp = downsample(error_epoch,8);
    correct_samp = downsample(correct_epoch,8);

    channels_cca = [1 2 3 4];
    
    error_feat = [];
    for i=1:size(error_epoch,3)
        elec = [];
        for j=channels_cca
            elec = cat(1, elec, error_samp(:,j,i));
            %HERE TRY NEW FEATURES
            %elec = cat(1, elec, new);
        end
        error_feat = cat(2, error_feat, elec);
    end
    error_feat = cat(1, zeros(1,size(error_epoch,3)), error_feat);

    
    correct_feat = [];
    for i=1:size(correct_epoch,3)
        elec = [];
        for j=channels_cca
            elec = cat(1, elec, correct_samp(:,j,i));
            %HERE TRY NEW FEATURES
            %elec = cat(1, elec, new);
        end
        correct_feat = cat(2, correct_feat, elec);
    end
    correct_feat = cat(1, ones(1,size(correct_epoch,3)), correct_feat);

    
end
