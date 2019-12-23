function TPTNFPFN = online_perf(rolling_data,idx_comp, classifier, coeff_cca, coeff_pca, params)

TPTNFPFN = zeros(length(params.thresh_vect), length(params.consec_vect), 4);

for trial = 1:length(idx_comp.te_index)

        test_trial = rolling_data{1,idx_comp.te_index(trial)};
        label = idx_comp.te_labels(trial);
        delay = idx_comp.te_delay(trial);

        Errp_detec = sliding_window(test_trial, delay, label, classifier, coeff_cca, coeff_pca, params);
        
        for verdict = 1:4
            to_add = Errp_detec==verdict;
            TPTNFPFN(:,:,verdict) = TPTNFPFN(:,:,verdict) + to_add;
        end
end


end