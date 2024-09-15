function [rejectnull, ...
          hyps_A_XY_vec, wins, jjj, ...
          nlmlA_XY_multiruns, ...
          structuralchanges, ...
          structuralchanges_chi2cdf_vec] = train_gp_and_do_test_structural(fromXY, toXY, sideXY,...
                run_ii, consider_win_point, wins, jjj, ...  
                Data, lag, do_test_using_fitted_model,...                
                mean_paramA_nr, cov_param_nrA,...
                paramrangesA, grid_nr, step_nr, requested_nr,...
                meanfunc, covfunc, likfunc, hypA_lik, inffunc, ref_date, ...
                data_cut_init_time, rejectnull, hyps_A_XY_vec, ...
                mean_predA_XY_vec, nlmlA_XY_multiruns, ...
                structuralchanges, structuralchanges_chi2cdf_vec)
                  
            [input_to_lagged, input_from_lagged, input_side_lagged] = compute_lags(Data,...
                                    toXY, fromXY, sideXY, lag);
            inputA_XY = [input_to_lagged, input_side_lagged];
            target_XY = Data(lag+1:end, toXY);
            inputA_XY_test = Data(end-lag+1:end,[toXY, sideXY])';
           
            if do_test_using_fitted_model == 0
                % fit GP
                %%%%%%%%%%%%%%%%%%%% optimise parameters: %%%%%%%%%%%%%%%%%%%%%%%%%%%%
                %%%%%%% many starting points:
                [hypA_XY_param_out, nA_XY_out] = get_starting_point_less_points_seed_version4NLP(mean_paramA_nr, cov_param_nrA,...
                    paramrangesA, ...
                    grid_nr,requested_nr, inputA_XY, target_XY,...
                    meanfunc, covfunc, likfunc, hypA_lik, inffunc, -10*run_ii);                
                meanA_nums = [1:mean_paramA_nr];
                covA_nums  = [mean_paramA_nr+1:mean_paramA_nr+cov_param_nrA]; 
                
                %%%%%%% try optimising the hyperparameters
                disp("Start fitting...")
                nA_XY_best = nA_XY_out(1);
                % add the lik parameter:
                hypA_XY_best.lik  = hypA_lik;
                hypA_XY_best.mean = hypA_XY_param_out(meanA_nums,1); 
                hypA_XY_best.cov  = hypA_XY_param_out(covA_nums,1);    
                hypA_XY_opt.lik   = hypA_lik;    
        
                for h_ii = 1:requested_nr        
                    hypA_XY_opt.mean = hypA_XY_param_out(meanA_nums, h_ii);   
                    hypA_XY_opt.cov = hypA_XY_param_out(covA_nums, h_ii);
                    [hypA_XY_opt,fA_XY,iA_XY,nA_XY] = minimize_modified(hypA_XY_opt, @gp, step_nr, ...
                                                                        inffunc, meanfunc, covfunc, ...
                                                                        likfunc, inputA_XY, target_XY);        
                    nlmlA_XY = gp(hypA_XY_opt, inffunc, meanfunc, ...
                                    covfunc, likfunc, inputA_XY, target_XY);        
                    if nlmlA_XY < nA_XY_best
                        hypA_XY_best = hypA_XY_opt; 
                        nA_XY_best   = nlmlA_XY;
                    end
                end    
                %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                % save hyperparameters
                hyps_A_XY_vec(jjj,:) = [hypA_XY_best.mean', hypA_XY_best.cov'];
                hyps_A_XY_vec(1:jjj,:);
                hypA_XY_best.mean;
                hypA_XY_best.cov;
        
                %%%%%%%%% for X --> Y | Z    
                % first get the additional predictive means:
                mean_predA_XY_vec(jjj) = gp(hypA_XY_best, inffunc, meanfunc, covfunc, likfunc, inputA_XY, target_XY, inputA_XY_test);    
                % then the usual negative log marginal likelihoods:
                nlmlA_XY_multiruns(jjj) = gp(hypA_XY_best, inffunc, meanfunc, covfunc, likfunc, inputA_XY, target_XY);
                nlmlA_XY_multiruns(1:jjj);
                if run_ii == 1
                    rejectnull(1,:) = {datestr(datetime(ref_date), "yyyy-mm-dd HH:00:00"), datestr(data_cut_init_time(run_ii, 1), ...
                                                "yyyy-mm-dd HH:00:00"), datestr(data_cut_init_time(run_ii, 2), "yyyy-mm-dd HH:00:00"), ... 
                                                        [], []};
                end
                %     hypA_XY_test.lik = hypA_XY_best.lik;
                %     hypA_XY_test.mean = hypA_XY_best.mean;
                %     hypA_XY_test.cov = [log(1), log(0.1), log(0.1)];
                %     [predictions] = ...
                %     sample_timeseries_gp_multistep(lag, [], [], hypA_XY_test, ...
                %         meanfunc, covfunc, length(inputA_XY));    
                %     sample_timeseries_gp_multistep(lag, inputA_XY, target_XY, 
                %         hypA_XY_best, meanfunc, covfunc, length(inputA_XY));
                
                if (run_ii > 1) && (run_ii >= consider_win_point)
                    % nlml is negative log likelihood, so need to put "-" in front    
                    % eval likelihood of current data with pre-report model                    
                    hypA_XY_first.mean = hyps_A_XY_vec(1, 1:lag)';    
                    hypA_XY_first.cov = hyps_A_XY_vec(1, lag+1:end)';                                     
                    hypA_XY_first.lik = nlmlA_XY_multiruns(1); 
                    M1_lik_current_data = gp(hypA_XY_first, inffunc, meanfunc, covfunc, likfunc, inputA_XY, target_XY);
                    % data window before (not including) the report date at position 1
                    structuralchanges(wins) = -2*(nlmlA_XY_multiruns(wins) - M1_lik_current_data); 
                    degf = mean_paramA_nr + cov_param_nrA;
                    structuralchanges_chi2cdf_vec(wins) = chi2cdf(structuralchanges(wins), degf);           
                    % store p-value 1-structuralchanges_chi2cdf_vec
                    rejectnull(wins,:) = {datestr(datetime(ref_date), "yyyy-mm-dd HH:00:00"), datestr(data_cut_init_time(run_ii, 1), ...
                                                "yyyy-mm-dd HH:00:00"), datestr(data_cut_init_time(run_ii, 2), "yyyy-mm-dd HH:00:00"), ... 
                                                        1-structuralchanges_chi2cdf_vec(wins), structuralchanges(wins)};  
                    structuralchanges_chi2cdf_vec(wins) = max(0, structuralchanges_chi2cdf_vec(wins));
                    wins = wins + 1;            
                end
                jjj = jjj + 1;
            else
                if run_ii == 1
                    rejectnull(1,:) = {datestr(datetime(ref_date), "yyyy-mm-dd HH:00:00"), datestr(data_cut_init_time(run_ii, 1), ...
                                                "yyyy-mm-dd HH:00:00"), datestr(data_cut_init_time(run_ii, 2), "yyyy-mm-dd HH:00:00"), ... 
                                                        [], []};
                end
                if (run_ii > 1) && (run_ii >= consider_win_point)                    
                    hypA_XY_first.mean = hyps_A_XY_vec(1, 1:lag)';
                    hypA_XY_first.cov = hyps_A_XY_vec(1, lag+1:end)';                         
                    hypA_XY_first.lik = nlmlA_XY_multiruns(1); 
                    M1_lik_current_data = gp(hypA_XY_first, inffunc, meanfunc, covfunc, likfunc, inputA_XY, target_XY);
                    structuralchanges(wins) = -2*(nlmlA_XY_multiruns(wins) - M1_lik_current_data); 
                    degf = mean_paramA_nr + cov_param_nrA;
                    structuralchanges_chi2cdf_vec(wins) = chi2cdf(structuralchanges(wins), degf);
                    rejectnull(wins,:) = {datestr(datetime(ref_date), "yyyy-mm-dd"), datestr(data_cut_init_time(run_ii, 1), ...
                                            "yyyy-mm-dd"), datestr(data_cut_init_time(run_ii, 2), "yyyy-mm-dd"), ... 
                                                  [1-structuralchanges_chi2cdf_vec(wins)], [structuralchanges(wins)]};    
                    structuralchanges_chi2cdf_vec(wins) = max(0, structuralchanges_chi2cdf_vec(wins));
                    wins = wins + 1;            
                end
                jjj = jjj + 1;
            end
end