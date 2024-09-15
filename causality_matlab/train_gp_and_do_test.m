function [Causality_XY_vec, Causality_YX_vec, XY_chi2cdf_vec, YX_chi2cdf_vec,...
          hyps_A_XY_vec, hyps_B_XY_vec, hyps_A_YX_vec, hyps_B_YX_vec,...
          mean_predA_XY_vec, mean_predB_XY_vec, nlmlA_XY_multiruns, nlmlB_XY_multiruns,...
          mean_predA_YX_vec, mean_predB_YX_vec, nlmlA_YX_multiruns, nlmlB_YX_multiruns,...
          pvalueXYGranger_ii, pvalueYXGranger_ii] = train_gp_and_do_test(fromXY, toXY, sideXY,...
                fromYX, toYX, sideYX,...
                Data, lag, mean_paramA_nr, cov_param_nrA,...
                paramrangesA, grid_nr, requested_nr, meanfunc, covfunc, likfunc, hypA_lik, inffunc,...
                run_ii, mean_paramB_nr, cov_param_nrB,...
                paramrangesB, hypB_lik, step_nr, causal_placement_switch, floor_n)
                
    % will want to test X --> Y | Z and Y --> X | Z
    %%%%%%%%% for X --> Y | Z
    % inputA_XY = Data(1:end-lag,[toXY, sideXY]); %remember: nonnested here
    % inputB_XY = Data(1:end-lag,[fromXY, toXY, sideXY]);
    % target_XY = Data(lag+1:end,toXY);

    %%%%
    % A XY, X --> Y | Z
    [input_to_lagged, input_from_lagged, input_side_lagged] = ...
                                        compute_lags(Data, toXY, fromXY, sideXY, lag);
    inputA_XY = [input_to_lagged, input_side_lagged];
    inputB_XY = [input_from_lagged, input_to_lagged, input_side_lagged];
    target_XY = Data(lag+1:end,toXY);
    
    % B YX, Y --> X | Z
    [input_to_lagged_yx, input_from_lagged_yx, input_side_lagged_yx] = ...
                                        compute_lags(Data, toYX, fromYX, sideYX, lag);
    inputA_YX = [input_to_lagged_yx, input_side_lagged_yx];
    inputB_YX = [input_from_lagged_yx, input_to_lagged_yx, input_side_lagged_yx];
    target_YX = Data(lag+1:end,toYX);

    inputA_XY_test = Data(end-lag+1:end,[toXY, sideXY])';
    inputA_XY_test = inputA_XY_test(:)';
    inputB_XY_test = Data(end-lag+1:end,[fromXY, toXY, sideXY]);
    inputB_XY_test = inputB_XY_test(:)'; 
    inputA_YX_test = Data(end-lag+1:end,[toYX, sideYX])'; 
    inputA_YX_test = inputA_YX_test(:)';
    inputB_YX_test = Data(end-lag+1:end,[fromYX, toYX, sideYX]);
    inputB_YX_test = inputB_YX_test(:)'; 
    %%%%

    
    %%%%%%%%% for Y --> X | Z
    % inputA_YX = Data(1:end-lag,[toYX, sideYX]); %remember: nonnested here
    % inputB_YX = Data(1:end-lag,[fromYX, toYX, sideYX]);
    % target_YX = Data(lag+1:end,toYX);
    % 
    
    % inputA_XY_test = Data(end-lag+1,[toXY, sideXY]);
    % inputB_XY_test = Data(end-lag+1,[fromXY, toXY, sideXY]);
    % inputA_YX_test = Data(end-lag+1,[toYX, sideYX]); 
    % inputB_YX_test = Data(end-lag+1,[fromYX, toYX, sideYX]);

    
    % Granger
    if ~strcmp(causal_placement_switch,'mean')
        if isempty(sideXY)
            [hXY,pvalueXYGranger_ii,statXY,cvalueXY] = gctest(Data(:, fromXY), Data(:, toXY), NumLags=lag);
            [hYX,pvalueYXGranger_ii,statYX,cvalueYX] = gctest(Data(:, fromYX), Data(:, toYX), NumLags=lag);
        else
            [hXY,pvalueXYGranger_ii,statXY,cvalueXY] = gctest(Data(:, fromXY), Data(:, toXY), Data(:, sideXY), NumLags=lag);
            [hYX,pvalueYXGranger_ii,statYX,cvalueYX] = gctest(Data(:, fromYX), Data(:, toYX), Data(:, sideYX), NumLags=lag);
        end
    else
        hXY = NaN;
        pvalueXYGranger_ii = NaN;
        statXY = NaN;
        cvalueXY = NaN;
        hYX = NaN;
        pvalueYXGranger_ii = NaN;
        statYX = NaN;
        cvalueYX = NaN;
    end


    %%%%%%%%%%%%%%%%%%%% optimise parameters: %%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%% many starting points:
    [hypA_XY_param_out, nA_XY_out] = get_starting_point_less_points_seed_version4NLP(mean_paramA_nr, cov_param_nrA,...
        paramrangesA,...
        grid_nr,requested_nr, inputA_XY, target_XY,...
        meanfunc, covfunc, likfunc, hypA_lik, inffunc, -10*run_ii);
    [hypB_XY_param_out, nB_XY_out] = get_starting_point_less_points_seed_version4NLP(mean_paramB_nr, cov_param_nrB,...
        paramrangesB,...
        grid_nr,requested_nr, inputB_XY, target_XY,...
        meanfunc, covfunc, likfunc, hypB_lik, inffunc, -10*run_ii);  
    [hypA_YX_param_out, nA_YX_out] = get_starting_point_less_points_seed_version4NLP(mean_paramA_nr, cov_param_nrA,...
        paramrangesA,...
        grid_nr,requested_nr, inputA_YX, target_YX,...
        meanfunc, covfunc, likfunc, hypA_lik, inffunc, -10*run_ii);
    [hypB_YX_param_out, nB_YX_out] = get_starting_point_less_points_seed_version4NLP(mean_paramB_nr, cov_param_nrB,...
        paramrangesB,...
        grid_nr,requested_nr, inputB_YX, target_YX,...
        meanfunc, covfunc, likfunc, hypB_lik, inffunc, -10*run_ii);   
    meanA_nums = [1:mean_paramA_nr]; 
    covA_nums  = [mean_paramA_nr+1:mean_paramA_nr+cov_param_nrA]; 
    meanB_nums = [1:mean_paramB_nr];  
    covB_nums  = [mean_paramB_nr+1:mean_paramB_nr+cov_param_nrB];

    %%%%%%% try optimising the hyperparameters
    nA_XY_best = nA_XY_out(1);
    nB_XY_best = nB_XY_out(1);
    nA_YX_best = nA_YX_out(1);
    nB_YX_best = nB_YX_out(1);

    % add the lik parameter:
    hypA_XY_best.lik =  hypA_lik;
    hypB_XY_best.lik =  hypA_lik;
    hypA_YX_best.lik =  hypA_lik;
    hypB_YX_best.lik =  hypA_lik;

    hypA_XY_best.mean=hypA_XY_param_out(meanA_nums,1); hypA_XY_best.cov=hypA_XY_param_out(covA_nums,1);
    hypB_XY_best.mean=hypB_XY_param_out(meanB_nums,1); hypB_XY_best.cov=hypB_XY_param_out(covB_nums,1);

    hypA_YX_best.mean=hypA_YX_param_out(meanA_nums,1); hypA_YX_best.cov=hypA_YX_param_out(covA_nums,1);
    hypB_YX_best.mean=hypB_YX_param_out(meanB_nums,1); hypB_YX_best.cov=hypB_YX_param_out(covB_nums,1); 

    hypA_XY_opt.lik =  hypB_lik;
    hypB_XY_opt.lik =  hypB_lik;
    hypA_YX_opt.lik =  hypB_lik;
    hypB_YX_opt.lik =  hypB_lik;
    for h_ii = 1:requested_nr

        hypA_XY_opt.mean=hypA_XY_param_out(meanA_nums,h_ii); hypA_XY_opt.cov=hypA_XY_param_out(covA_nums,h_ii);
        hypB_XY_opt.mean=hypB_XY_param_out(meanB_nums,h_ii); hypB_XY_opt.cov=hypB_XY_param_out(covB_nums,h_ii);

        hypA_YX_opt.mean=hypA_YX_param_out(meanA_nums,h_ii); hypA_YX_opt.cov=hypA_YX_param_out(covA_nums,h_ii);
        hypB_YX_opt.mean=hypB_YX_param_out(meanB_nums,h_ii); hypB_YX_opt.cov=hypB_YX_param_out(covB_nums,h_ii);

        [hypA_XY_opt,fA_XY,iA_XY,nA_XY] = minimize_modified(hypA_XY_opt, @gp, step_nr, ...
           inffunc, meanfunc, covfunc, likfunc, inputA_XY, target_XY);
        [hypB_XY_opt,fB_XY,iB_XY,nB_XY] = minimize_modified(hypB_XY_opt, @gp, step_nr, ...
           inffunc, meanfunc, covfunc, likfunc, inputB_XY, target_XY);
        [hypA_YX_opt,fA_YX,iA_YX,nA_YX] = minimize_modified(hypA_YX_opt, @gp, step_nr, ...
           inffunc, meanfunc, covfunc, likfunc, inputA_YX, target_YX);
        [hypB_YX_opt,fB_YX,iB_YX,nB_YX] = minimize_modified(hypB_YX_opt, @gp, step_nr, ...
           inffunc, meanfunc, covfunc, likfunc, inputB_YX, target_YX);

       nlmlA_XY = gp(hypA_XY_opt, inffunc, meanfunc, ...
        covfunc, likfunc, inputA_XY, target_XY);
       nlmlB_XY = gp(hypB_XY_opt, inffunc, meanfunc, ...
            covfunc, likfunc, inputB_XY, target_XY);
       nlmlA_YX = gp(hypA_YX_opt, inffunc, meanfunc, ...
            covfunc, likfunc, inputA_YX, target_YX);
       nlmlB_YX = gp(hypB_YX_opt, inffunc, meanfunc, ...
        covfunc, likfunc, inputB_YX, target_YX);
       % now replace the best ones:
       if nlmlA_XY<nA_XY_best
           hypA_XY_best = hypA_XY_opt; nA_XY_best = nlmlA_XY;
       end
       if nlmlB_XY<nB_XY_best
           hypB_XY_best = hypB_XY_opt; nB_XY_best = nlmlB_XY;
       end
       if nlmlA_YX<nA_YX_best
           hypA_YX_best = hypA_YX_opt; nA_YX_best = nlmlA_YX;
       end
       if nlmlB_YX<nB_YX_best
           hypB_YX_best = hypB_YX_opt; nB_YX_best = nlmlB_YX;
       end
    end
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    % save hyperparameters
    hyps_A_XY_vec = [hypA_XY_best.mean', hypA_XY_best.cov'];
    hyps_B_XY_vec = [hypB_XY_best.mean', hypB_XY_best.cov'];
    hyps_A_YX_vec = [hypA_YX_best.mean', hypA_YX_best.cov'];
    hyps_B_YX_vec = [hypB_YX_best.mean', hypB_YX_best.cov'];

    
    %%%%%%%%% for X --> Y | Z    
    % first get the additional predictive means:
    mean_predA_XY_vec = gp(hypA_XY_best, inffunc, meanfunc, covfunc, likfunc, inputA_XY, target_XY, inputA_XY_test);
    mean_predB_XY_vec = gp(hypB_XY_best, inffunc, meanfunc, covfunc, likfunc, inputB_XY, target_XY, inputB_XY_test);

    % then the usual negative log marginal likelihoods:
    nlmlA_XY_multiruns = gp(hypA_XY_best, inffunc, meanfunc, covfunc, likfunc, inputA_XY, target_XY);
    nlmlB_XY_multiruns = gp(hypB_XY_best, inffunc, meanfunc, covfunc, likfunc, inputB_XY, target_XY);

    % nlml is negative log likelihood, so need to put "-" in front
    Causality_XY_vec = - 2*(nlmlB_XY_multiruns - nlmlA_XY_multiruns);
    
    %%%%%%%%% for Y --> X | Z
    % first get the additional predictive means:
    mean_predA_YX_vec = gp(hypA_YX_best, inffunc, meanfunc, covfunc, likfunc, inputA_YX, target_YX, inputA_YX_test);
    mean_predB_YX_vec = gp(hypB_YX_best, inffunc, meanfunc, covfunc, likfunc, inputB_YX, target_YX, inputB_YX_test);

    % then the usual negative log marginal likelihoods:
    nlmlA_YX_multiruns = gp(hypA_YX_best, inffunc, meanfunc, covfunc, likfunc, inputA_YX, target_YX);
    nlmlB_YX_multiruns = gp(hypB_YX_best, inffunc, meanfunc, covfunc, likfunc, inputB_YX, target_YX);

    % nlml is negative log likelihood, so need to put "-" in front
    Causality_YX_vec = - 2*(nlmlB_YX_multiruns - nlmlA_YX_multiruns);

    if strcmp(causal_placement_switch,'mean') 
        XY_chi2cdf_vec = chi2cdf(Causality_XY_vec,1);
        YX_chi2cdf_vec = chi2cdf(Causality_YX_vec,1);
    else
        XY_chi2cdf_vec = chi2cdf(Causality_XY_vec,2);
        YX_chi2cdf_vec = chi2cdf(Causality_YX_vec,2);
    end
    
    XY_chi2cdf_vec = max(floor_n, XY_chi2cdf_vec);
    YX_chi2cdf_vec = max(floor_n, YX_chi2cdf_vec);

end