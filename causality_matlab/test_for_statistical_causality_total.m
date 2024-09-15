function [] = ...
        test_for_statistical_causality_total(all_data, from_loop, to_loop,...
                                                meanf, likf, name, lag, floor_switch, covf, ...
                                                    dir_out, data_cut_init_time, causal_placement_switch)

time_now = now;
starting_time = floor(1.e+06 * rem(time_now,1));
starting_time_name = num2str(starting_time);

switch floor_switch
    case 'true'
        floor_n = 0;
    case 1
        floor_n = 0;
    case 0
        floor_n = -Inf;
    case 'false'
        floor_n = -Inf;
end
length_data = size(all_data,1);

%% 2. specify the model - here Matern kernel, additive Gaussian iid noise
% remember -- to have nested models we need ARD kernel!
% and also we need the fixed version of Matern

switch covf
    case 'Matern'
        covfunc = {@covSum, {{@covMaternard_modified,3}, @covNoise}};
    case {'poly', 'poly-cov'}
        covfunc = {@covSum, {{@covPoly,2}, @covNoise}};
    otherwise
        covfunc = {@covSum, {{@covMaternard_modified,3}, @covNoise}};
end

switch meanf
    case {'lin', 'linear'}
        meanfunc = {@meanLinear};
    case {'poly', 'polynomial'}
        meanfunc = {@meanPoly,2};
    otherwise
        meanfunc = {@meanLinear};
end
    
switch likf
    case {'Gauss', 'Gaus', 'Gaussian', 'likGauss'}
        likfunc  = {@likGauss}; 
        inffunc = @infExact;
        hypA.lik  = log(0); hypB.lik  = log(0);
    case {'t-student', 'student-t', 't', 'likT'}
        likfunc = {@likT};
        inffunc = @infLaplace;
        hypA.lik  = [log(0); log(1)]; hypB.lik  = [log(0); log(1)]; % for starting points...

    otherwise
        likfunc  = {@likGauss}; 
        inffunc = @infExact;
        hypA.lik  = log(0); hypB.lik  = log(0);
end
    

%% run many tests to compare!
multirun_number = 100;
if from_loop == 1
    multirun_number = to_loop
end
step_nr = -30; % in optimisation
requested_nr = 5; % how many best starting points
grid_nr = 5;
if lag == 1
    grid_nr = 5; % grid for the starting points
elseif lag < 5
    % reduce compute cost
    grid_nr = 2;
elseif lag <= 8
    % reduce compute cost
    grid_nr = 1.9;    
else
    grid_nr = 1.45;
end

Causality_XY_vec_pricesent = zeros(multirun_number,1);
Causality_YX_vec_pricesent = zeros(multirun_number,1);
XY_chi2cdf_vec_pricesent = zeros(multirun_number,1);
YX_chi2cdf_vec_pricesent = zeros(multirun_number,1);
Causality_XY_vec_Granger_pricesent = zeros(multirun_number,1);
Causality_YX_vec_Granger_pricesent = zeros(multirun_number,1);

Causality_XY_vec_priceretsent = zeros(multirun_number,1);
Causality_YX_vec_priceretsent = zeros(multirun_number,1);
XY_chi2cdf_vec_priceretsent = zeros(multirun_number,1);
YX_chi2cdf_vec_priceretsent = zeros(multirun_number,1);
Causality_XY_vec_Granger_priceretsent = zeros(multirun_number,1);
Causality_YX_vec_Granger_priceretsent = zeros(multirun_number,1);

Causality_XY_vec_volsent = zeros(multirun_number,1);
Causality_YX_vec_volsent = zeros(multirun_number,1);
XY_chi2cdf_vec_volsent = zeros(multirun_number,1);
YX_chi2cdf_vec_volsent = zeros(multirun_number,1);
Causality_XY_vec_Granger_volsent = zeros(multirun_number,1);
Causality_YX_vec_Granger_volsent = zeros(multirun_number,1);

Causality_XY_vec_realvolsent = zeros(multirun_number,1);
Causality_YX_vec_realvolsent = zeros(multirun_number,1);
XY_chi2cdf_vec_realvolsent = zeros(multirun_number,1);
YX_chi2cdf_vec_realvolsent = zeros(multirun_number,1);
Causality_XY_vec_Granger_realvolsent = zeros(multirun_number,1);
Causality_YX_vec_Granger_realvolsent = zeros(multirun_number,1);

Causality_XY_vec_volrealvolsent = zeros(multirun_number,1);
Causality_YX_vec_volrealvolsent = zeros(multirun_number,1);
XY_chi2cdf_vec_volrealvolsent = zeros(multirun_number,1);
YX_chi2cdf_vec_volrealvolsent = zeros(multirun_number,1);
Causality_XY_vec_Granger_volrealvolsent = zeros(multirun_number,1);
Causality_YX_vec_Granger_volrealvolsent = zeros(multirun_number,1);


%%%%%%% many starting points:
if strcmp(char(meanfunc{1}),'meanLinear') 
    % mean_paramA_nr = 1; mean_paramB_nr = 2;
    % mean_rangesA = [-1,1]; mean_rangesB = [-1,1;-1,1];
    mean_paramA_nr = lag; 
    mean_rangesA = repmat([-1, 1],lag,1);
    mean_paramB_nr = 2*lag;
    mean_rangesB = repmat([-1, 1],2*lag,1);
else %assume poly
    disp("Modify code to support polynomial mean.");
    return
    mean_paramA_nr = 2; mean_paramB_nr = 4;
    mean_rangesA = [-1,1;-1,1]; mean_rangesB = [-1,1;-1,1;-1,1;-1,1];
end

if strcmp(char(covfunc{2}{1}{1}),'covPoly') 
    disp("Modify code to support polynomial kernel.");
    return
    cov_param_nrA = 3; 
    cov_param_nrB = 3; % c, sf, sn
    cov_rangesA = [-10,0;-10,-2;-4,-1];
    cov_rangesB = [-10,0;-10,-2;-4,-1];
else % assume covMaternard_modified
    % cov_param_nrA = 3; cov_param_nrB = 4;
    % cov_rangesA = [-10,0;-10,-2;-4,-1];
    % cov_rangesB = [-10,0;-10,0;-10,-2;-4,-1];
    cov_param_nrA = 2 + lag;
    cov_rangesA = [repmat([-10, 0],lag,1); -10,-2; -4,-1];
    cov_param_nrB = 2 + 2*lag; % an extra set of lags for the extra covariate
    cov_rangesB = [repmat([-10, 0],2*lag,1);-10,-2;-4,-1];
end

if strcmp(causal_placement_switch,'mean') %now cater only for causality in mean or all
    if strcmp(char(covfunc{2}{1}{1}),'covPoly') 
        disp("Modify code to support polynomial kernel.");
        return
        cov_param_nrA = 3; cov_param_nrB = 3; % c, sf, sn
        cov_rangesA   = [-10,0;-10,-2;-4,-1];
        cov_rangesB   = [-10,0;-10,-2;-4,-1];
    else % assume covMaternard_modified
        % cov_param_nrA = 3; cov_param_nrB = 4;
        % cov_rangesA = [-10,0;-10,-2;-4,-1];
        % cov_rangesB = [-Inf,-Inf;-10,0;-10,-2;-4,-1];
        cov_param_nrA = 2 + lag; 
        cov_rangesA   = [repmat([-10, 0],lag,1); -10,-2; -4,-1];
        cov_param_nrB = 2 + 2*lag;
        cov_rangesB   = [repmat([-Inf, -Inf],lag,1);repmat([-10, 0],lag,1);-10,-2;-4,-1];
    end
end

% more parameters, due to extra covariates - side information
if strcmp(char(meanfunc{1}),'meanLinear') 
    % side_info_mean_paramA_nr = 2; side_info_mean_paramB_nr = 3;
    % side_info_mean_rangesA = [-1,1;-1,1]; side_info_mean_rangesB = [-1,1;-1,1;-1,1];
    side_info_mean_paramA_nr = 2*lag; %lags for one covariate and lags for side information 
    side_info_mean_rangesA = repmat([-1, 1],2*lag,1); 
    side_info_mean_paramB_nr = 3*lag;
    side_info_mean_rangesB = repmat([-1, 1],3*lag,1);
else %assume poly
    disp("Modify code to support polynomial kernel.");
    return
    side_info_mean_paramA_nr = 2; side_info_mean_paramB_nr = 4;
    side_info_mean_rangesA = [-1,1;-1,1]; side_info_mean_rangesB = [-1,1;-1,1;-1,1;-1,1];
end

if strcmp(char(covfunc{2}{1}{1}),'covPoly') 
    disp("Modify code to support polynomial kernel.");
    return
    side_info_cov_param_nrA = 3; 
    side_info_cov_param_nrB = 3; % c, sf, sn
    side_info_cov_rangesA = [-10,0;-10,-2;-4,-1];
    side_info_cov_rangesB = [-10,0;-10,-2;-4,-1];
else % assume covMaternard_modified
    % side_info_cov_param_nrA = 4; 
    % side_info_cov_param_nrB = 5;
    % side_info_cov_rangesA = [-10,0;-10,0;      -10,-2;-4,-1];
    % side_info_cov_rangesB = [-10,0;-10,0;-10,0;-10,-2;-4,-1];
    side_info_cov_param_nrA = 2 + 2*lag;
    side_info_cov_rangesA = [repmat([-10, 0],lag,1);repmat([-10, 0],lag,1);                       -10,-2;-4,-1];
    side_info_cov_param_nrB = 2 + 3*lag;
    side_info_cov_rangesB = [repmat([-10, 0],lag,1);repmat([-10, 0],lag,1);repmat([-10, 0],lag,1);-10,-2;-4,-1];
end

if strcmp(causal_placement_switch,'mean') %now cater only for causality in mean or all
    if strcmp(char(covfunc{2}{1}{1}),'covPoly') 
        disp("Modify code to support polynomial kernel.");
        return
        side_info_cov_param_nrA = 3; side_info_cov_param_nrB = 3; % c, sf, sn
        side_info_cov_rangesA = [-10,0;-10,-2;-4,-1];
        side_info_cov_rangesB = [-10,0;-10,-2;-4,-1];
    else % assume covMaternard_modified
        % side_info_cov_param_nrA = 4; side_info_cov_param_nrB = 5;
        % side_info_cov_rangesA = [-10,0;-Inf,-Inf;-10,-2;-4,-1];
        % side_info_cov_rangesB = [-Inf,-Inf;-10,0;-Inf,-Inf;-10,-2;-4,-1];
        side_info_cov_param_nrA = 2 + 2*lag;
        side_info_cov_rangesA = [repmat([-10, 0],lag,1);repmat([-Inf, -Inf],lag,1);                          -10,-2;-4,-1];
        side_info_cov_param_nrB = 2 + 3*lag;
        side_info_cov_rangesB = [repmat([-Inf,-Inf],lag,1);repmat([-10, 0],lag,1);repmat([-Inf, -Inf],lag,1);-10,-2;-4,-1];
    end
end
    
% preallocate memory for the predicted (fitted) values
mean_predA_XY_vec_pricesent = zeros(multirun_number,1);
mean_predB_XY_vec_pricesent = zeros(multirun_number,1);
mean_predA_YX_vec_pricesent = zeros(multirun_number,1);
mean_predB_YX_vec_pricesent = zeros(multirun_number,1);
% preallocate memory for saving the parameters
hyps_A_XY_vec_pricesent = nan(multirun_number,mean_paramA_nr+cov_param_nrA);
hyps_B_XY_vec_pricesent = nan(multirun_number,mean_paramB_nr+cov_param_nrB);
hyps_A_YX_vec_pricesent = nan(multirun_number,mean_paramA_nr+cov_param_nrA);
hyps_B_YX_vec_pricesent = nan(multirun_number,mean_paramB_nr+cov_param_nrB);
% preallocate memory for saving the neg loglik
nlmlA_XY_multiruns_pricesent = zeros(multirun_number,1);
nlmlB_XY_multiruns_pricesent = zeros(multirun_number,1);
nlmlA_YX_multiruns_pricesent = zeros(multirun_number,1);
nlmlB_YX_multiruns_pricesent = zeros(multirun_number,1);

% preallocate memory for the predicted (fitted) values
mean_predA_XY_vec_priceretsent = zeros(multirun_number,1);
mean_predB_XY_vec_priceretsent = zeros(multirun_number,1);
mean_predA_YX_vec_priceretsent = zeros(multirun_number,1);
mean_predB_YX_vec_priceretsent = zeros(multirun_number,1);
% preallocate memory for saving the parameters
hyps_A_XY_vec_priceretsent = nan(multirun_number,mean_paramA_nr+cov_param_nrA);
hyps_B_XY_vec_priceretsent = nan(multirun_number,mean_paramB_nr+cov_param_nrB);
hyps_A_YX_vec_priceretsent = nan(multirun_number,mean_paramA_nr+cov_param_nrA);
hyps_B_YX_vec_priceretsent = nan(multirun_number,mean_paramB_nr+cov_param_nrB);
% preallocate memory for saving the neg loglik
nlmlA_XY_multiruns_priceretsent = zeros(multirun_number,1);
nlmlB_XY_multiruns_priceretsent = zeros(multirun_number,1);
nlmlA_YX_multiruns_priceretsent = zeros(multirun_number,1);
nlmlB_YX_multiruns_priceretsent = zeros(multirun_number,1);

% preallocate memory for the predicted (fitted) values
mean_predA_XY_vec_volsent = zeros(multirun_number,1);
mean_predB_XY_vec_volsent = zeros(multirun_number,1);
mean_predA_YX_vec_volsent = zeros(multirun_number,1);
mean_predB_YX_vec_volsent = zeros(multirun_number,1);
% preallocate memory for saving the parameters
hyps_A_XY_vec_volsent = nan(multirun_number,mean_paramA_nr+cov_param_nrA);
hyps_B_XY_vec_volsent = nan(multirun_number,mean_paramB_nr+cov_param_nrB);
hyps_A_YX_vec_volsent = nan(multirun_number,mean_paramA_nr+cov_param_nrA);
hyps_B_YX_vec_volsent = nan(multirun_number,mean_paramB_nr+cov_param_nrB);
% preallocate memory for saving the neg loglik
nlmlA_XY_multiruns_volsent = zeros(multirun_number,1);
nlmlB_XY_multiruns_volsent = zeros(multirun_number,1);
nlmlA_YX_multiruns_volsent = zeros(multirun_number,1);
nlmlB_YX_multiruns_volsent = zeros(multirun_number,1);

% preallocate memory for the predicted (fitted) values
mean_predA_XY_vec_realvolsent = zeros(multirun_number,1);
mean_predB_XY_vec_realvolsent = zeros(multirun_number,1);
mean_predA_YX_vec_realvolsent = zeros(multirun_number,1);
mean_predB_YX_vec_realvolsent = zeros(multirun_number,1);
% preallocate memory for saving the parameters
hyps_A_XY_vec_realvolsent = nan(multirun_number,mean_paramA_nr+cov_param_nrA);
hyps_B_XY_vec_realvolsent = nan(multirun_number,mean_paramB_nr+cov_param_nrB);
hyps_A_YX_vec_realvolsent = nan(multirun_number,mean_paramA_nr+cov_param_nrA);
hyps_B_YX_vec_realvolsent = nan(multirun_number,mean_paramB_nr+cov_param_nrB);
% preallocate memory for saving the neg loglik
nlmlA_XY_multiruns_realvolsent = zeros(multirun_number,1);
nlmlB_XY_multiruns_realvolsent = zeros(multirun_number,1);
nlmlA_YX_multiruns_realvolsent = zeros(multirun_number,1);
nlmlB_YX_multiruns_realvolsent = zeros(multirun_number,1);

% preallocate memory for the predicted (fitted) values
mean_predA_XY_vec_volrealvolsent = zeros(multirun_number,1);
mean_predB_XY_vec_volrealvolsent = zeros(multirun_number,1);
mean_predA_YX_vec_volrealvolsent = zeros(multirun_number,1);
mean_predB_YX_vec_volrealvolsent = zeros(multirun_number,1);
% preallocate memory for saving the parameters
hyps_A_XY_vec_volrealvolsent = nan(multirun_number, side_info_mean_paramA_nr+side_info_cov_param_nrA);
hyps_B_XY_vec_volrealvolsent = nan(multirun_number,side_info_mean_paramB_nr+side_info_cov_param_nrB);
hyps_A_YX_vec_volrealvolsent = nan(multirun_number,side_info_mean_paramA_nr+side_info_cov_param_nrA);
hyps_B_YX_vec_volrealvolsent = nan(multirun_number,side_info_mean_paramB_nr+side_info_cov_param_nrB);
% preallocate memory for saving the neg loglik
nlmlA_XY_multiruns_volrealvolsent = zeros(multirun_number,1);
nlmlB_XY_multiruns_volrealvolsent = zeros(multirun_number,1);
nlmlA_YX_multiruns_volrealvolsent = zeros(multirun_number,1);
nlmlB_YX_multiruns_volrealvolsent = zeros(multirun_number,1);

eppsepps_ee100 = zeros(multirun_number, 13);
eppsepps_gp100 = zeros(multirun_number, 13);
eppsepps_ee7 = zeros(multirun_number, 13);
eppsepps_gp7 = zeros(multirun_number, 13);
bol7 = zeros(multirun_number, 16);
bol100 = zeros(multirun_number, 16);

tic   
jjjj = 1;
clear rejectnullXY;
clear rejectnullYX;
for run_ii = from_loop:to_loop
    run_ii
    
    Data = all_data(:,:,:,:,:,run_ii);

    % [win_data_zscore_price, win_data_zscore_priceret, win_data_zscore_vol, win_data_zscore_realvol, win_sent_zscore, win_data_zscore_logrealvol, win_data_zscore_logvo100, win_data_zscore_logvo7];

    % CONSIDER X: SENTIMENT - Y: MARKET
    %%%%%%% PRICE <-> SENTIMENT %%%%%%%
    % will want to test X --> Y | Z and Y --> X | Z
    %%%%%%%%% for X --> Y | Z 
    fromXY = 5; toXY = 1; sideXY = [];
    %%%%%%%%% for Y --> X | Z
    fromYX = 1; toYX = 5; sideYX = [];    
    [Causality_XY_vec_ii, Causality_YX_vec_ii, XY_chi2cdf_vec_ii, YX_chi2cdf_vec_ii,...
          hyps_A_XY_vec_ii, hyps_B_XY_vec_ii, hyps_A_YX_vec_ii, hyps_B_YX_vec_ii,...
          mean_predA_XY_vec_ii, mean_predB_XY_vec_ii, nlmlA_XY_multiruns_ii, nlmlB_XY_multiruns_ii,...
          mean_predA_YX_vec_ii, mean_predB_YX_vec_ii, nlmlA_YX_multiruns_ii, nlmlB_YX_multiruns_ii,...
          pvalueXYGranger_ii, pvalueYXGranger_ii] =...
          train_gp_and_do_test(fromXY, toXY, sideXY,...
                            fromYX, toYX, sideYX,...
                            Data, lag, mean_paramA_nr, cov_param_nrA,...
                            [mean_rangesA;... % mean
                            cov_rangesA],... %cov
                            grid_nr, requested_nr, meanfunc, covfunc, likfunc, hypA.lik, inffunc,...
                            run_ii, mean_paramB_nr, cov_param_nrB,...
                            [mean_rangesB;... % mean
                            cov_rangesB],... %cov
                            hypB.lik, step_nr, causal_placement_switch, floor_n);
    mean_predA_XY_vec_pricesent(run_ii) = mean_predA_XY_vec_ii;
    mean_predB_XY_vec_pricesent(run_ii) = mean_predB_XY_vec_ii;
    mean_predA_YX_vec_pricesent(run_ii) = mean_predA_YX_vec_ii;
    mean_predB_YX_vec_pricesent(run_ii) = mean_predB_YX_vec_ii;
    hyps_A_XY_vec_pricesent(run_ii,:) = hyps_A_XY_vec_ii;
    hyps_B_XY_vec_pricesent(run_ii,:) = hyps_B_XY_vec_ii;
    hyps_A_YX_vec_pricesent(run_ii,:) = hyps_A_YX_vec_ii;
    hyps_B_YX_vec_pricesent(run_ii,:) = hyps_B_YX_vec_ii;
    nlmlA_XY_multiruns_pricesent(run_ii) = nlmlA_XY_multiruns_ii;
    nlmlB_XY_multiruns_pricesent(run_ii) = nlmlB_XY_multiruns_ii;
    nlmlA_YX_multiruns_pricesent(run_ii) = nlmlA_YX_multiruns_ii;
    nlmlB_YX_multiruns_pricesent(run_ii) = nlmlB_YX_multiruns_ii;
    Causality_XY_vec_pricesent(run_ii) = Causality_XY_vec_ii;
    Causality_YX_vec_pricesent(run_ii) = Causality_YX_vec_ii;
    XY_chi2cdf_vec_pricesent(run_ii) = XY_chi2cdf_vec_ii;
    YX_chi2cdf_vec_pricesent(run_ii) = YX_chi2cdf_vec_ii;
    Causality_XY_vec_Granger_pricesent(run_ii) = pvalueXYGranger_ii;
    Causality_YX_vec_Granger_pricesent(run_ii) = pvalueYXGranger_ii;


    %%%%%%% PRICE_RET <-> SENTIMENT %%%%%%%
    % will want to test X --> Y | Z and Y --> X | Z
    %%%%%%%%% for X --> Y | Z 
    fromXY = 5; toXY = 2; sideXY = [];
    %%%%%%%%% for Y --> X | Z
    fromYX = 2; toYX = 5; sideYX = [];    
    [Causality_XY_vec_ii, Causality_YX_vec_ii, XY_chi2cdf_vec_ii, YX_chi2cdf_vec_ii,...
          hyps_A_XY_vec_ii, hyps_B_XY_vec_ii, hyps_A_YX_vec_ii, hyps_B_YX_vec_ii,...
          mean_predA_XY_vec_ii, mean_predB_XY_vec_ii, nlmlA_XY_multiruns_ii, nlmlB_XY_multiruns_ii,...
          mean_predA_YX_vec_ii, mean_predB_YX_vec_ii, nlmlA_YX_multiruns_ii, nlmlB_YX_multiruns_ii,...
          pvalueXYGranger_ii, pvalueYXGranger_ii] =...
          train_gp_and_do_test(fromXY, toXY, sideXY,...
                            fromYX, toYX, sideYX,...
                            Data, lag, mean_paramA_nr, cov_param_nrA,...
                            [mean_rangesA;... % mean
                            cov_rangesA],... %cov
                            grid_nr, requested_nr, meanfunc, covfunc, likfunc, hypA.lik, inffunc,...
                            run_ii, mean_paramB_nr, cov_param_nrB,...
                            [mean_rangesB;... % mean
                            cov_rangesB],... %cov
                            hypB.lik, step_nr, causal_placement_switch, floor_n);
    mean_predA_XY_vec_priceretsent(run_ii) = mean_predA_XY_vec_ii;
    mean_predB_XY_vec_priceretsent(run_ii) = mean_predB_XY_vec_ii;
    mean_predA_YX_vec_priceretsent(run_ii) = mean_predA_YX_vec_ii;
    mean_predB_YX_vec_priceretsent(run_ii) = mean_predB_YX_vec_ii;
    hyps_A_XY_vec_priceretsent(run_ii,:) = hyps_A_XY_vec_ii;
    hyps_B_XY_vec_priceretsent(run_ii,:) = hyps_B_XY_vec_ii;
    hyps_A_YX_vec_priceretsent(run_ii,:) = hyps_A_YX_vec_ii;
    hyps_B_YX_vec_priceretsent(run_ii,:) = hyps_B_YX_vec_ii;
    nlmlA_XY_multiruns_priceretsent(run_ii) = nlmlA_XY_multiruns_ii;
    nlmlB_XY_multiruns_priceretsent(run_ii) = nlmlB_XY_multiruns_ii;
    nlmlA_YX_multiruns_priceretsent(run_ii) = nlmlA_YX_multiruns_ii;
    nlmlB_YX_multiruns_priceretsent(run_ii) = nlmlB_YX_multiruns_ii;
    Causality_XY_vec_priceretsent(run_ii) = Causality_XY_vec_ii;
    Causality_YX_vec_priceretsent(run_ii) = Causality_YX_vec_ii;
    XY_chi2cdf_vec_priceretsent(run_ii) = XY_chi2cdf_vec_ii;
    YX_chi2cdf_vec_priceretsent(run_ii) = YX_chi2cdf_vec_ii;
    Causality_XY_vec_Granger_priceretsent(run_ii) = pvalueXYGranger_ii;
    Causality_YX_vec_Granger_priceretsent(run_ii) = pvalueYXGranger_ii;


    %%%%%%% VOLUME <-> SENTIMENT %%%%%%%
    % will want to test X --> Y | Z and Y --> X | Z
    %%%%%%%%% for X --> Y | Z 
    fromXY = 5; toXY = 3; sideXY = [];
    %%%%%%%%% for Y --> X | Z
    fromYX = 3; toYX = 5; sideYX = [];    
    [Causality_XY_vec_ii, Causality_YX_vec_ii, XY_chi2cdf_vec_ii, YX_chi2cdf_vec_ii,...
          hyps_A_XY_vec_ii, hyps_B_XY_vec_ii, hyps_A_YX_vec_ii, hyps_B_YX_vec_ii,...
          mean_predA_XY_vec_ii, mean_predB_XY_vec_ii, nlmlA_XY_multiruns_ii, nlmlB_XY_multiruns_ii,...
          mean_predA_YX_vec_ii, mean_predB_YX_vec_ii, nlmlA_YX_multiruns_ii, nlmlB_YX_multiruns_ii,...
          pvalueXYGranger_ii, pvalueYXGranger_ii] =...
          train_gp_and_do_test(fromXY, toXY, sideXY,...
                            fromYX, toYX, sideYX,...
                            Data, lag, mean_paramA_nr, cov_param_nrA,...
                            [mean_rangesA;... % mean
                            cov_rangesA],... %cov
                            grid_nr, requested_nr, meanfunc, covfunc, likfunc, hypA.lik, inffunc,...
                            run_ii, mean_paramB_nr, cov_param_nrB,...
                            [mean_rangesB;... % mean
                            cov_rangesB],... %cov
                            hypB.lik, step_nr, causal_placement_switch, floor_n);
    mean_predA_XY_vec_volsent(run_ii) = mean_predA_XY_vec_ii;
    mean_predB_XY_vec_volsent(run_ii) = mean_predB_XY_vec_ii;
    mean_predA_YX_vec_volsent(run_ii) = mean_predA_YX_vec_ii;
    mean_predB_YX_vec_volsent(run_ii) = mean_predB_YX_vec_ii;
    hyps_A_XY_vec_volsent(run_ii,:) = hyps_A_XY_vec_ii;
    hyps_B_XY_vec_volsent(run_ii,:) = hyps_B_XY_vec_ii;
    hyps_A_YX_vec_volsent(run_ii,:) = hyps_A_YX_vec_ii;
    hyps_B_YX_vec_volsent(run_ii,:) = hyps_B_YX_vec_ii;
    nlmlA_XY_multiruns_volsent(run_ii) = nlmlA_XY_multiruns_ii;
    nlmlB_XY_multiruns_volsent(run_ii) = nlmlB_XY_multiruns_ii;
    nlmlA_YX_multiruns_volsent(run_ii) = nlmlA_YX_multiruns_ii;
    nlmlB_YX_multiruns_volsent(run_ii) = nlmlB_YX_multiruns_ii;
    Causality_XY_vec_volsent(run_ii) = Causality_XY_vec_ii;
    Causality_YX_vec_volsent(run_ii) = Causality_YX_vec_ii;
    XY_chi2cdf_vec_volsent(run_ii) = XY_chi2cdf_vec_ii;
    YX_chi2cdf_vec_volsent(run_ii) = YX_chi2cdf_vec_ii;
    Causality_XY_vec_Granger_volsent(run_ii) = pvalueXYGranger_ii;
    Causality_YX_vec_Granger_volsent(run_ii) = pvalueYXGranger_ii;



    %%%%%%% REAL VOL <-> SENTIMENT %%%%%%%
    % will want to test X --> Y | Z and Y --> X | Z
    %%%%%%%%% for X --> Y | Z 
    fromXY = 5; toXY = 4; sideXY = [];
    %%%%%%%%% for Y --> X | Z
    fromYX = 4; toYX = 5; sideYX = [];    
    [Causality_XY_vec_ii, Causality_YX_vec_ii, XY_chi2cdf_vec_ii, YX_chi2cdf_vec_ii,...
          hyps_A_XY_vec_ii, hyps_B_XY_vec_ii, hyps_A_YX_vec_ii, hyps_B_YX_vec_ii,...
          mean_predA_XY_vec_ii, mean_predB_XY_vec_ii, nlmlA_XY_multiruns_ii, nlmlB_XY_multiruns_ii,...
          mean_predA_YX_vec_ii, mean_predB_YX_vec_ii, nlmlA_YX_multiruns_ii, nlmlB_YX_multiruns_ii,...
          pvalueXYGranger_ii, pvalueYXGranger_ii] =...
          train_gp_and_do_test(fromXY, toXY, sideXY,...
                            fromYX, toYX, sideYX,...
                            Data, lag, mean_paramA_nr, cov_param_nrA,...
                            [mean_rangesA;... % mean
                            cov_rangesA],... %cov
                            grid_nr, requested_nr, meanfunc, covfunc, likfunc, hypA.lik, inffunc,...
                            run_ii, mean_paramB_nr, cov_param_nrB,...
                            [mean_rangesB;... % mean
                            cov_rangesB],... %cov
                            hypB.lik, step_nr, causal_placement_switch, floor_n);
    mean_predA_XY_vec_realvolsent(run_ii) = mean_predA_XY_vec_ii;
    mean_predB_XY_vec_realvolsent(run_ii) = mean_predB_XY_vec_ii;
    mean_predA_YX_vec_realvolsent(run_ii) = mean_predA_YX_vec_ii;
    mean_predB_YX_vec_realvolsent(run_ii) = mean_predB_YX_vec_ii;
    hyps_A_XY_vec_realvolsent(run_ii,:) = hyps_A_XY_vec_ii;
    hyps_B_XY_vec_realvolsent(run_ii,:) = hyps_B_XY_vec_ii;
    hyps_A_YX_vec_realvolsent(run_ii,:) = hyps_A_YX_vec_ii;
    hyps_B_YX_vec_realvolsent(run_ii,:) = hyps_B_YX_vec_ii;
    nlmlA_XY_multiruns_realvolsent(run_ii) = nlmlA_XY_multiruns_ii;
    nlmlB_XY_multiruns_realvolsent(run_ii) = nlmlB_XY_multiruns_ii;
    nlmlA_YX_multiruns_realvolsent(run_ii) = nlmlA_YX_multiruns_ii;
    nlmlB_YX_multiruns_realvolsent(run_ii) = nlmlB_YX_multiruns_ii;
    Causality_XY_vec_realvolsent(run_ii) = Causality_XY_vec_ii;
    Causality_YX_vec_realvolsent(run_ii) = Causality_YX_vec_ii;
    XY_chi2cdf_vec_realvolsent(run_ii) = XY_chi2cdf_vec_ii;
    YX_chi2cdf_vec_realvolsent(run_ii) = YX_chi2cdf_vec_ii;
    Causality_XY_vec_Granger_realvolsent(run_ii) = pvalueXYGranger_ii;
    Causality_YX_vec_Granger_realvolsent(run_ii) = pvalueYXGranger_ii;


    %%%%%%% REAL VOL, SENTIMENT <-> VOLUME %%%%%%%
    % will want to test X --> Y | Z and Y --> X | Z
    %%%%%%%%% for X --> Y | Z 
    fromXY = 5; toXY = 3; sideXY = [4];
    %%%%%%%%% for Y --> X | Z
    fromYX = 3; toYX = 5; sideYX = [4];    
    [Causality_XY_vec_ii, Causality_YX_vec_ii, XY_chi2cdf_vec_ii, YX_chi2cdf_vec_ii,...
          hyps_A_XY_vec_ii, hyps_B_XY_vec_ii, hyps_A_YX_vec_ii, hyps_B_YX_vec_ii,...
          mean_predA_XY_vec_ii, mean_predB_XY_vec_ii, nlmlA_XY_multiruns_ii, nlmlB_XY_multiruns_ii,...
          mean_predA_YX_vec_ii, mean_predB_YX_vec_ii, nlmlA_YX_multiruns_ii, nlmlB_YX_multiruns_ii,...
          pvalueXYGranger_ii, pvalueYXGranger_ii] =...
          train_gp_and_do_test(fromXY, toXY, sideXY,...
                            fromYX, toYX, sideYX,...
                            Data, lag, side_info_mean_paramA_nr, side_info_cov_param_nrA,...
                            [side_info_mean_rangesA;... % mean
                            side_info_cov_rangesA],... %cov
                            grid_nr, requested_nr, meanfunc, covfunc, likfunc, hypA.lik, inffunc,...
                            run_ii, side_info_mean_paramB_nr, side_info_cov_param_nrB,...
                            [side_info_mean_rangesB;... % mean
                            side_info_cov_rangesB],... %cov
                            hypB.lik, step_nr, causal_placement_switch, floor_n);
    mean_predA_XY_vec_volrealvolsent(run_ii) = mean_predA_XY_vec_ii;
    mean_predB_XY_vec_volrealvolsent(run_ii) = mean_predB_XY_vec_ii;
    mean_predA_YX_vec_volrealvolsent(run_ii) = mean_predA_YX_vec_ii;
    mean_predB_YX_vec_volrealvolsent(run_ii) = mean_predB_YX_vec_ii;
    hyps_A_XY_vec_volrealvolsent(run_ii,:) = hyps_A_XY_vec_ii;
    hyps_B_XY_vec_volrealvolsent(run_ii,:) = hyps_B_XY_vec_ii;
    hyps_A_YX_vec_volrealvolsent(run_ii,:) = hyps_A_YX_vec_ii;
    hyps_B_YX_vec_volrealvolsent(run_ii,:) = hyps_B_YX_vec_ii;
    nlmlA_XY_multiruns_volrealvolsent(run_ii) = nlmlA_XY_multiruns_ii;
    nlmlB_XY_multiruns_volrealvolsent(run_ii) = nlmlB_XY_multiruns_ii;
    nlmlA_YX_multiruns_volrealvolsent(run_ii) = nlmlA_YX_multiruns_ii;
    nlmlB_YX_multiruns_volrealvolsent(run_ii) = nlmlB_YX_multiruns_ii;
    Causality_XY_vec_volrealvolsent(run_ii) = Causality_XY_vec_ii;
    Causality_YX_vec_volrealvolsent(run_ii) = Causality_YX_vec_ii;
    XY_chi2cdf_vec_volrealvolsent(run_ii) = XY_chi2cdf_vec_ii;
    YX_chi2cdf_vec_volrealvolsent(run_ii) = YX_chi2cdf_vec_ii;
    Causality_XY_vec_Granger_volrealvolsent(run_ii) = pvalueXYGranger_ii;
    Causality_YX_vec_Granger_volrealvolsent(run_ii) = pvalueYXGranger_ii;


    if ~strcmp(causal_placement_switch,'mean')
        % save everything only in meancov case
        % report ref date, window start date, window end data, p-val GP XY, teststat GP XY,
        rejectnullXY(jjjj,:) = {char(datestr(data_cut_init_time(run_ii,1), "yyyy-mm-dd")), char(datestr(data_cut_init_time(run_ii,2), "yyyy-mm-dd")),...
                                1-XY_chi2cdf_vec_pricesent(run_ii), Causality_XY_vec_pricesent(run_ii),...
                                1-XY_chi2cdf_vec_priceretsent(run_ii), Causality_XY_vec_priceretsent(run_ii),...
                                1-XY_chi2cdf_vec_volsent(run_ii), Causality_XY_vec_volsent(run_ii),...
                                1-XY_chi2cdf_vec_realvolsent(run_ii), Causality_XY_vec_realvolsent(run_ii),...
                                1-XY_chi2cdf_vec_volrealvolsent(run_ii), Causality_XY_vec_volrealvolsent(run_ii),...
                                Causality_XY_vec_Granger_pricesent(run_ii), Causality_XY_vec_Granger_priceretsent(run_ii),...
                                Causality_XY_vec_Granger_volsent(run_ii), Causality_XY_vec_Granger_realvolsent(run_ii),...
                                Causality_XY_vec_Granger_volrealvolsent(run_ii), eppsepps_gp100(run_ii, :), eppsepps_ee100(run_ii, :),...
                                eppsepps_gp7(run_ii, :), eppsepps_ee7(run_ii, :), bol7(run_ii, :), bol100(run_ii, :)};
        rejectnullYX(jjjj,:) = {char(datestr(data_cut_init_time(run_ii,1), "yyyy-mm-dd")), char(datestr(data_cut_init_time(run_ii,2), "yyyy-mm-dd")),...
                                1-YX_chi2cdf_vec_pricesent(run_ii), Causality_YX_vec_pricesent(run_ii),...
                                1-YX_chi2cdf_vec_priceretsent(run_ii), Causality_YX_vec_priceretsent(run_ii),...
                                1-YX_chi2cdf_vec_volsent(run_ii), Causality_YX_vec_volsent(run_ii),...
                                1-YX_chi2cdf_vec_realvolsent(run_ii), Causality_YX_vec_realvolsent(run_ii),...
                                1-YX_chi2cdf_vec_volrealvolsent(run_ii), Causality_YX_vec_volrealvolsent(run_ii),...
                                Causality_YX_vec_Granger_pricesent(run_ii), Causality_YX_vec_Granger_priceretsent(run_ii),...
                                Causality_YX_vec_Granger_volsent(run_ii), Causality_YX_vec_Granger_realvolsent(run_ii),...
                                Causality_YX_vec_Granger_volrealvolsent(run_ii)};
    else
        rejectnullXY(jjjj,:) = {char(datestr(data_cut_init_time(run_ii,1), "yyyy-mm-dd")), char(datestr(data_cut_init_time(run_ii,2), "yyyy-mm-dd")),...
                                1-XY_chi2cdf_vec_pricesent(run_ii), Causality_XY_vec_pricesent(run_ii),...
                                1-XY_chi2cdf_vec_priceretsent(run_ii), Causality_XY_vec_priceretsent(run_ii),...
                                1-XY_chi2cdf_vec_volsent(run_ii), Causality_XY_vec_volsent(run_ii),...
                                1-XY_chi2cdf_vec_realvolsent(run_ii), Causality_XY_vec_realvolsent(run_ii),...
                                1-XY_chi2cdf_vec_volrealvolsent(run_ii), Causality_XY_vec_volrealvolsent(run_ii)};
        rejectnullYX(jjjj,:) = {char(datestr(data_cut_init_time(run_ii,1), "yyyy-mm-dd")), char(datestr(data_cut_init_time(run_ii,2), "yyyy-mm-dd")),...
                                1-YX_chi2cdf_vec_pricesent(run_ii), Causality_YX_vec_pricesent(run_ii),...
                                1-YX_chi2cdf_vec_priceretsent(run_ii), Causality_YX_vec_priceretsent(run_ii),...
                                1-YX_chi2cdf_vec_volsent(run_ii), Causality_YX_vec_volsent(run_ii),...
                                1-YX_chi2cdf_vec_realvolsent(run_ii), Causality_YX_vec_realvolsent(run_ii),...
                                1-YX_chi2cdf_vec_volrealvolsent(run_ii), Causality_YX_vec_volrealvolsent(run_ii)};
    end
    jjjj = jjjj + 1;

    outpart = strcat(dir_out, 'test1_optim_predm_', causal_placement_switch, '_');
    eval(strcat('save(',char(39), outpart, meanf,'_',covf,'_', name,...
                '_from',num2str(from_loop),'_to' ,num2str(to_loop),...
                '.mat',char(39), ', "-v7")'));
    run_ii
end
toc