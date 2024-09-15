function [structuralchanges, structuralchanges_chi2cdf_vec] = ...
         test_for_structural_change(all_data, from_loop, to_loop,...
                                    meanf, likf, name, lag, covf, ...
                                    causal_placement_switch, ...
                                    before_and_after_report_switch, dir_out, ...
                                    ref_date, data_cut_init_time, ...
                                    consider_win_point, stepshift, do_fit, do_save)


% If we want to just load a trained GP model and repeat the testing
% procedure - set to 1
do_test_using_fitted_model = 0;  %0  set to 1 for means  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Load data for Study IV
forecast = 1;  %1   % set to 0 for means
disp('Start testing!');
time_now = now;
starting_time = floor(1.e+06 * rem(time_now,1));
starting_time_name = num2str(starting_time);
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
        hypA.lik  = log(0); 
    case {'t-student', 'student-t', 't', 'likT'}
        likfunc = {@likT};
        inffunc = @infLaplace;
        hypA.lik  = [log(0); log(1)]; 
    otherwise
        likfunc  = {@likGauss}; 
        inffunc = @infExact;
        hypA.lik  = log(0); 
end

%% run many tests to compare!
multirun_number = 100;
if from_loop == 1
    multirun_number = to_loop;
end
step_nr = -30; % in optimisation
requested_nr = 10; % how many best starting points  
if lag == 1
    grid_nr = 5; % grid for the starting points
elseif lag < 5
    % reduce compute cost
    grid_nr = 2.7;
elseif lag <= 8 
    grid_nr = 1.6;
else
    grid_nr = 1.5;
end

nlmlA_XY_multiruns = zeros(multirun_number, 1);
%%%%%%% many starting points:
if strcmp(char(meanfunc{1}), 'meanLinear') 
    mean_paramA_nr = lag; 
    mean_rangesA = repmat([-1, 1],lag,1);
else %assume poly
    disp("Adjust code to support polynomial mean")
    return
    mean_paramA_nr = 2;
    mean_rangesA = [-1, 1; -1, 1]; 
end
% assume covMaternard_modified
cov_param_nrA = 2 + lag;
cov_rangesA = [repmat([-10, 0],lag,1); -10,-2; -4,-1];

% preallocate memory for the predicted (fitted) values
mean_predA_XY_vec = zeros(multirun_number, 1);
% preallocate memory for saving the parameters
hyps_A_XY_vec = nan(multirun_number, mean_paramA_nr+cov_param_nrA);

tic    
jjj = 1;
% wins from position 2 to correspond the test outputs with the window position in Data
wins = 2;
clear structuralchanges_chi2cdf_vec;
clear structuralchanges;
clear rejectnull;
rejectnull = {};
structuralchanges = [];
structuralchanges_chi2cdf_vec = [];

medians_of_means = {};
means = [];

%% Load saved data
optimise = do_fit;  
if optimise == 0 || do_test_using_fitted_model == 1 
    disp("Loading saved model...");
    outname = strcat(dir_out, 'test1_optim_predm', meanf, '_', covf, '_', ...
                    name, '_len', num2str(length_data),...
                    '_from', num2str(from_loop), '_to', num2str(to_loop), '.mat')
    eval(strcat('load(',char(39), outname, char(39), ...
                ',"hyps_A_XY_vec"', ...
                ',"nlmlA_XY_multiruns"', ...
                ',"rejectnull")')); 
    hypA_XY_first.mean = hyps_A_XY_vec(1, 1:lag)';
    hypA_XY_first.cov  = hyps_A_XY_vec(1, lag+1:end)';
    hypA_XY_first.lik  = nlmlA_XY_multiruns(1);
end

%% Fit GP for testing structural change
if optimise == 1 || do_test_using_fitted_model == 1
    for run_ii = 1:to_loop

        if (run_ii > 1) && (run_ii < consider_win_point)
                % fit only the first window (before report release) and windows
                % that overlap with the report release date
                continue
        end

        Data = all_data(:, run_ii);
        toXY = 1; fromXY = []; sideXY = []; 
        [rejectnull, ...
         hyps_A_XY_vec, wins, jjj, ...
         nlmlA_XY_multiruns, ...
         structuralchanges, ...
         structuralchanges_chi2cdf_vec] = train_gp_and_do_test_structural(fromXY, toXY, sideXY, ...
                                            run_ii, consider_win_point, wins, jjj, ...  
                                            Data, lag, do_test_using_fitted_model, ...                
                                            mean_paramA_nr, cov_param_nrA, ...
                                            [mean_rangesA; cov_rangesA], ...
                                            grid_nr, step_nr, requested_nr, meanfunc, ...
                                            covfunc, likfunc, hypA.lik, inffunc, ref_date, ...
                                            data_cut_init_time, rejectnull, hyps_A_XY_vec, ...
                                            mean_predA_XY_vec, nlmlA_XY_multiruns, ...
                                            structuralchanges, structuralchanges_chi2cdf_vec);
        
        %%%%%%%%%%%%%%%%%%%%%% FOR STUDY III %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % get here with: do_test_using_fitted_model == 1 
        % compute mean per timepoint in window, get median of all means per window        
        % meanparam = hyps_A_XY_vec(jjj,1:lag);
        % jjjj = 1;
        % clear means;
        % means = [];
        % for ii=lag+1:size(Data,1)
        %     x = Data(ii-lag:ii-1);
        %     % mean = meanparam * x; - equiv. to below for meanLinear
        %     mean = feval(meanfunc{:}, hyps_A_XY_vec(jjj,1:lag)', Data(ii-lag:ii-1)');
        %     means(jjjj) = mean;
        %     jjjj = jjjj + 1; 
        % end
        % [medy midx] = min(abs(means-median(means)));
        % medians_of_means(jjj, :) = {datestr(datetime(ref_date), "yyyy-mm-dd HH:00:00"), 
        %                             datestr(data_cut_init_time(run_ii, 1), "yyyy-mm-dd HH:00:00"), 
        %                             datestr(data_cut_init_time(run_ii, 2), "yyyy-mm-dd HH:00:00"), 
        %                             double(medy), 
        %                             means};
        % if sum(strcmp(rejectnull(jjj,1:3), medians_of_means(jjj,1:3)))~=3
        %     disp("sgs")
        % end
        % jjj = jjj + 1;
        %%%%%%%%%%%%%%%%%%%%%% FOR STUDY III END %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        

    end
    toc
    if do_save
        outname = strcat(dir_out, 'test1_optim_predm', meanf, '_', covf, '_', name, '_len', ...
                  num2str(length_data), '_from', num2str(from_loop),'_to', ndir_outum2str(to_loop), '.mat')
        % eval(strcat('save(',char(39), outname, char(39), ',"hyps_A_XY_vec"', ...
        %     ',"nlmlA_XY_multiruns"', ',"structuralchanges",', ',"structuralchanges_chi2cdf_vec",', ',"rejectnull",', ' "-v7")'));
        
        %%%%%%%%%%%%%%%%%%%%%% FOR STUDY III %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % outname = strcat(dir_out, 'test1_optim_predm', meanf, '_', covf, '_', name, '_len', ...
        %           num2str(length_data), '_from', num2str(from_loop),'_to', num2str(to_loop), '_medianofmeans.mat')
        % eval(strcat('save(',char(39), outname, char(39), ',"medians_of_means", "-v7")'));
        %%%%%%%%%%%%%%%%%%%%%% FOR STUDY III END %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    end
end

%% Load data for Study IV
forecast_horizons = [0:1:size(all_data,1)];
variance_forecast = zeros(2, length(forecast_horizons));
clear allpaths;
clear control_study_rejectnull;
clear control_study_structuralchanges;
clear control_study_structuralchanges_chi2cdf_vec;
iii = 1;

if isfile(strcat(dir_out, "forecast_variances_0_1_91_", name, "_updFeb24.mat"))
    return
end

% change to dir_out for GP fitting. otherwise dir_out2
if forecast 
    % forecast using pre-report fitted model and first model fitted on
    % window with report date
    jjj = 1;
    for wiiin=1:to_loop
        if (wiiin > 1) && (wiiin < consider_win_point)
            continue
        end
        path = zeros([size(forecast_horizons,2)+lag,1]);        
        Data = all_data(:,wiiin);
        toXY = 1; fromXY = []; sideXY = [];         
        [input_to_lagged, input_from_lagged, input_side_lagged] = ...
                                compute_lags(Data, toXY, fromXY, sideXY, lag);
        xxxM1 = [input_to_lagged, input_side_lagged];
        yyyM1 = Data(lag+1:end, toXY);

        % h=0 -> in sample variance
        % in sample
        variance_forecast(jjj, 1) = var(yyyM1);
        path(1:lag) = yyyM1((end-lag+1):end); 

        % parameters of M1
        hypA_XY_M1.mean = hyps_A_XY_vec(jjj, 1:lag)';
        hypA_XY_M1.cov  = hyps_A_XY_vec(jjj, lag+1:end)';   
        hypA_XY_M1.lik  = nlmlA_XY_multiruns(jjj);        
        for f=(lag+1):1:length(forecast_horizons)
            xsM1 = path(f-lag:f-1)';
            [yysM1, ys2, fmu, fs2   ] = gp(hypA_XY_M1, inffunc, meanfunc, ...
                                            covfunc, likfunc, xxxM1, yyyM1, xsM1);
            path(f) = yysM1; 
            % gap of zeros in variance_forecast after in-sample value at 1, 
            % same length as lags
            variance_forecast(jjj, f) = fs2;
        end           
        
        allpaths(jjj, :) = path;
        % evaluate GLRT test statistic on forecast data from no-report model and actual data in the corresponding original model period - same model at each comparison
        if jjj > 1
            [input_to_lagged, input_from_lagged, input_side_lagged] = ...
                                compute_lags(allpaths(1,:)', 1, [], [], lag); % forecast data from no-report model
            allpaths_xx = input_to_lagged;
            allpaths_yy = allpaths(1, lag+1:end)';
            % current model fit on forecasts of the model originally fit on no-report data
            data_no_report_lik = gp(hypA_XY_M1, inffunc, meanfunc, covfunc, ...
                                    likfunc, allpaths_xx, ...
                                    allpaths_yy);

            % [input_to_lagged, input_from_lagged, input_side_lagged] = ...
            %                         compute_lags(path, 1, [], [], lag);
            % path_xx = input_to_lagged;
            % path_yy = path(lag+1:end);
            % % current model fit on its forecasts
            % current_data_lik = gp(hypA_XY_M1, inffunc, meanfunc, covfunc, ...
            %                         likfunc, path_xx, path_yy);
            % actual data in the corresponding original model period
            current_data_lik = hypA_XY_M1.lik;

            control_study_structuralchanges(iii, 1) = data_no_report_lik; 
            control_study_structuralchanges(iii, 2) = current_data_lik; 
            control_study_structuralchanges(iii, 3) = -2*(current_data_lik - data_no_report_lik); % if reject null: in favor of report publication actually making a difference - ie no report, no structural change
            degf = mean_paramA_nr + cov_param_nrA;
            control_study_structuralchanges_chi2cdf_vec(iii) = chi2cdf(control_study_structuralchanges(iii), degf);
            control_study_rejectnull(iii,:) = {datestr(datetime(ref_date), "yyyy-mm-dd"), datestr(data_cut_init_time(wiiin, 1), ... # for hourly lags, the window shift does not mean that its start/end dates change
                                    "yyyy-mm-dd"), datestr(data_cut_init_time(wiiin, 2), "yyyy-mm-dd"),... 
                                    1-control_study_structuralchanges_chi2cdf_vec(iii), control_study_structuralchanges(iii)};  
            iii = iii + 1;
        end            
        jjj = jjj + 1;
    end
    savenameout = strcat(dir_out, "forecast_variances_0_1_91_", name, "_updFeb24.mat");
    eval(strcat('save(',char(39), savenameout, char(39), ',"variance_forecast"', ',"control_study_structuralchanges"', ...
                             ',"control_study_structuralchanges_chi2cdf_vec"', ',"control_study_rejectnull"',...
                            ',"all_data",', ' "-v7")'));
end

end
