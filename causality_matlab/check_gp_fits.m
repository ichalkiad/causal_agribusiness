function [] = ...
         check_gp_fits(all_data,from_loop, to_loop,before_and_after_report_switch,...
                                          meanf, likf, name, lag, covf, ...
                                          dir_out, consider_win_point)
% change for cluster:
disp('Hi!');
time_now = now;
starting_time = floor(1.e+06 * rem(time_now,1));
starting_time_name = num2str(starting_time);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if nargin<3
    to_loop = 100;
    if nargin<2 
        from_loop = 1;
    end
end

length_data = size(all_data,1);
if nargin < 7
    lag = 1;
end

if nargin < 4
    meanfunc = {@meanLinear}; likfunc  = {@likGauss}; 
else
    switch meanf
        case {'lin', 'linear'}
            meanfunc = {@meanLinear};
        case {'poly', 'polynomial'}
            meanfunc = {@meanPoly,2};
        otherwise
            meanfunc = {@meanLinear};
    end
    if nargin < 5
    likfunc  = {@likGauss};
    else
        switch likf
            case {'Gauss', 'Gaus', 'Gaussian', 'likGauss'}
                likfunc  = {@likGauss}; 
                inffunc = @infExact;
                hypA.lik  = log(0); 
                hypB.lik  = log(0);
            case {'t-student', 'student-t', 't', 'likT'}
                likfunc = {@likT};
                inffunc = @infLaplace;
                hypA.lik  = [log(0); log(1)]; 
                hypB.lik  = [log(0); log(1)]; % for starting points...
            otherwise
                likfunc  = {@likGauss}; 
                inffunc = @infExact;
                hypA.lik  = log(0); 
                hypB.lik  = log(0);
        end
    end
end

%% 2. specify the model - here Matern kernel, additive Gaussian iid noise
% remember -- to have nested models we need ARD kernel!
% and also we need the fixed version of Matern

if nargin <9
    covfunc  = {@covSum, {{@covMaternard_modified,3}, @covNoise}};
else
    switch covf
        case 'Matern'
            covfunc = {@covSum, {{@covMaternard_modified,3}, @covNoise}};
        case {'poly', 'poly-cov'}
            covfunc = {@covSum, {{@covPoly,2}, @covNoise}};
        otherwise
            covfunc = {@covSum, {{@covMaternard_modified,3}, @covNoise}};
    end
end

if nargin < 10
    causal_placement_switch = 'all';
end


%%%
outname = strcat(dir_out, 'test1_optim_predm',meanf,'_',covf,'_',name,...
     '_len',num2str(length_data),...
     '_from',num2str(from_loop),'_to' ,num2str(to_loop),...
     '.mat')
eval(strcat('load(',char(39), outname, char(39), ',"hyperparameters"', ',"structuralchanges"', ',"structuralchanges_chi2cdf_vec"', ',"hyps_A_XY_vec"', ...
    ',"nlmlA_XY_multiruns"', ',"mean_predA_XY_vec"', ',"hypA_XY_best"', ',"nA_XY_best"' , ',"rejectnull")')); 
%%%
jjj = 1;
wins = 1;
for run_ii = 1:to_loop
    run_ii
    if (run_ii > 1) && (run_ii < consider_win_point)
        % fit only the first window (before report release) and windows
        % that overlap with the report release date
        continue
    end

    pppp = binornd(1, 0.05)
    if ~pppp
        continue
    end

    Data = all_data(:,run_ii);
    toXY = 1; sideXY = []; 
    inputA_XY = Data(1:end-lag,[toXY, sideXY]); 
    target_XY = Data(lag+1:end,toXY);
    try 
       inputA_XY_test = Data(end-lag+1,[toXY, sideXY]);
    catch
        disp("sfgsf")
    end

    % nlml is negative log likelihood, so need to put "-" in front
    if (run_ii >= before_and_after_report_switch)
        % eval likelihood of current data with pre-report model
        hypA_XY_first.mean = hyps_A_XY_vec(jjj, 1);
        hypA_XY_first.cov = hyps_A_XY_vec(jjj, 2:end); 
        hypA_XY_first.lik = nlmlA_XY_multiruns(jjj); 
%         nlmlA_XY_multiruns(1) = gp(hypA_XY_first, inffunc, meanfunc, covfunc, likfunc, inputA_XY, target_XY);

        sample_timeseries_gp_multistep(lag, inputA_XY, target_XY, hypA_XY_first, ...
                meanfunc, covfunc, length(inputA_XY));
         
        wins = wins + 1;
        pause(1)
        close
    end 
    jjj = jjj + 1;
end

hyperparameters{1} = hyps_A_XY_vec;

disp('Hej!')
end
