function [hyp_param_out,nlmls] = get_starting_point_less_points_seed_version4NLP(mean_param_nr, cov_param_nr,...
    param_ranges,grid_nr,requested_out_nr, x_data, y_data, meanfunc, ...
    covfunc, likfunc,hyp_lik, inffunc, seed_add)

if nargin < 12
    inffunc = @infExact;
end
if nargin<13
    seed_add = 0;
end

% check the orientation of the param_ranges
% the param ranges should be eg: 
param_nr = mean_param_nr + cov_param_nr;
if not(param_nr == size(param_ranges,1))
    error('Number of parameters and parameter ranges are inconsistent.')
end

if grid_nr <= 2
    grid_nr_upd = 2;
else
    grid_nr_upd = grid_nr;
end

% linspaceNDim needed - it's the same as linspace, but works on vectors
same_nr   = find(param_ranges(:,1)==param_ranges(:,2)); % no ranges here
normal_nr = find(param_ranges(:,1) <param_ranges(:,2)); % here - ok

param_ranges_altered = param_ranges(normal_nr,:);
param_nr_altered = length(normal_nr) % count only those for which ranges are indeed produced
hyp_param_grid = linspaceNDim(param_ranges_altered(:,1),param_ranges_altered(:,2), grid_nr_upd);


% this table will contain hyperparameters and the nlml functions
hyp_param_all = zeros(param_nr+1,grid_nr_upd^param_nr_altered);
cell_hyp_param = num2cell(hyp_param_grid',[1,param_nr_altered]);

hyp_param_all(normal_nr,1:grid_nr_upd^param_nr_altered) = allcomb(cell_hyp_param{:})';
% note!!! problem if for example some parameter will only have 1 value
% because it's also put on a grid grid_nr times, and we have too many
% calculations!


hyp_param_all(same_nr,:)=repmat(param_ranges(same_nr,1),1,grid_nr_upd^param_nr_altered);
% here get the param_ranges(same_nr,1) multiplied, as they are still used,
% just don't increase the number of combinations

hyperparameters.lik = hyp_lik;
for comb_ii = 1:round(grid_nr^param_nr_altered) % keep grid_nr here, not nr_upd so that we assign to subset
    %calculate some of the nlmls (randomly assign -Inf to half)
    seed = comb_ii + seed_add;
    rand('seed', seed);
    r = rand;
    if r > 0.5
        hyperparameters.mean = hyp_param_all(1:mean_param_nr,comb_ii);
        hyperparameters.cov  = hyp_param_all(mean_param_nr+1:param_nr,comb_ii);
        hyp_param_all(end,comb_ii) = gp(hyperparameters, inffunc, meanfunc, ...
            covfunc, likfunc, x_data, y_data); 
    else
        hyp_param_all(end,comb_ii) = Inf;
    end    
end

% all parameter pairs
[~,indices] = sort(hyp_param_all(end,:)); % increasing
if strcmp(requested_out_nr,'all')
    hyp_param_out = hyp_param_all(1:end-1,indices);
    nlmls = hyp_param_all(end,indices);
else
    hyp_param_out = hyp_param_all(1:end-1,indices((end-round(grid_nr^param_nr_altered)+1):((end-round(grid_nr^param_nr_altered)+1)+requested_out_nr))); % was indices(1:requested_out_nr)
    nlmls = hyp_param_all(end,indices((end-round(grid_nr^param_nr_altered)+1):((end-round(grid_nr^param_nr_altered)+1)+requested_out_nr))); % was indices(1:requested_out_nr)
end
