addpath(genpath('./causality_matlab/'));
addpath(genpath('./cbot_data/'));
addpath(genpath('./gpmlv3.5/'));


% delete(gcp('nocreate'));
rng('default');
timescale = 'daily';
lags = [1, 3, 5]; 

parallelrun = 1;
if parallelrun
    localcluster = parcluster("local");
    localcluster.NumWorkers = 4;
    saveProfile(localcluster);
    parpool("local", 4)
end

%% causality options
meanf = 'lin';
likf  = 'Gauss';
covf  = 'Matern';

jj = 1;
for dictionary=["usdacmecftc", "customdict"]
    for commodity=["wheat","corn"]
        for contract_no=["front","second"]          
            input_file = strcat(commodity, "_postproc_priceVol_", timescale);
            if contract_no == "front" && commodity == "corn"
                input_file = strcat(input_file, "_ZC1.csv");
            elseif contract_no == "second" && commodity == "corn"
                input_file = strcat(input_file, "_ZC2.csv");
            elseif contract_no == "front" && commodity == "wheat"
                input_file = strcat(input_file, "_ZW1.csv");
            elseif contract_no == "second" && commodity == "wheat"
                input_file = strcat(input_file, "_ZW2.csv");
            end            
            dir_out = strcat("./results_out_causality_Sep2023/", commodity, "/", contract_no, "/", timescale, "/", dictionary, "/");
            mkdir(dir_out);
            addpath(genpath("./results_out_causality_Sep2023/"));
            % Extract report dates and convert to strings of date
            datain = load_findata_vol(input_file, [2, Inf], commodity, contract_no, timescale);    
            % Load sentiment file
            sentiment = load_sentiment(strcat('./cbot_data/DowJonesNews_', commodity, '_carry_fwd_alldictssentiment.csv'));
            sent_dates = sentiment.Dates;
            sentiment_ts = sentiment.(dictionary);
                            
            st = strcat(timescale, "_", commodity, "_", contract_no);   
            for if_caus_in_mean=[1]
                if if_caus_in_mean                                       
                    causal_placement_switch = 'mean'
                else
                    causal_placement_switch = 'meancov'
                end
                if parallelrun
                    F(jj) = parfeval(@exec_script_upd_causality_total, 0, ...
                                        sent_dates, sentiment_ts, timescale, ...
                                        commodity, contract_no, input_file, ...
                                        lags, meanf, likf, covf, dir_out, ...
                                        causal_placement_switch);
                    jj = jj + 1;
                else                    
                    exec_script_upd_causality_total(sent_dates, sentiment_ts, timescale, ...
                                                    commodity, contract_no, input_file, ...
                                                    lags, meanf, likf, covf, dir_out, ...
                                                    causal_placement_switch);   
                end
            end                                
        end
    end
end

