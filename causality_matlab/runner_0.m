addpath(genpath('./causality_matlab/'));
addpath(genpath('./cbot_data/'));   
addpath(genpath('./gpmlv3.5/'));

%delete(gcp('nocreate'));
parallelrun = 1;
if parallelrun
    localcluster = parcluster("local");
    localcluster.NumWorkers = 8;
    saveProfile(localcluster);
    parpool("local", 8)
end
%% Save data switch
do_save = 1;
%% GP fit switch
do_fit = 0;

%% causality options
meanf = 'lin';
likf  = 'Gauss';
covf  = 'Matern';

jj = 1;
for commodity=["wheat", "corn"] 
    for contract_no=["front", "second"] 
        if commodity == "wheat"
            reports = ["WASDE", "SGAS", "GS", "PP", "ACR", "CPAS", "CPM", "CPSS", "WWS"];
        elseif commodity == "corn"
            reports = ["WASDE", "GS", "PP", "ACR", "CPAS", "CPSS"];
        end
        for if_caus_in_mean=[0]  
            if if_caus_in_mean
                where_cause = 'mean';
                name_cause  = 'mean';
            else
                where_cause = 'all';
                name_cause = 'meancov';
            end            
            for report_idx=1:length(reports)
                report_name = reports(report_idx);
                switch report_name                    
                    case {"CPSS", "WASDE", "CPM", "GS"}
                        analysis_timeframe = "monthly";
                        timescale = "hourly";
                        % shift window 1h  - was 3h 
                        stepshift = 1;
                    case {"PP", "ACR", "CPAS", "WWS", "SGAS"}
                        analysis_timeframe = "annual";
                        timescale = "daily";
                        % shift window 1 day
                        stepshift = 1;
                end    
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
                dir_out = strcat("./results_structural_change_Sep2023/", commodity, ...
                                "/", contract_no, "/", timescale, "/", report_name , "/");
                mkdir(dir_out);            
                addpath(genpath("./results_structural_change_Sep2023/"));
            
                % Extract report dates and convert to strings of date
                datain = load_findata_vol(input_file, [2, Inf], commodity, contract_no, timescale);
                for signal_name=["volume", "RealVolatility", "PriceRaw", "PriceReturns"]
                    st = strcat(timescale, "_", commodity, "_", contract_no, "_", analysis_timeframe,...
                                "_", report_name, "_", signal_name)
                    % lags determination
                    if timescale == "daily"
                        report_dates = datestr(datain.timestamp(datain.(report_name)==1), "yyyy-mm-dd");
                        lags = [2,3,4,5,1]; 
                        % window extends up to 2 business months after previous report
                        winlen = 40;
                        % window reaches up to 1 week before current report
                        pastbarrier = 5;
                        % window reaches up to 30 days after current report
                        futurelookout = 30;                    
                    elseif timescale == "hourly"
                        report_dates = datestr(datain.timestamp(datain.(report_name)==1), "yyyy-mm-dd HH:00:00");
                        lags = [1,4,8]; 
                        % window extends up to 1 day after previous report - buffer prev.
                        winlen = 24;
                        % window reaches up to 12 hours before current report - buffer current
                        pastbarrier = 12;
                        % window reaches up to 12 hours after current report - was 12h
                        futurelookout = 12; 
                    end            
                    if parallelrun                                    
                        F(jj) = parfeval(@exec_script_upd, 0, timescale, ...
                            commodity, contract_no, report_dates, input_file, analysis_timeframe,...
                            report_name, signal_name, lags, winlen,...
                            pastbarrier, futurelookout, meanf, likf, covf, where_cause, name_cause,...
                            dir_out, stepshift, do_save, do_fit);
                        jj = jj + 1;
                    else
                        exec_script_upd(timescale, ...
                            commodity, contract_no, report_dates, input_file, analysis_timeframe,...
                            report_name, signal_name, lags, winlen,...
                            pastbarrier, futurelookout, meanf, likf, covf, where_cause, name_cause,...
                            dir_out, stepshift, do_save, do_fit);
                    end                           
                end
            end              
        end
    end
end
