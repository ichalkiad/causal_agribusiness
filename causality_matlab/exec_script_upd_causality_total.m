function [] = exec_script_upd_causality_total(sent_dates, sentiment_ts, timescale, ...
                                                commodity, contract_no, input_file, ...
                                                lags, meanf, likf, covf, dir_out, ...
                                                causal_placement_switch)
    
    datain = load_findata_vol(input_file, [2, Inf], commodity, contract_no, timescale);
    dates_only = dateshift(datain.("timestamp"), "start", "day");
    
    % convert report dates from strings to datetimes
    fin_dates = datetime(datestr(dateshift(datain.timestamp, "start", "day"), "yyyy-mm-dd"));
    [sharedVals,idxsIntoSentDates] = intersect(sent_dates, fin_dates, 'stable');
    sent_idx = idxsIntoSentDates;
    sent_dates = sent_dates(sent_idx);
    sent_ts = sentiment_ts(sent_idx);

    [sharedVals,idxsIntoDataIn] = intersect(fin_dates, sent_dates, 'stable');
    datain = datain(idxsIntoDataIn, :);
    fin_dates = fin_dates(idxsIntoDataIn);
    ppr = 1;
    
    priceret = datain.(strcat(commodity,"_", contract_no,"contractLogRet_", timescale));
    priceraw = datain.(strcat(commodity,"_", contract_no,"contractRaw_", timescale));
    % volume
    volume = datain.(strcat(commodity,"_tradedvolmedian7",contract_no,"contract_", timescale));
    % realised Volatility - Parkinsons 
    realvol = datain.(strcat(commodity,"_parkinsonVolatility_", timescale));
    vol100 = datain.(strcat(commodity,"_logretVolatility100_", timescale));
    vol7 = datain.(strcat(commodity,"_logretVolatility7_", timescale));

    report_dates = datestr(datain.timestamp, "yyyy-mm-dd");
    report_dates = datetime(report_dates);

    window_size = 91;
    idxxx = 1;
    k = 1;
    clear Data_cuts;
    clear data_cut_init_time;   
    
    while idxxx + window_size < min([length(priceraw),length(priceret),length(volume),length(realvol),length(sent_ts)])

        win_data_price = priceraw(idxxx:idxxx+window_size);
        win_data_priceret = priceret(idxxx:idxxx+window_size);
        win_data_vol = volume(idxxx:idxxx+window_size);
        win_data_realvol = realvol(idxxx:idxxx+window_size);
        win_data_vol100 = vol100(idxxx:idxxx+window_size);
        win_data_vol7 = vol7(idxxx:idxxx+window_size);
        win_sent = sent_ts(idxxx:idxxx+window_size);
        
        % zscored data
        win_data_zscore_price = zscore(win_data_price);    
        win_data_zscore_priceret = zscore(win_data_priceret);
        win_data_zscore_vol = zscore(log(win_data_vol));    
        win_data_zscore_realvol = zscore(win_data_realvol);    
        win_data_zscore_logrealvol = zscore(log(win_data_realvol));
        win_data_zscore_logvol100 = zscore(log(win_data_vol100));
        win_data_zscore_logvol7 = zscore(log(win_data_vol7));    
        win_sent_zscore = zscore(win_sent);

        if sum(isnan(win_data_zscore_price))>1 || sum(isnan(win_sent_zscore))>1 ||...
                sum(isnan(win_data_zscore_priceret))>1 || sum(isnan(win_data_zscore_vol))>1 || sum(isnan(win_data_zscore_realvol))>1 ...
                sum(isnan(win_data_zscore_logrealvol))>1 
            idxxx = idxxx + 5;  
            disp("NAN in input - shifting window")
            continue            
        end

        Data_cuts(:,:,:,:,:,k) = [win_data_zscore_price, win_data_zscore_priceret, win_data_zscore_vol, win_data_zscore_realvol,...
                                    win_sent_zscore, win_data_zscore_logrealvol, win_data_zscore_logvol100, win_data_zscore_logvol7];
        data_cut_init_time(k,:) = [datetime(report_dates(idxxx)), datetime(report_dates(idxxx+window_size))];
        k = k + 1;
        % 1 business-week step
        idxxx = idxxx + 5;  
    end
    
    % eval(strcat('save(',char(39), strcat(dir_out,"dates"), char(39), ',"data_cut_init_time",', ' "-v7")'));
    from_loop = 1;
    to_loop = k-1;
    floor_switch = true;        
    for lag=lags
        name = strcat(commodity, '_', contract_no, '_lag', num2str(lag), '_meanchi_report_');
        test_for_statistical_causality_total(Data_cuts, from_loop, to_loop,...
                                                  meanf, likf, name, lag, floor_switch, covf, ...
                                                  dir_out, data_cut_init_time, causal_placement_switch);
        % check_gp_causal_fits(Data_cuts,from_loop, to_loop,...
        %                                 meanf, likf, name, lag, floor_switch,covf, ...
        %                                 dir_out, data_cut_init_time)
    end
    progress(ppr, :) = name;
    disp(progress)
    ppr = ppr + 1;
    eval(strcat('save(',char(39), strcat("causality_", strcat(commodity, '_', contract_no, ...
                '_meanchi_report__')), char(39), ',"progress", "Data_cuts", "data_cut_init_time",', ' "-v7")'));
end