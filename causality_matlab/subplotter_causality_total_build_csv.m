for commodity=["wheat","corn"] 
    commodity_table = table;
    i = 0;
    clear marketprocess;
    clear oneminusp;
    clear msentdirection;
    clear lagval;
    clear causalloc;
    clear contractval;
    clear lexicon;
    clear datesval;
    for dictionary=["customdict", "usdacmecftc"]        
        for causalityloc=["meancov", "mean"]        
            for contract=["front", "second"]            
                for lag=[1, 3, 5]
                    if lag==5 && causalityloc=="meancov"
                        continue
                    end
                    name = strcat(commodity, "_", contract, "_", causalityloc,"_", "lag", num2str(lag));                    
                    dir_out2 = strcat(pwd + "results_out_causality_Sep2023/", commodity, "/", contract, "/daily/", dictionary, "/");
                    filenam = dir(strcat(dir_out2, "/test1_optim_predm_", causalityloc, ...
                                            "_lin_Matern_", commodity, "_", contract, ...
                                            "_lag",num2str(lag),"_meanchi_report__from1_to*.mat"));
                    load(strcat(dir_out2,filenam.name));
                    % since init running saved mismatched dates - load data_cut_init_time
                    % load(strcat(dir_out2,"dates.mat"));
                    dir_out2 = strcat(pwd + "results_out_causality_Sep2023/studyII/");
                    mkdir(dir_out2);
                    lag = uint8(lag);
                    % market -> sentiment
                    rejectnullYXtable = cell2table(rejectnullYX);
                    rejectnullXYtable = cell2table(rejectnullXY);
                    window_starts = data_cut_init_time(:,1);
                    window_ends = data_cut_init_time(:,2);
                    xx = datetime(window_ends, 'Format', 'dd-MMM-yyyy');
                                       
                    % X -> Y, sentiment to market
                    price_gp1minusp = 1-rejectnullXYtable.rejectnullXY3;
                    priceret_gp1minusp = 1-rejectnullXYtable.rejectnullXY5;
                    vol_gp1minusp = 1-rejectnullXYtable.rejectnullXY7;
                    realvol_gp1minusp = 1-rejectnullXYtable.rejectnullXY9;
                    volrealvol_gp1minusp = 1-rejectnullXYtable.rejectnullXY11;
                    
                    for j=[1:1:length(xx)]
                        i = i + 1;
                        marketprocess(i) = "price";
                        oneminusp(i) = price_gp1minusp(j);
                        msentdirection(i) = "m2sent";
                        lagval(i) = lag;
                        causalloc(i) = causalityloc;
                        datesval(i) = xx(j);
                        contractval(i) = contract;
                        lexicon(i) = dictionary;
                        i = i + 1;
                        marketprocess(i) = "priceret";
                        oneminusp(i) = priceret_gp1minusp(j);
                        msentdirection(i) = "m2sent";
                        lagval(i) = lag;
                        causalloc(i) = causalityloc;
                        datesval(i) = xx(j);
                        contractval(i) = contract;
                        lexicon(i) = dictionary;
                        i = i + 1;
                        marketprocess(i) = "vol";
                        oneminusp(i) = vol_gp1minusp(j);
                        msentdirection(i) = "m2sent";
                        lagval(i) = lag;
                        causalloc(i) = causalityloc;
                        datesval(i) = xx(j);
                        contractval(i) = contract;
                        lexicon(i) = dictionary;
                        i = i + 1;
                        marketprocess(i) = "realvol";
                        oneminusp(i) = realvol_gp1minusp(j);
                        msentdirection(i) = "m2sent";
                        lagval(i) = lag;
                        causalloc(i) = causalityloc;
                        datesval(i) = xx(j);
                        contractval(i) = contract;
                        lexicon(i) = dictionary;
                        i = i + 1;
                        marketprocess(i) = "volrealvol";
                        oneminusp(i) = volrealvol_gp1minusp(j);
                        msentdirection(i) = "m2sent";
                        lagval(i) = lag;
                        causalloc(i) = causalityloc;
                        datesval(i) = xx(j);
                        contractval(i) = contract;
                        lexicon(i) = dictionary;
                    end

                    % Y -> X, market to sentiment
                    price_gp1minuspYX = 1-rejectnullYXtable.rejectnullYX3;
                    priceret_gp1minuspYX = 1-rejectnullYXtable.rejectnullYX5;
                    vol_gp1minuspYX = 1-rejectnullYXtable.rejectnullYX7;
                    realvol_gp1minuspYX = 1-rejectnullYXtable.rejectnullYX9;
                    volrealvol_gp1minuspYX = 1-rejectnullYXtable.rejectnullYX11;
                    
                    for j=[1:1:length(xx)]
                        i = i + 1;
                        marketprocess(i) = "price";
                        oneminusp(i) = price_gp1minuspYX(j);
                        msentdirection(i) = "sent2m";
                        lagval(i) = lag;
                        causalloc(i) = causalityloc;
                        datesval(i) = xx(j);
                        contractval(i) = contract;
                        lexicon(i) = dictionary;
                        i = i + 1;
                        marketprocess(i) = "priceret";
                        oneminusp(i) = priceret_gp1minuspYX(j);
                        msentdirection(i) = "sent2m";
                        lagval(i) = lag;
                        causalloc(i) = causalityloc;
                        datesval(i) = xx(j);
                        contractval(i) = contract;
                        lexicon(i) = dictionary;
                        i = i + 1;
                        marketprocess(i) = "vol";
                        oneminusp(i) = vol_gp1minuspYX(j);
                        msentdirection(i) = "sent2m";
                        lagval(i) = lag;
                        causalloc(i) = causalityloc;
                        datesval(i) = xx(j);
                        contractval(i) = contract;
                        lexicon(i) = dictionary;
                        i = i + 1;
                        marketprocess(i) = "realvol";
                        oneminusp(i) = realvol_gp1minuspYX(j);
                        msentdirection(i) = "sent2m";
                        lagval(i) = lag;
                        causalloc(i) = causalityloc;
                        datesval(i) = xx(j);
                        contractval(i) = contract;
                        lexicon(i) = dictionary;
                        i = i + 1;
                        marketprocess(i) = "volrealvol";
                        oneminusp(i) = volrealvol_gp1minuspYX(j);
                        msentdirection(i) = "sent2m";
                        lagval(i) = lag;
                        causalloc(i) = causalityloc;
                        datesval(i) = xx(j);
                        contractval(i) = contract;
                        lexicon(i) = dictionary;
                    end                    
                end
            end
        end
    end
    marketprocess = marketprocess';
    oneminusp = oneminusp';
    msentdirection = msentdirection';
    lagval = lagval';
    causalloc = causalloc';
    contractval = contractval';
    lexicon = lexicon';
    datesval = datestr(datesval', "dd yyyy mmm");
    commodity_table = table(marketprocess, oneminusp, msentdirection, lagval, ...
        causalloc, contractval, lexicon, datesval);
    sorted_commodity_table = sortrows(commodity_table, {'marketprocess', ...
        'msentdirection', 'causalloc', 'contractval', ...
        'lagval', 'datesval', 'lexicon'}); 
    writetable(sorted_commodity_table, strcat(dir_out2, commodity, ".csv"));
 end