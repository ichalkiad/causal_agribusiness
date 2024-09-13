import time
import datetime
import ipdb
import pandas as pd
import pathlib
import numpy as np


if __name__ == '__main__':

    pwd = pathlib.Path.cwd()
    DIR = "{}/cbot_data/".format(pwd) 

    DIR_out = "{}/cbot_data_out/"
    pathlib.Path(DIR_out).mkdir(parents=True, exist_ok=True)

    cp = pd.read_csv("{}/cp_alldates.csv".format(DIR))["dates"].values.tolist()
    cpss = pd.read_csv("{}/cpss_alldates.csv".format(DIR))["dates"].values.tolist()
    gs = pd.read_csv("{}/gs_alldates.csv".format(DIR))["dates"].values.tolist()
    wasde = pd.read_csv("{}/wasde_alldates.csv".format(DIR))["dates"].values.tolist()
    cpm = pd.read_csv("{}/cpm_alldates.csv".format(DIR))["dates"].values.tolist()
    wws = pd.read_csv("{}/wws_alldates.csv".format(DIR))["dates"].values.tolist()
    pp = pd.read_csv("{}/pp_alldates.csv".format(DIR))["dates"].values.tolist()
    acr = pd.read_csv("{}/acr_alldates.csv".format(DIR))["dates"].values.tolist()
    cpas = pd.read_csv("{}/cpas_alldates.csv".format(DIR))["dates"].values.tolist()
    sgas = pd.read_csv("{}/sgas_alldates.csv".format(DIR))["dates"].values.tolist()
    
    # lambda function helpers
    matlab_date = lambda x: pd.Timestamp.strftime(x, "%d-%b-%Y %H:%M:%S")    
    daily = False
    hourly = True
    inputs = dict()
    if daily:
        timescale = "daily"
        inputs["corn_frontcontract_daily"] = "CBOT_DL_ZC1!, 1D.csv"
        inputs["corn_secondcontract_daily"] = "CBOT_DL_ZC2!, 1D.csv"
        inputs["wheat_frontcontract_daily"] = "CBOT_DL_ZW1!, 1D.csv"
        inputs["wheat_secondcontract_daily"] = "CBOT_DL_ZW2!, 1D.csv"
    elif hourly:
        timescale = "hourly"  
        inputs["corn_frontcontract_hourly"] = "CBOT_DL_ZC1!, 60_H.csv"
        inputs["corn_secondcontract_hourly"] = "CBOT_DL_ZC2!, 60_H.csv"
        inputs["wheat_frontcontract_hourly"] = "CBOT_DL_ZW1!, 60_H.csv"
        inputs["wheat_secondcontract_hourly"] = "CBOT_DL_ZW2!, 60_H.csv" 

    kwargs_list = [ "wheat", "corn" ]
    contracts   = { "wheat1": "ZW1",
                    "wheat2": "ZW2",
                    "corn1" : "ZC1",
                    "corn2" : "ZC2"
                  }    

    pd.options.mode.use_inf_as_na = True

    t0 = time.time()
    for top in kwargs_list:
        postprocessed_out_1 = dict()
        postprocessed_out_2 = dict()
        print("{}/{}_postproc_priceVol_{}.csv".format(DIR_out, top, timescale))

        DIR_tmp = "{}/CBOT_{}Futures_13062022/{}Futures_CBOT/".format(DIR, top, top)

        # price processes - close price
        # daily volatility diff of loghigh-loglow
        datain = pd.read_csv("{}/{}".format(DIR_tmp, inputs["{}_frontcontract_{}".format(top, timescale)]))
        times = datain.time
        tmplist = []            
        tmplist.extend(np.log(datain.high.values/datain.low.values).tolist())            

        postprocessed_out_1["{}_frontcontractRaw_{}".format(top, timescale)] = datain.close.values.tolist()
        postprocessed_out_1["{}_frontcontractIntraVolat_{}".format(top, timescale)] = tmplist
        tmplist = [np.nan]
        tmplist.extend(np.diff(np.log(postprocessed_out_1["{}_frontcontractIntraVolat_{}".format(top, timescale)])).tolist())
        postprocessed_out_1["{}_frontcontractIntraVolatDiff_{}".format(top, timescale)] = tmplist

        postprocessed_out_1["{}_frontcontractTradedVol_{}".format(top, timescale)] = datain.Volume
        postprocessed_out_1["{}_frontcontractTradedVolLog_{}".format(top, timescale)] = np.log(datain.Volume)
        tmplist = [np.nan]
        tmplist.extend(np.diff(postprocessed_out_1["{}_frontcontractTradedVolLog_{}".format(top, timescale)]).tolist())
        postprocessed_out_1["{}_frontcontractTradedVolLogDiff_{}".format(top, timescale)] = tmplist
        
        postprocessed_out_1["timestamp"] = times.values        
        postprocessed_df_1 = pd.DataFrame.from_dict(postprocessed_out_1)   
        postprocessed_df_1.timestamp = pd.to_datetime(postprocessed_df_1["timestamp"], utc=True)
        postprocessed_df_1.timestamp = postprocessed_df_1.timestamp.apply(matlab_date).apply(pd.Timestamp)
        postprocessed_df_1 = postprocessed_df_1.ffill()
        # drop first row to avoid NaNs/backfilling
        postprocessed_df_1 = postprocessed_df_1.dropna().reset_index(drop=True)
        
        ####################################
        datain = pd.read_csv("{}/{}".format(DIR_tmp, inputs["{}_secondcontract_{}".format(top, timescale)]))
        
        tmplist = []
        tmplist.extend(np.log(datain.high.values/datain.low.values).tolist())            
        postprocessed_out_2["{}_secondcontractRaw_{}".format(top, timescale)] = datain.close.values.tolist()
        
        postprocessed_out_2["{}_secondcontractIntraVolat_{}".format(top, timescale)] = tmplist
        tmplist = [np.nan]
        tmplist.extend(np.diff(np.log(postprocessed_out_2["{}_secondcontractIntraVolat_{}".format(top, timescale)])).tolist())
        postprocessed_out_2["{}_secondcontractIntraVolatDiff_{}".format(top, timescale)] = tmplist

        postprocessed_out_2["{}_secondcontractTradedVol_{}".format(top, timescale)] = datain.Volume
        postprocessed_out_2["{}_secondcontractTradedVolLog_{}".format(top, timescale)] = np.log(datain.Volume)
        tmplist = [np.nan]
        tmplist.extend(np.diff(postprocessed_out_2["{}_secondcontractTradedVolLog_{}".format(top, timescale)]).tolist())
        postprocessed_out_2["{}_secondcontractTradedVolLogDiff_{}".format(top, timescale)] = tmplist

        
        postprocessed_out_2["timestamp"] = datain.time
        postprocessed_df_2 = pd.DataFrame.from_dict(postprocessed_out_2)
        postprocessed_df_2.timestamp = pd.to_datetime(postprocessed_df_2["timestamp"], utc=True)
        postprocessed_df_2.timestamp = postprocessed_df_2.timestamp.apply(matlab_date).apply(pd.Timestamp)            
        postprocessed_df_2 = postprocessed_df_2.ffill()
        # drop first row to avoid NaNs/backfilling
        postprocessed_df_2 = postprocessed_df_2.dropna().reset_index(drop=True)
        ####################################

        postprocessed_df_1 = postprocessed_df_1.set_index(postprocessed_df_1["timestamp"])
        postprocessed_df_1 = postprocessed_df_1.drop(columns=["timestamp"])
        postprocessed_df_2 = postprocessed_df_2.set_index(postprocessed_df_2["timestamp"])
        postprocessed_df_2 = postprocessed_df_2.drop(columns=["timestamp"])

        ####################################
        if daily:
            postprocessed_df_1["CP"] = 0
            postprocessed_df_1.loc[np.in1d(postprocessed_df_1.index.date, pd.to_datetime(cp).date), "CP"] = 1            
            postprocessed_df_1["CPSS"] = 0
            postprocessed_df_1.loc[np.in1d(postprocessed_df_1.index.date, pd.to_datetime(cpss).date), "CPSS"] = 1
            postprocessed_df_1["ACR"] = 0
            postprocessed_df_1.loc[np.in1d(postprocessed_df_1.index.date, pd.to_datetime(acr).date), "ACR"] = 1
            postprocessed_df_1["PP"] = 0
            postprocessed_df_1.loc[np.in1d(postprocessed_df_1.index.date, pd.to_datetime(pp).date),"PP"] = 1
            postprocessed_df_1["CPAS"] = 0
            postprocessed_df_1.loc[np.in1d(postprocessed_df_1.index.date, pd.to_datetime(cpas).date),"CPAS"] = 1
            postprocessed_df_1["GS"] = 0
            postprocessed_df_1.loc[np.in1d(postprocessed_df_1.index.date, pd.to_datetime(gs).date), "GS"] = 1
            postprocessed_df_1["WASDE"] = 0
            postprocessed_df_1.loc[np.in1d(postprocessed_df_1.index.date, pd.to_datetime(wasde).date), "WASDE"] = 1
            
            postprocessed_df_2["CP"] = 0
            postprocessed_df_2.loc[np.in1d(postprocessed_df_2.index.date, pd.to_datetime(cp).date), "CP"] = 1
            postprocessed_df_2["CPSS"] = 0
            postprocessed_df_2.loc[np.in1d(postprocessed_df_2.index.date, pd.to_datetime(cpss).date), "CPSS"] = 1
            postprocessed_df_2["ACR"] = 0
            postprocessed_df_2.loc[np.in1d(postprocessed_df_2.index.date, pd.to_datetime(acr).date), "ACR"] = 1
            postprocessed_df_2["PP"] = 0
            postprocessed_df_2.loc[np.in1d(postprocessed_df_2.index.date, pd.to_datetime(pp).date), "PP"] = 1
            postprocessed_df_2["CPAS"] = 0
            postprocessed_df_2.loc[np.in1d(postprocessed_df_2.index.date, pd.to_datetime(cpas).date), "CPAS"] = 1
            postprocessed_df_2["GS"] = 0
            postprocessed_df_2.loc[np.in1d(postprocessed_df_2.index.date, pd.to_datetime(gs).date), "GS"] = 1
            postprocessed_df_2["WASDE"] = 0
            postprocessed_df_2.loc[np.in1d(postprocessed_df_2.index.date, pd.to_datetime(wasde).date), "WASDE"] = 1
            
            if top == "wheat":
                postprocessed_df_1["SGAS"] = 0
                postprocessed_df_1.loc[np.in1d(postprocessed_df_1.index.date, pd.to_datetime(sgas).date),"SGAS"] = 1                
                postprocessed_df_1["WWS"] = 0
                postprocessed_df_1.loc[np.in1d(postprocessed_df_1.index.date, pd.to_datetime(wws).date), "WWS"] = 1                            
                postprocessed_df_1["CPM"] = 0
                postprocessed_df_1.loc[np.in1d(postprocessed_df_1.index.date, pd.to_datetime(cpm).date), "CPM"] = 1                
                postprocessed_df_2["SGAS"] = 0
                postprocessed_df_2.loc[np.in1d(postprocessed_df_2.index.date, pd.to_datetime(sgas).date), "SGAS"] = 1
                postprocessed_df_2["WWS"] = 0
                postprocessed_df_2.loc[np.in1d(postprocessed_df_2.index.date, pd.to_datetime(wws).date), "WWS"] = 1                
                postprocessed_df_2["CPM"] = 0
                postprocessed_df_2.loc[np.in1d(postprocessed_df_2.index.date, pd.to_datetime(cpm).date), "CPM"] = 1
        elif hourly:
            postprocessed_df_1["CP"] = 0
            postprocessed_df_1.loc[
                    np.in1d(postprocessed_df_1.index, pd.to_datetime(cp)+np.timedelta64(16,'h')), "CP"] = 1
            
            postprocessed_df_1["CPSS"] = 0
            postprocessed_df_1.loc[np.in1d(postprocessed_df_1.index, pd.to_datetime(cpss)+np.timedelta64(16,'h')), "CPSS"] = 1

            postprocessed_df_1["ACR"] = 0
            postprocessed_df_1.loc[np.in1d(postprocessed_df_1.index, pd.to_datetime(acr)+np.timedelta64(12,'h')), "ACR"] = 1

            postprocessed_df_1["PP"] = 0
            postprocessed_df_1.loc[np.in1d(postprocessed_df_1.index, pd.to_datetime(pp)+np.timedelta64(12,'h')), "PP"] = 1

            postprocessed_df_1["CPAS"] = 0
            postprocessed_df_1.loc[np.in1d(postprocessed_df_1.index, pd.to_datetime(cpas)+np.timedelta64(12,'h')), "CPAS"] = 1

            postprocessed_df_1["GS"] = 0
            postprocessed_df_1.loc[np.in1d(postprocessed_df_1.index, pd.to_datetime(gs)+np.timedelta64(12,'h')), "GS"] = 1
            
            postprocessed_df_1["WASDE"] = 0
            postprocessed_df_1.loc[np.in1d(postprocessed_df_1.index, pd.to_datetime(wasde)+np.timedelta64(12,'h')), "WASDE"] = 1
            
            postprocessed_df_2["CP"] = 0
            postprocessed_df_2.loc[np.in1d(postprocessed_df_2.index, pd.to_datetime(cp) + np.timedelta64(16, 'h')), "CP"] = 1
            
            postprocessed_df_2["CPSS"] = 0
            postprocessed_df_2.loc[np.in1d(postprocessed_df_2.index, pd.to_datetime(cpss) + np.timedelta64(16, 'h')), "CPSS"] = 1

            postprocessed_df_2["ACR"] = 0
            postprocessed_df_2.loc[np.in1d(postprocessed_df_2.index, pd.to_datetime(acr) + np.timedelta64(12, 'h')), "ACR"] = 1

            postprocessed_df_2["PP"] = 0
            postprocessed_df_2.loc[np.in1d(postprocessed_df_2.index, pd.to_datetime(pp) + np.timedelta64(12, 'h')), "PP"] = 1

            postprocessed_df_2["CPAS"] = 0
            postprocessed_df_2.loc[np.in1d(postprocessed_df_2.index, pd.to_datetime(cpas) + np.timedelta64(12, 'h')), "CPAS"] = 1

            postprocessed_df_2["GS"] = 0
            postprocessed_df_2.loc[np.in1d(postprocessed_df_2.index, pd.to_datetime(gs) + np.timedelta64(12, 'h')), "GS"] = 1
            
            postprocessed_df_2["WASDE"] = 0
            postprocessed_df_2.loc[np.in1d(postprocessed_df_2.index, pd.to_datetime(wasde) + np.timedelta64(12, 'h')), "WASDE"] = 1

            if top == "wheat":
                postprocessed_df_1["SGAS"] = 0
                postprocessed_df_1.loc[np.in1d(postprocessed_df_1.index, pd.to_datetime(sgas) + np.timedelta64(12, 'h')), "SGAS"] = 1

                postprocessed_df_1["WWS"] = 0
                postprocessed_df_1.loc[np.in1d(postprocessed_df_1.index, pd.to_datetime(wws) + np.timedelta64(12, 'h')), "WWS"] = 1
                
                postprocessed_df_1["CPM"] = 0
                postprocessed_df_1.loc[np.in1d(postprocessed_df_1.index, pd.to_datetime(cpm) + np.timedelta64(12, 'h')), "CPM"] = 1

                postprocessed_df_2["SGAS"] = 0
                postprocessed_df_2.loc[np.in1d(postprocessed_df_2.index, pd.to_datetime(sgas) + np.timedelta64(12, 'h')), "SGAS"] = 1

                postprocessed_df_2["WWS"] = 0
                postprocessed_df_2.loc[np.in1d(postprocessed_df_2.index, pd.to_datetime(wws) + np.timedelta64(12, 'h')), "WWS"] = 1
                
                postprocessed_df_2["CPM"] = 0
                postprocessed_df_2.loc[np.in1d(postprocessed_df_2.index, pd.to_datetime(cpm) + np.timedelta64(12, 'h')), "CPM"] = 1
            
        postprocessed_df_1.to_csv("{}/{}_postproc_boller_{}_{}.csv".format(DIR_out, top, timescale, contracts[top + "1"]), 
                            index_label="timestamp")
        postprocessed_df_2.to_csv("{}/{}_postproc_boller_{}_{}.csv".format(DIR_out, top, timescale, contracts[top + "2"]),
                            index_label="timestamp")

    t1 = time.time()
    print("Time to completion: " + str(datetime.timedelta(seconds=t1-t0)))
