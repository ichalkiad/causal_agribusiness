import pickle
import time
import logging
import datetime
import ipdb
import pandas as pd
import pathlib
import numpy as np
import plotly.graph_objects as go
from src.text_utils import fix_plot_layout_and_save

def set_period_corn(x): 
        
    if pd.to_datetime(x).date().month in [4,5]:
        return "planting"
    elif pd.to_datetime(x).date().month in [8,9]:
        return "harvest"
    elif pd.to_datetime(x).date().month in [6,7]:
        return "grow"
    elif pd.to_datetime(x).date().month in [10,11,12,1,2,3]:
        return "rest"        

def set_period_wheat(x): 
        
    if pd.to_datetime(x).date().month in [9,10]:
        return "planting"
    elif pd.to_datetime(x).date().month in [6,7,8]:
        return "harvest"
    elif pd.to_datetime(x).date().month in [3,4,5]:
        return "grow"
    elif pd.to_datetime(x).date().month in [11,12,1,2]:
        return "rest"  


if __name__ == '__main__':

    symbols = {"0.6": "cross",
                "0.7": "x",
                "0.8": "circle",
                "0.9": "triangle-up"}

    colors = {"0.6": "#1f77b4",
                "0.7": "#ff7f0e",
                "0.8": "#2ca02c",
                "0.9": "#d62728"}    


    pwd = pathlib.Path.cwd()
    DIR_out = "{}/cbot_data_out/".format(pwd)
    DIRvol = DIR_out    
    pathlib.Path(DIR_out).mkdir(parents=True, exist_ok=True)
    
    dictionary = "customdict" #"usdacmecftc"
    ret_and_vol = 1
    smooth = 3
    window_around_event = 10 + smooth
    daily = True
    hourly = False
    if daily:
        timescale = "daily"
    elif hourly:
        timescale = "hourly"

    kwargs_list = ["wheat", "corn"] 
    contracts   = {"wheat front": "ZW1",
                   "wheat second": "ZW2",
                   "corn front" : "ZC1",
                   "corn second" : "ZC2"}
    t0 = time.time()
    def to_date(x) : return pd.to_datetime(x).date()    
    medianvaly1 = []
    medianvaly2 = []
    periodq = []


    for top in kwargs_list:
        if top == "wheat":
            reports = ["PP", "ACR", "CPAS", "GS", "CPM", "WASDE", "SGAS", "CPSS", "WWS"]
        elif top == "corn":
            reports = ["PP", "WASDE", "ACR", "CPAS", "GS", "CPSS"]
        
        for contract in ["front", "second"]:

            datain = pd.read_csv("{}/{}_postproc_priceVol_{}_{}.csv".format(DIR_out, top, timescale, contracts["{} {}".format(top, contract)]))            
            datain.timestamp = datain.timestamp.apply(to_date)            

            datavol = pd.read_csv("{}/{}_postproc_boller_{}_{}.csv".format(DIRvol, top, timescale, contracts["{} {}".format(top, contract)]))
            datavol = datavol.ffill()
            datavol.timestamp = datavol.timestamp.apply(to_date)            

            sentiment = pd.read_csv("{}/DowJonesNews_{}_carry_fwd_alldictssentiment.csv".format(DIR_out, top))
            sentiment.Dates = sentiment.Dates.apply(to_date)

            # get dates for sentiment data
            dateswithsentiment = datain[datain.timestamp.isin(sentiment.Dates)].reset_index(drop=True)
            voldateswithsentiment = datavol[datavol.timestamp.isin(sentiment.Dates)].reset_index(drop=True)
            sentiment_clean = sentiment[sentiment.Dates.isin(dateswithsentiment.timestamp)].reset_index(drop=True)
            
            dateswithsentiment["sent_quantile"] = 0
            for rep in reports:
                # Get report dates
                dateswithsentiment["sent_quantile"] += dateswithsentiment[rep]
            
            index2use = np.array(dateswithsentiment.index[dateswithsentiment["sent_quantile"] > 0])          
            # indices of extreme events
            index2use = index2use[index2use >= window_around_event]
            index2use = index2use[index2use <= index2use.tolist()[-1] - window_around_event]
            extreme_val_window_usda = []
            extreme_val_window_custom = []      
            extreme_val_window_usda_above_q_percentage = dict()
            extreme_val_window_custom_above_q_percentage = dict()                        
            if ret_and_vol:
                extreme_val_window_logRet = []
                extreme_val_window_vol = []                          
            for i in index2use:
                indexrange = np.arange(i-window_around_event, i+window_around_event+1, 1).tolist()                                                            
                for q in [0.6, 0.7, 0.8, 0.9]:
                    sent_quantile_usda = sentiment_clean["usdacmecftc"].quantile(q)
                    sss = sentiment_clean.loc[indexrange, "usdacmecftc"].values
                    sssq = sss[sss > sent_quantile_usda]
                    if len(sssq) > 0:
                        if q in extreme_val_window_usda_above_q_percentage.keys():
                            extreme_val_window_usda_above_q_percentage[q].append(len(sssq)/len(sss))
                        else:
                            extreme_val_window_usda_above_q_percentage[q] = [len(sssq)/len(sss)]
                    sent_quantile_custom = sentiment_clean["customdict"].quantile(q)
                    sss = sentiment_clean.loc[indexrange, "customdict"].values
                    sssq = sss[sss > sent_quantile_custom]
                    if len(sssq) > 0:
                        if q in extreme_val_window_custom_above_q_percentage.keys():
                            extreme_val_window_custom_above_q_percentage[q].append(len(sssq)/len(sss))
                        else:
                            extreme_val_window_custom_above_q_percentage[q] = [len(sssq)/len(sss)]
            
            fig_usda_box = go.Figure()
            fig_custom_box = go.Figure()
            for q in [0.6, 0.7, 0.8, 0.9]:
                percentage_above_quantile = extreme_val_window_usda_above_q_percentage[q]
                fig_usda_box.add_trace(go.Box(y=percentage_above_quantile,
                                                boxpoints='all', 
                                                jitter=0.3,
                                                pointpos=-1.8,
                                                name = "Quantile: {:.1f}".format(q)
                                                ))
                percentage_above_quantile = extreme_val_window_custom_above_q_percentage[q]
                fig_custom_box.add_trace(go.Box(y=percentage_above_quantile,
                                                boxpoints='all', 
                                                jitter=0.3,
                                                pointpos=-1.8,
                                                name = "Quantile: {:.1f}".format(q)
                                                ))
            
            savename = "{}/sentiment_aroundallreports_smooth{}_May2024/announcesentiment_{}_{}_{}_customdict_box.html".format(DIR_out, smooth, top, contract, timescale)                
            fix_plot_layout_and_save(fig_custom_box, savename, xaxis_title="", 
                                    yaxis_title="Percentage of window days with sentiment<br>(custom Agribusiness dictionary)<br>value above given quantile level", 
                                    title="", showgrid=False, showlegend=True,
                                    print_png=True)                   
            savename = "{}/sentiment_aroundallreports_smooth{}_May2024/announcesentiment_{}_{}_{}_usdadict_box.html".format(DIR_out, smooth, top, contract, timescale)                
            fix_plot_layout_and_save(fig_usda_box, savename, xaxis_title="", 
                                    yaxis_title="Percentage of window days with sentiment<br>(USDA technical dictionary)<br>value above given quantile level", 
                                    title="", showgrid=False, showlegend=True,
                                    print_png=True)                                                               
            # fig_usda_box.show()
            # fig_custom_box.show()
            

    t1 = time.time()
    print("Time to completion: " + str(datetime.timedelta(seconds=t1-t0)))
