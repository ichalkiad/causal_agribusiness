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
from plotly.subplots import make_subplots

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
    
    ret_and_vol = 1
    smooth = 3
    window_around_event = 10 + smooth
    daily = True
    hourly = False
    if daily:
        timescale = "daily"
    elif hourly:
        timescale = "hourly"

    kwargs_list = ["corn", "wheat"] 
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
            reports = ["ACR", "PP", "CPAS", "GS", "CPM", "WASDE", "SGAS", "CPSS", "WWS"]
        elif top == "corn":
            reports = ["ACR", "PP", "WASDE", "CPAS", "GS", "CPSS"]
        
        for contract in ["front", "second"]:

            datain = pd.read_csv("{}/{}_postproc_priceVol_{}_{}.csv".format(DIR_out, top, timescale, contracts["{} {}".format(top, contract)]))
            # datain = datain.ffill()
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
             
            for rep in reports:
                # Get report dates
                sent_quantile = dateswithsentiment.loc[dateswithsentiment[rep] == 1]
                index2use = np.array(sent_quantile.index)
                # indices of extreme events
                index2use = index2use[index2use >= window_around_event]
                index2use = index2use[index2use <= index2use.tolist()[-1] - window_around_event]
                extreme_val_window_usda = []
                extreme_val_window_custom = []                
                if ret_and_vol:
                    extreme_val_window_logRet = []
                    extreme_val_window_vol = []   
                extreme_val_window_usda_above_q_percentage = dict()
                extreme_val_window_custom_above_q_percentage = dict()              
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
                    
                    # usda         
                    vals_usda = np.diff(sentiment_clean.loc[indexrange, "usdacmecftc"].values)
                    if vals_usda[0] == 0:                        
                        continue       
                    else:
                        cumret = 100*(np.cumsum(vals_usda) - vals_usda[0])/vals_usda[0]
                    extreme_val_window_usda.append(cumret)
                    
                    # custom
                    vals_custom = np.diff(sentiment_clean.loc[indexrange, "customdict"].values)                    
                    if vals_custom[0] == 0:
                        continue
                    else:
                        cust = 100*(np.cumsum(vals_custom) - vals_custom[0])/vals_custom[0]
                    extreme_val_window_custom.append(cust)

                    if ret_and_vol:
                        # returns                        
                        vals_logRet = dateswithsentiment.loc[indexrange, "{}_{}contractLogRet_daily".format(top, contract)].values
                        if vals_logRet[0] == 0:                            
                            if vals_logRet[1] == 0:                            
                                continue
                            else:
                                cumret = [0]
                                cumret.extend((100*(np.cumsum(vals_logRet[1:]) - vals_logRet[1])/vals_logRet[1]).tolist())
                        else:
                            cumret = 100*(np.cumsum(vals_logRet) - vals_logRet[0])/vals_logRet[0]
                        extreme_val_window_logRet.append(cumret)                        
                        # vol
                        vals_vol = voldateswithsentiment.loc[indexrange, "{}_{}contractTradedVolLogDiff_daily".format(top, contract)].values
                        if vals_vol[0] == 0:                            
                            if vals_vol[1] == 0:                            
                                continue
                            else:
                                custvol = [0]
                                custvol.extend((100*(np.cumsum(vals_vol[1:]) - vals_vol[1])/vals_vol[1]).tolist())
                        else:
                            custvol = 100*(np.cumsum(vals_vol) - vals_vol[0])/vals_vol[0]
                        extreme_val_window_vol.append(custvol)
                
                allusda = np.vstack(extreme_val_window_usda)
                allcustom = np.vstack(extreme_val_window_custom)                
                print(allusda.shape)
                y1 = np.mean(allusda, 0)
                y2 = np.mean(allcustom, 0)                
                if ret_and_vol:
                    alllogret = np.vstack(extreme_val_window_logRet)
                    allvol = np.vstack(extreme_val_window_vol)                    
                    y3 = np.mean(alllogret[:, 1:], 0) # shift due to differencing the sentiment
                    y4 = np.mean(allvol[:, 1:], 0)                    
                    dfplot = pd.DataFrame.from_dict({"usda":y1, "custom":y2, "logret": y3, "logretvol": y4})                                                       
                    dfplot["logret"] = dfplot["logret"].rolling(smooth).mean()
                    dfplot["logretvol"] = dfplot["logretvol"].rolling(smooth).mean()
                    dfplot["usdasmooth"] = dfplot["usda"].rolling(smooth).mean()
                    dfplot["customsmooth"] = dfplot["custom"].rolling(smooth).mean()                    
                    if sum(np.isnan(y1)) or sum(np.isnan(y2)) or sum(np.isnan(y3)) or sum(np.isnan(y4)):
                        ipdb.set_trace()
                else:
                    dfplot = pd.DataFrame.from_dict({"usda":y1, "custom":y2})
                    dfplot["usdasmooth"] = dfplot["usda"].rolling(smooth).mean()
                    dfplot["customsmooth"] = dfplot["custom"].rolling(smooth).mean()
                    if sum(np.isnan(y1)) or sum(np.isnan(y2)):
                        ipdb.set_trace()
                    
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
                savename = "{}/sentiment_aroundreports_smooth{}_May2024/announcesentiment_{}_{}_{}_{}_usdacmecftc_size_{}_box.html".format(DIR_out, smooth, 
                                                                                                            top, contract, rep, timescale, allusda.shape[0])
                pathlib.Path("{}/sentiment_aroundreports_smooth{}_May2024/".format(DIR_out, smooth)).mkdir(parents=True, exist_ok=True)               
                
                fix_plot_layout_and_save(fig_usda_box, savename, xaxis_title="", 
                                        yaxis_title="Percentage of window days with sentiment<br>(USDA technical dictionary)<br>value above given quantile level", 
                                        title="", showgrid=False, showlegend=True,
                                        print_png=True)
                savename = "{}/sentiment_aroundreports_smooth{}_May2024/announcesentiment_{}_{}_{}_{}_customdict_size_{}_box.html".format(DIR_out, smooth, 
                                                                                                            top, contract, rep, timescale, allusda.shape[0])
                fix_plot_layout_and_save(fig_custom_box, savename, xaxis_title="Event window ({})".format(timescale), 
                                        yaxis_title="Percentage of window days with sentiment<br>(custom Agribusiness dictionary)<br>value above given quantile level", 
                                        title="", showgrid=False, showlegend=True,
                                        print_png=True)     
                

                fig_usda = make_subplots(specs=[[{"secondary_y": True}]])
                fig_custom = make_subplots(specs=[[{"secondary_y": True}]])

                xxx = np.arange(-window_around_event+smooth, window_around_event-smooth+1, 1)            
                                                                 
                fig_usda.add_trace(
                            go.Scatter(
                                x=np.arange(-window_around_event+smooth, window_around_event-smooth+1, 1),
                                y=dfplot["usdasmooth"].values[smooth-1:2*window_around_event-smooth],
                                mode="lines",
                                # marker=dict(size=8, symbol=symbols[col], color=colors[col]),
                                name="USDA tech. sentim.")
                        )                                    
                if len(dfplot["customsmooth"].values[smooth-1:2*window_around_event-smooth])==0:
                    print("Check?!")         


                fig_custom.add_trace(
                            go.Scatter(
                                x=np.arange(-window_around_event+smooth, window_around_event-smooth+1, 1),
                                y=dfplot["customsmooth"].values[smooth-1:2*window_around_event-smooth],
                                mode="lines",
                                # marker=dict(size=8, symbol=symbols[col], color=colors[col]),
                                name="Agribusiness sentim.")
                        )   
                if ret_and_vol:
                    fig_usda.add_trace(
                            go.Scatter(
                                x=np.arange(-window_around_event+smooth, window_around_event-smooth+1, 1),
                                y=dfplot["logret"].values[smooth-1:2*window_around_event-smooth],
                                mode="lines",
                                # marker=dict(size=8, symbol=symbols[col], color=colors[col]),
                                name="avg log returns"), secondary_y=True
                        )   
                    fig_custom.add_trace(
                            go.Scatter(
                                x=np.arange(-window_around_event+smooth, window_around_event-smooth+1, 1),
                                y=dfplot["logret"].values[smooth-1:2*window_around_event-smooth],
                                mode="lines",
                                # marker=dict(size=8, symbol=symbols[col], color=colors[col]),
                                name="avg log returns"), secondary_y=True
                        )   
                    fig_usda.add_trace(
                            go.Scatter(
                                x=np.arange(-window_around_event+smooth, window_around_event-smooth+1, 1),
                                y=dfplot["logretvol"].values[smooth-1:2*window_around_event-smooth],
                                mode="lines",
                                # marker=dict(size=8, symbol=symbols[col], color=colors[col]),
                                name="avg log volume"), secondary_y=True
                        )   
                    fig_custom.add_trace(
                            go.Scatter(
                                x=np.arange(-window_around_event+smooth, window_around_event-smooth+1, 1),
                                y=dfplot["logretvol"].values[smooth-1:2*window_around_event-smooth],
                                mode="lines",
                                # marker=dict(size=8, symbol=symbols[col], color=colors[col]),
                                name="avg log volume"), secondary_y=True
                        )   
                fig_usda.update_layout(
                    xaxis = dict(
                        tickmode = 'array',
                        tickvals = np.arange(-window_around_event+smooth,window_around_event-smooth+1, 1),
                        ticktext = ["{}".format(i) for i in np.arange(-window_around_event+smooth,window_around_event-smooth+1, 1)]
                        )                    
                )
                fig_custom.update_layout(
                    xaxis = dict(
                        tickmode = 'array',
                        tickvals = np.arange(-window_around_event+smooth,window_around_event-smooth+1, 1),
                        ticktext = ["{}".format(i) for i in np.arange(-window_around_event+smooth,window_around_event-smooth+1, 1)]
                        )                    
                )                        
                savename = "{}/sentiment_aroundreports_smooth{}_May2024/announcesentiment_{}_{}_{}_{}_usdacmecftc_size_{}.html".format(DIR_out, smooth, 
                                                                                                            top, contract, rep, timescale, allusda.shape[0])
                pathlib.Path("{}/sentiment_aroundreports_smooth{}_May2024/".format(DIR_out, smooth)).mkdir(parents=True, exist_ok=True)
                fig_usda.update_yaxes(title_text="Relative average cumulative log returns/<br>log volume returns (%)", secondary_y=True)               
                fig_usda.update_layout(legend=dict(
                                    orientation="h",
                                    yanchor="top"))            
                fix_plot_layout_and_save(fig_usda, savename, xaxis_title="Event window ({})".format(timescale), 
                                         yaxis_title="Relative average cumulative sentiment returns(%)", 
                                         title="", showgrid=False, showlegend=True,
                                            print_png=True)
                
                savename = "{}/sentiment_aroundreports_smooth{}_May2024/announcesentiment_{}_{}_{}_{}_customdict_size_{}.html".format(DIR_out, 
                                                            smooth, top, contract, rep, timescale, allusda.shape[0])                
                fig_custom.update_yaxes(title_text="Relative average cumulative log returns/<br>log volume returns (%)", secondary_y=True)                
                fig_custom.update_layout(legend=dict(
                                    orientation="h",
                                    yanchor="top"))
                fix_plot_layout_and_save(fig_custom, savename, xaxis_title="Event window ({})".format(timescale), 
                                         yaxis_title="Relative average cumulative sentiment returns (%)", 
                                         title="", showgrid=False, showlegend=True, print_png=True)                                                               
                # fig_usda.show()
                # fig_custom.show()
                # ipdb.set_trace()
            

    t1 = time.time()
    print("Time to completion: " + str(datetime.timedelta(seconds=t1-t0)))
