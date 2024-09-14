import time
import datetime
import pandas as pd
import pathlib
import numpy as np
import plotly.graph_objects as go
from src.text_utils import fix_plot_layout_and_save
from wordcloud import WordCloud
import os 
from collections import Counter
from nltk.corpus import stopwords
from nltk.stem.regexp import RegexpStemmer

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
    DIRtext = "{}/data_text/".format(pwd)
    pathlib.Path(DIR_out).mkdir(parents=True, exist_ok=True)
    stops = set(stopwords.words('english'))
    stopwords = list(stops)
    stopwords.extend(["wheat", "corn", "na", "pct", "soybean", "soybeans", "dow jones", "cent", "dow", "jones"])
    stemmer = RegexpStemmer('s$|ies$') 
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
        
        # read in text postprocessed data
        ppngrams_custom = pd.read_csv("{}/DowJones_{}_cleanedtext_customdict/start_end_ngram_sentiment_total.csv".format(DIRtext, top))                    
        ppngrams_custom.Date = ppngrams_custom.Date.apply(pd.Timestamp)
        ppngrams_usda = pd.read_csv("{}/DowJones_{}_cleanedtext_usdacmecftc/start_end_ngram_sentiment_total.csv".format(DIRtext, top))                    
        ppngrams_usda.Date = ppngrams_usda.Date.apply(pd.Timestamp)
        
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

            if top == "corn":
                periods = {"planting": [4,5],
                           "harvest" : [9,10,11],
                           "grow"    : [6,7,8],
                           "rest"    : [12,1,2,3],
                           "all"     : [1,2,3,4,5,6,7,8,9,10,11,12]}
                dateswithsentiment["period"] = dateswithsentiment.timestamp.apply(set_period_corn)                
            else:
                periods = {"planting": [9,10],
                           "harvest" : [6,7],
                           "grow"    : [11,12,1,2,3,4,5],
                           "rest"    : [8],
                           "all"     : [1,2,3,4,5,6,7,8,9,10,11,12]}
                dateswithsentiment["period"] = dateswithsentiment.timestamp.apply(set_period_wheat)                

            for period in ["planting", "harvest", "grow", "rest"]:
                                
                dateswithsentiment_upd = dateswithsentiment.loc[dateswithsentiment["period"]==period].reset_index(drop=True)                
                sentiment_clean2 = sentiment_clean[sentiment_clean.Dates.isin(dateswithsentiment_upd.timestamp)].reset_index(drop=True)

                index2use = np.array(dateswithsentiment_upd.index[dateswithsentiment_upd["sent_quantile"] > 0])
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
                        sent_quantile_usda = sentiment_clean2["usdacmecftc"].quantile(q)
                        sss = sentiment_clean2.loc[indexrange, "usdacmecftc"].values
                        sssq = sss[sss > sent_quantile_usda]
                        sss_dates = sentiment_clean2.loc[indexrange, "Dates"].values
                        sssq_dates = sss_dates[sss > sent_quantile_usda]
                                                    
                        if len(sssq) > 0:
                            if q in extreme_val_window_usda_above_q_percentage.keys():
                                extreme_val_window_usda_above_q_percentage[q].append(len(sssq)/len(sss))
                            else:
                                extreme_val_window_usda_above_q_percentage[q] = [len(sssq)/len(sss)]
                            if q in [0.6, 0.9]:
                                if not os.path.exists("{}/sentiment_aroundallreports_seasonal_smooth{}_May2024_wordclouds/announcesentiment_{}_{}_{}_{}_usdacmecftc_{}_q{}.png".format(DIR_out, 
                                                                                                                    smooth, top, contract, period, timescale, dateswithsentiment_upd.timestamp.loc[i].strftime('%Y-%m-%d'), str(q).replace(".", ""))):
                                    windowngrams = ppngrams_usda[ppngrams_usda.Date.isin(sssq_dates)].reset_index(drop=False)
                                    if len(windowngrams) > 0:
                                        windowngrams.NgramPostprocessed = windowngrams.NgramPostprocessed.apply(eval) 
                                        aggregate_winngrams_usda = windowngrams.groupby('Date')['NgramPostprocessed'].agg(lambda x: sum(x, []))
                                        tokens_usda = aggregate_winngrams_usda.values.flatten()[0] 
                                        wwc = WordCloud(background_color="white", repeat=False, 
                                                        collocations=True, normalize_plurals=True, 
                                                        include_numbers=False, stopwords=stopwords)
                                        # remove plurals
                                        ngramsin = [stemmer.stem(ngr) for ngr in tokens_usda if ngr not in stopwords]                                        
                                        token_freqs = Counter(ngramsin)
                                        wwc.generate_from_frequencies(token_freqs)
                                        pathlib.Path("{}/sentiment_aroundallreports_seasonal_smooth{}_May2024_wordclouds/".format(DIR_out, smooth)).mkdir(parents=True, exist_ok=True)
                                        wwc.to_file("{}/sentiment_aroundallreports_seasonal_smooth{}_May2024_wordclouds/announcesentiment_{}_{}_{}_{}_usdacmecftc_{}_q{}.png".format(DIR_out, 
                                                                                                                        smooth, top, contract, period, timescale, dateswithsentiment_upd.timestamp.loc[i].strftime('%Y-%m-%d'), str(q).replace(".", "")))
                                    
                            
                        sent_quantile_custom = sentiment_clean2["customdict"].quantile(q)
                        sss = sentiment_clean2.loc[indexrange, "customdict"].values
                        sssq = sss[sss > sent_quantile_custom]
                        sss_dates = sentiment_clean2.loc[indexrange, "Dates"].values
                        sssq_dates = sss_dates[sss > sent_quantile_custom]
                        if len(sssq) > 0:
                            if q in extreme_val_window_custom_above_q_percentage.keys():
                                extreme_val_window_custom_above_q_percentage[q].append(len(sssq)/len(sss))
                            else:
                                extreme_val_window_custom_above_q_percentage[q] = [len(sssq)/len(sss)]
                            if q in [0.6, 0.9]:
                                if not os.path.exists("{}/sentiment_aroundallreports_seasonal_smooth{}_May2024_wordclouds/announcesentiment_{}_{}_{}_{}_custom_{}_q{}.png".format(DIR_out, 
                                                                                                                    smooth, top, contract, period, timescale, dateswithsentiment_upd.timestamp.loc[i].strftime('%Y-%m-%d'), str(q).replace(".", ""))):
                                    windowngrams = ppngrams_custom[ppngrams_custom.Date.isin(sssq_dates)].reset_index(drop=False)
                                    if len(windowngrams) > 0:
                                        windowngrams.NgramPostprocessed = windowngrams.NgramPostprocessed.apply(eval) 
                                        aggregate_winngrams_custom = windowngrams.groupby('Date')['NgramPostprocessed'].agg(lambda x: sum(x, []))                                        
                                        tokens_custom = aggregate_winngrams_custom.values.flatten()[0] 
                                        wwc = WordCloud(background_color="white", repeat=False, 
                                                        collocations=True, normalize_plurals=True, 
                                                        include_numbers=False, stopwords=stopwords)
                                        ngramsin = [stemmer.stem(ngr) for ngr in tokens_custom if ngr not in stopwords]
                                        token_freqs = Counter(ngramsin)
                                        wwc.generate_from_frequencies(token_freqs)
                                        pathlib.Path("{}/sentiment_aroundallreports_seasonal_smooth{}_May2024_wordclouds/".format(DIR_out, smooth)).mkdir(parents=True, exist_ok=True)
                                        wwc.to_file("{}/sentiment_aroundallreports_seasonal_smooth{}_May2024_wordclouds/announcesentiment_{}_{}_{}_{}_custom_{}_q{}.png".format(DIR_out, 
                                                                                                                    smooth, top, contract, period, timescale, dateswithsentiment_upd.timestamp.loc[i].strftime('%Y-%m-%d'), str(q).replace(".", "")))
                                
                            

                fig_usda_box = go.Figure()
                fig_custom_box = go.Figure()
                     
                savename = "{}/sentiment_aroundallreports_seasonal_smooth{}_May2024/announcesentiment_{}_{}_{}_{}_usdacmecftc_box.html".format(DIR_out, smooth, top, contract, period, timescale)
                pathlib.Path("{}/sentiment_aroundallreports_seasonal_smooth{}_May2024/".format(DIR_out, smooth)).mkdir(parents=True, exist_ok=True)                
                fix_plot_layout_and_save(fig_usda_box, savename, xaxis_title="", 
                                        yaxis_title="Percentage of window days with sentiment<br>(USDA technical dictionary)<br>value above given quantile level", 
                                        title="", showgrid=False, showlegend=True,
                                        print_png=True)
                savename = "{}/sentiment_aroundallreports_seasonal_smooth{}_May2024/announcesentiment_{}_{}_{}_{}_customdict_box.html".format(DIR_out, smooth, top, contract, period, timescale)                                
                fix_plot_layout_and_save(fig_custom_box, savename, xaxis_title="Event window ({})".format(timescale), 
                                        yaxis_title="Percentage of window days with sentiment<br>(custom Agribusiness dictionary)<br>value above given quantile level", 
                                        title="", showgrid=False, showlegend=True,
                                        print_png=True)                                                               
                # fig_usda.show()
                # fig_custom.show()
            

    t1 = time.time()
    print("Time to completion: " + str(datetime.timedelta(seconds=t1-t0)))
