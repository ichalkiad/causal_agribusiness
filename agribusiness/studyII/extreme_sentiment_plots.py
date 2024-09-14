import time
import datetime
import pandas as pd
import pathlib
import numpy as np
import plotly.graph_objects as go
from src.text_utils import fix_plot_layout_and_save
from plotly.subplots import make_subplots
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

    pwd = pathlib.Path.cwd()
    DIR_out = "{}/cbot_data_out/".format(pwd)
    DIRvol = DIR_out
    DIRtext = "{}/data_text/".format(pwd)
    pathlib.Path(DIR_out).mkdir(parents=True, exist_ok=True)
    
    smooth = 3
    window_around_event = 10 + smooth
    dictionary = "usdacmecftc" #"customdict"
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
    stops = set(stopwords.words('english'))
    stopwords = list(stops)
    stopwords.extend(["wheat", "corn", "na", "pct", "soybean", "soybeans", "dow jones", "cent", "dow", "jones", "quote", "usda", "crop"])
    stemmer = RegexpStemmer('s$|ies$') 
    medianvaly1 = []
    medianvaly2 = []
    periodq = []
    for top in kwargs_list:
        # read in text postprocessed data
        ppngrams = pd.read_csv("{}/DowJones_{}_cleanedtext_{}/start_end_ngram_sentiment_total.csv".format(DIRtext, top, dictionary))                    
        ppngrams.Date = ppngrams.Date.apply(pd.Timestamp)
        
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
            
            if top == "corn":
                periods = {"planting": [4,5],
                           "harvest" : [9,10,11],
                           "grow"    : [6,7,8],
                           "rest"    : [12,1,2,3],
                           "all"     : [1,2,3,4,5,6,7,8,9,10,11,12]}
                dateswithsentiment["period"] = dateswithsentiment.timestamp.apply(set_period_corn)
                voldateswithsentiment["period"] = voldateswithsentiment.timestamp.apply(set_period_corn)
            else:
                periods = {"planting": [9,10],
                           "harvest" : [6,7],
                           "grow"    : [11,12,1,2,3,4,5],
                           "rest"    : [8],
                           "all"     : [1,2,3,4,5,6,7,8,9,10,11,12]}
                dateswithsentiment["period"] = dateswithsentiment.timestamp.apply(set_period_wheat)
                voldateswithsentiment["period"] = voldateswithsentiment.timestamp.apply(set_period_wheat)

            for period in ["all", "planting", "harvest", "grow", "rest"]:                
                if period == "all":
                    dateswithsentiment_upd = dateswithsentiment
                    voldateswithsentiment_upd = voldateswithsentiment
                    sentiment_clean2 = sentiment_clean
                else:
                    dateswithsentiment_upd = dateswithsentiment.loc[dateswithsentiment["period"]==period].reset_index(drop=True)
                    voldateswithsentiment_upd = voldateswithsentiment.loc[voldateswithsentiment["period"]==period].reset_index(drop=True)
                    sentiment_clean2 = sentiment_clean[sentiment_clean.Dates.isin(dateswithsentiment_upd.timestamp)].reset_index(drop=True)
                
                for q in [0.6, 0.7, 0.8, 0.9]:
                    # Get extreme quantile of sentiment
                    sent_quantile = sentiment_clean2[dictionary].quantile(q)
                    index2use = np.array([sentiment_clean2.index[sentiment_clean2[dictionary] > sent_quantile]])
                    # indices of extreme events
                    index2use = index2use[index2use >= window_around_event]
                    index2use = index2use[index2use <= index2use.tolist()[-1] - window_around_event]
                    extreme_val_window_logRet = []
                    extreme_val_window_vol = []    
                    ngrams = []            
                    if os.path.exists("{}/extremeplots_smooth{}_May2024_wordclouds/announcesentiment_{}_{}_{}_{}_{}_q{}.png".format(DIR_out, 
                                                                    smooth, top, contract, period, timescale, dictionary, str(q).replace(".", ""))):
                        continue
                    
                    for i in index2use:
                        indexrange = np.arange(i-window_around_event, i+window_around_event+1, 1).tolist()                                            
                        # returns                        
                        vals_logRet = dateswithsentiment_upd.loc[indexrange, "{}_{}contractLogRet_daily".format(top, contract)].values
                        
                        if vals_logRet[0] == 0:
                            continue
                        cumret = 100*(np.cumsum(vals_logRet) - vals_logRet[0])/vals_logRet[0]
                        extreme_val_window_logRet.append(cumret)
                        
                        # vol
                        vals_vol = voldateswithsentiment_upd.loc[indexrange, "{}_{}contractTradedVolLogDiff_daily".format(top, contract)].values                        
                        if vals_vol[0] == 0:
                            continue
                        extreme_val_window_vol.append(100*(np.cumsum(vals_vol) - vals_vol[0])/vals_vol[0])

                        # get text dates
                        windowngrams = ppngrams[ppngrams.Date.isin(dateswithsentiment_upd.timestamp.loc[indexrange])].reset_index(drop=False)
                        windowngrams.NgramPostprocessed = windowngrams.NgramPostprocessed.apply(eval) 
                        aggregate_winngrams = windowngrams.groupby('Date')['NgramPostprocessed'].agg(lambda x: sum(x, []))
                        if len(aggregate_winngrams.values) > 0:
                            ngrams.extend(aggregate_winngrams.values.flatten().tolist()[0])
                    
                    # text in all sentiment quantile periods
                    wwc = WordCloud(background_color="white", repeat=False, 
                                collocations=True, normalize_plurals=True, 
                                include_numbers=False, stopwords=stopwords)
                    ngramsin = [stemmer.stem(ngr) for ngr in ngrams if ngr not in stopwords]
                    token_freqs = Counter(ngramsin)
                    wwc.generate_from_frequencies(token_freqs)
                    pathlib.Path("{}/extremeplots_smooth{}_May2024_wordclouds/".format(DIR_out, smooth)).mkdir(parents=True, exist_ok=True)
                    wwc.to_file("{}/extremeplots_smooth{}_May2024_wordclouds/announcesentiment_{}_{}_{}_{}_{}_q{}.png".format(DIR_out, 
                                                                    smooth, top, contract, period, timescale, dictionary, str(q).replace(".", "")))
                    
                    alllogret = np.vstack(extreme_val_window_logRet)
                    allvol = np.vstack(extreme_val_window_vol)
                    print(alllogret.shape)
                    y1 = np.mean(alllogret, 0)
                    y2 = np.mean(allvol, 0)
                    dfplot = pd.DataFrame.from_dict({"y1":y1, "y2":y2})
                    dfplot["y1"] = dfplot["y1"].rolling(smooth).mean()
                    dfplot["y2"] = dfplot["y2"].rolling(smooth).mean()                    
                    periodq.append("{}-{:.1f}".format(period, q))
                                        
                    fig = make_subplots(specs=[[{"secondary_y": True}]])
                    fig.add_trace(
                                go.Scatter(
                                    x=np.arange(-window_around_event+smooth, window_around_event-smooth+1, 1),
                                    y=dfplot["y1"].values[smooth-1:2*window_around_event-smooth],
                                    mode="lines",
                                    # marker=dict(size=8, symbol=symbols[col], color=colors[col]),
                                    name="avg log returns")
                            )                                        
                    fig.add_trace(
                                go.Scatter(
                                    x=np.arange(-window_around_event+smooth, window_around_event-smooth+1, 1),
                                    y=dfplot["y2"].values[smooth-1:2*window_around_event-smooth],
                                    mode="lines",
                                    # marker=dict(size=8, symbol=symbols[col], color=colors[col]),
                                    name="avg log volume"), secondary_y=True
                            )   
                    fig.update_layout(
                    xaxis = dict(
                        tickmode = 'array',
                        tickvals = np.arange(-window_around_event+smooth,window_around_event-smooth+1, 1),
                        ticktext = ["{}".format(i) for i in np.arange(-window_around_event+smooth,window_around_event-smooth+1, 1)]
                        )
                    )                            
                    savename = "{}/extremeplots_smooth{}/extremesentiment_{}_{}_{}_{:.2f}_{}_{}_size_{}.html".format(DIR_out, 
                                                        smooth, top, contract, dictionary, q, timescale, period, alllogret.shape[0])
                    pathlib.Path("{}/extremeplots_smooth{}/".format(DIR_out, smooth)).mkdir(parents=True, exist_ok=True)
                    fig.update_yaxes(title_text="Relative average cumulative log volume (%)", secondary_y=True)
                    # fig.show()                    
                    fix_plot_layout_and_save(fig, savename, xaxis_title="Event window ({})".format(timescale), 
                                yaxis_title="Relative average cumulative log returns (%)", title="", showgrid=False, showlegend=True,
                                print_png=True)

    t1 = time.time()
    print("Time to completion: " + str(datetime.timedelta(seconds=t1-t0)))
