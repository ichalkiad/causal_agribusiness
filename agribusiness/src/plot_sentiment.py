import json
import pandas as pd
import numpy as np
import pathlib
from agribusiness.src.text_utils_agr import plot_timeseries_with_confidence


if __name__ == '__main__':

    pwd = pathlib.Path.cwd()
    DIR = "{}/data_text/".format(pwd)     
    
    kwargs_list = ["corn","wheat"]
    sites = ["DowJones"]
    median_win = 7
    mean_win = 20 # monthly mean

    for com in kwargs_list:
        dataout = dict()
        for wlist in ["usdacmecftc", "customdict"]:              
            out = "{}/DowJones_{}_cleanedtext_{}/".format(DIR, com, wlist)

            datain = pd.read_csv("{}start_end_ngram_sentiment_total.csv".format(out))
            data_daily = datain.groupby('Date')['Entropy'].apply(np.percentile, q=50,
                                                               interpolation="lower").reset_index(name='Sentiment')
            
            data_tmp = dict()
            data_tmp["dates"] = np.arange(np.datetime64(data_daily["Date"][0]), np.datetime64(data_daily["Date"].values.tolist()[-1]) + np.timedelta64(1, 'D'))
            data_tmp = pd.DataFrame.from_dict(data_tmp)
            data_tmp["sentiment"] = np.nan
            data_tmp.loc[np.in1d(data_tmp["dates"], data_daily["Date"].apply(np.datetime64)), "sentiment"] = data_daily["Sentiment"].values
            data_tmp["sentiment"] = data_tmp["sentiment"].ffill()
            sent_mediansmotthed_backward = data_tmp["sentiment"].rolling(mean_win).mean().dropna()
            print(sent_mediansmotthed_backward)
            if "Dates" not in dataout.keys():
                dataout["Dates"] = data_tmp["dates"]

            if not isinstance(dataout, pd.DataFrame):
                dataout = pd.DataFrame.from_dict(dataout)
            dataout.loc[np.in1d(dataout["Dates"], data_daily["Date"].apply(np.datetime64)), wlist] = data_tmp["sentiment"]
            dataout[wlist] = dataout[wlist].ffill()

            with open(DIR + "src/config_dframes.json", 'rt') as config_file:  # src/config.json
                cfg = json.load(config_file)
            cfg["output_plot_folder"] = "{}ngram_entropy.html".format(out)
            cfg["metadata"] = {"annotation": datain["NgramPostprocessed"]}
            
            plot_timeseries_with_confidence(data_daily["Sentiment"].to_numpy().tolist(), [data_daily["Sentiment"].to_numpy().tolist(),sent_mediansmotthed_backward.to_numpy().tolist()],
                                            data_daily["Date"].to_numpy().tolist(), xaxis_title="Date", yaxis_title="Sentiment",
                                            title="", name=["Raw","Rollingmean"], 
                                            savename="{}daily_sentiment_{}_{}_rollmean.html".format(out, com, wlist), lcolor="blue", 
                                            dashed=["dash"], applyfn=1, smoothing_win=7, bounds=[], yrange=[], ci_window_smooth=7)

        dataout.to_csv("{}/DowJonesNews_{}_carry_fwd_alldictssentiment.csv".format(out, com), index=False)