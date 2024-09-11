import time
import datetime
import pandas as pd
import pathlib
import numpy as np
import plotly.graph_objects as go
from agribusiness.src.text_utils_agr import fix_plot_layout_and_save

if __name__ == '__main__':
    
    DIR_out = "/home/yannis/Dropbox (Heriot-Watt University Team)/agrinlp/cbot_data_out/"
    pathlib.Path(DIR_out).mkdir(parents=True, exist_ok=True)

    kwargs_list = ["wheat"]
    symbols = {"CP": "cross",
                "CPSS": "x",
                "ACR": "circle",
                "PP": "triangle-up",
                "CPAS": "diamond",
                "GS": "square",
                "WASDE":  "triangle-left",
                "SGAS": "triangle-right",
                "WWS": "triangle-down",
                "CPM": "circle-dot"}

    colors = {"CP": "#1f77b4",
                       "CPSS": "#ff7f0e",
                       "ACR": "#2ca02c",
                       "PP": "#d62728",
                       "CPAS": "#9467bd",
                       "GS": "#8c564b",
                       "WASDE": "#e377c2",
                       "SGAS": "#7f7f7f",
                       "WWS": "#bcbd22",
                       "CPM": "#17becf" }

    cols = sorted(["WASDE", "CP",  "CPM",  "CPSS", "ACR", "PP", "CPAS", "GS", 
                   "SGAS", "WWS"])

    fig = go.Figure()
    t0 = time.time()

    for top in kwargs_list:
         start = True
         datain = pd.read_csv("{}/{}_postproc_priceVol_daily_ZW1.csv".format(DIR_out, top), index_col="timestamp")
         total_reports = datain.iloc[:,6:-1].sum(axis=1)
         multi_reports_dates = np.argwhere(total_reports.values).flatten()         
         for col in cols:                
                yshow = datain[col]
                yshow.loc[yshow > 0] = total_reports[yshow > 0].values                
                yshow.loc[yshow == 0] = None
                fig.add_trace(
                    go.Scatter(
                        x=datain.index.values,
                        y=yshow.values+cols.index(col)*0.1, #datain[col].values
                        mode="markers",
                        marker=dict(size=8, symbol=symbols[col], color=colors[col]),
                        name=col)
                )
                
    fig.update_layout(
        yaxis={"tickmode":"array","tickvals":[1,2,3,4]}        
    )
    fig.update_layout(legend=dict(orientation="h", yanchor="top"))
    fig.add_hline(y=1.9)
    fig.add_hline(y=2.9)
    fig.add_hline(y=3.9)
    fig.show()    
    savename = "{}/announcement_days_upd_all.html".format(DIR_out)
    fix_plot_layout_and_save(fig, savename, xaxis_title="Date", yaxis_title="Number of issued announcements", 
                  title="", showgrid=False, showlegend=True, print_png=True)

    t1 = time.time()
    print("Time to completion: " + str(datetime.timedelta(seconds=t1-t0)))

