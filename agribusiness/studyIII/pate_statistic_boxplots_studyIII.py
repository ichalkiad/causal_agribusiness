import pickle
import time
import logging
import datetime
import ipdb
import pandas as pd
import pathlib
import numpy as np
import plotly.graph_objects as go
from text_utils_agr import fix_plot_layout_and_save
from plotly.subplots import make_subplots

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
    DIR = "{}/cbot_data/".format(pwd)    
    DIR_out = "{}/results_structural_change_Sep2023/studyIII/".format(pwd)
    pathlib.Path(DIR_out).mkdir(parents=True, exist_ok=True)
    
    kwargs_list = ["corn", "wheat"] 
    contracts   = {"wheat front": "ZW1",
                   "wheat second": "ZW2",
                   "corn front" : "ZC1",
                   "corn second" : "ZC2"}
    t0 = time.time()
    def to_date(x) : return pd.to_datetime(x).date()    
    market = ["PriceRaw", "RealVolatility", "volume", "PriceReturns"]    
    for top in kwargs_list:
        if top == "wheat":
            reports = ["WASDE", "PP", "ACR", "CPAS", "GS", "CPM", "SGAS", "CPSS", "WWS"]
        elif top == "corn":
            reports = ["WASDE", "GS", "PP", "ACR", "CPAS", "CPSS"]
        
        for contract in ["front", "second"]:

            datain = pd.read_csv("{}/studyI_data_single_window_gp_mean.csv".format(DIR_out))                       
            datainnoreport = pd.read_csv("{}/studyI_data_noreport_window_gp_mean.csv".format(DIR_out))           
                        
            for rep in reports:

                if rep in ["PP", "ACR", "CPAS", "WWS", "SGAS"]:
                    model_lag = [1, 2, 3, 4, 5]
                    timescale = "daily"                
                else:
                    model_lag = [1, 4] 
                    timescale = "hourly"

                for lag in model_lag:                   
                    
                    fig = go.Figure()
                    fig_norep = go.Figure()
                    for process in market:
                        
                        data = datain.loc[datain.commodity==top]
                        data = data.loc[data.contracts==contract]                        
                        data = data.loc[data.marketproc==process]       

                        datanoreport = datainnoreport.loc[datainnoreport.commodity==top]
                        datanoreport = datanoreport.loc[datanoreport.contracts==contract]                        
                        datanoreport = datanoreport.loc[datanoreport.marketproc==process]                                         
    
                        pate_reports = dict()
                        pate_parallel_reports = dict()
                        pate_reports_norep = dict()
                        # plotting 1 vs multiple reports                        
                        for q in reports:
                            # max 4 reports on the same day
                            for p in [1,2,3,4]:                                
                                pate_reports["{}_{}".format(q, p)] = []
                                pate_parallel_reports["{}_{}".format(q, p)] = []
                        
                        # plotting 1 vs none report
                        pate_reports_norep["{}".format(rep)] = {"rep":[], "norep":[]}
                        
                        report_dates = pd.read_csv("{}/{}_alldates.csv".format(DIR, rep.lower()), index_col=0)                                  
                        data.reportdate = np.asarray([str(pd.Timestamp(i).date()) for i in data.reportdate.values.tolist()])
                        datanoreport.reportdate = np.asarray([str(pd.Timestamp(i).date()) for i in datanoreport.reportdate.values.tolist()])                        
                      
                        for rday in report_dates.dates:
                            # for each report day of each report, keep the current lag
                            data_all_rep = data.loc[data.reportdate==rday]
                            data_rep = data_all_rep.loc[data_all_rep.reportsall==rep]
                            data_rep = data_rep.loc[data_rep.lagstruct==lag]                            
                            if len(data_all_rep)==0 or len(data_rep) == 0:
                                continue   
                            # get all distinct reports that were published on that day                                                                                 
                            reps_on_day = np.unique(data_all_rep.reportsall.values)
                            
                            # for each report day of each report, keep the current lag
                            # this is based on the window of data right before the publication hence excluding the publication event
                            data_all_norep = datanoreport.loc[datanoreport.reportdate==rday]
                            data_norep = data_all_norep.loc[data_all_norep.reportsall==rep]
                            data_norep = data_norep.loc[data_norep.lagstruct==lag]                            
                            if len(data_norep) == 0:
                                continue                                                                

                            # plotting 1 vs multiple reports                            
                            if len(reps_on_day) == 1:
                                irep = reps_on_day[0]
                            elif len(reps_on_day) > 1:
                                irep = ",".join(reps_on_day)                            
                            try:                                                        
                                pate_reports["{}_{}".format(rep, len(reps_on_day))][irep].append(data_rep["gp_mean"].values[0])
                            except:
                                pate_reports["{}_{}".format(rep, len(reps_on_day))] = {irep: [data_rep["gp_mean"].values[0]]}
                                  
                            # plotting 1 vs none report
                            pate_reports_norep["{}".format(rep)]["rep"].append(data_rep["gp_mean"].values[0])
                            pate_reports_norep["{}".format(rep)]["norep"].append(data_norep["gp_mean"].values[0])
                            
                            
                        # plotting 1 vs multiple reports                                             
                        for p in range(1,5,1):
                            if len(pate_reports["{}_{}".format(rep, p)]) == 0:
                                continue
                            for q in range(p+1,5,1):
                                if len(pate_reports["{}_{}".format(rep, q)]) == 0:
                                    continue                                                              
                                for first_key in pate_reports["{}_{}".format(rep, p)].keys():
                                    for parallel_reps in pate_reports["{}_{}".format(rep, q)].keys():                                        
                                        alldiffs = np.array(pate_reports["{}_{}".format(rep, p)][first_key])[:,np.newaxis] - np.array(pate_reports["{}_{}".format(rep, q)][parallel_reps])
                                        parallel_reps = parallel_reps.replace(",", "+")
                                        fig.add_trace(go.Box(y=alldiffs.flatten(), name="{}-{}<br>{}".format(first_key, parallel_reps, process),
                                                            boxpoints='all',
                                                            jitter=0.3, 
                                                            pointpos=-1.5,
                                                            notched=False))
                                                    
                        # plotting 1 vs no report
                        alldiffs_norep = np.array(pate_reports_norep["{}".format(rep)]["rep"])[:,np.newaxis] - np.array(pate_reports_norep["{}".format(rep)]["norep"])   
                        fig_norep.add_trace(go.Box(y=alldiffs_norep.flatten(), name="{}-no {}<br> {}".format(rep, rep, process),
                                                            boxpoints='all',
                                                            jitter=0.3, 
                                                            pointpos=-1.5,
                                                            notched=True))                        
                    fig.add_hline(y=0)
                    fig.update_layout(legend=dict(
                                            orientation="h",
                                            yanchor="top"))     
                    fig_norep.add_hline(y=0)
                    fig_norep.update_layout(legend=dict(
                                            orientation="h",
                                            yanchor="top"))                                      
                    # plotting 1 vs multiple reports
                    savename = "{}/pate/{}_{}_lag_{}_contract_{}.html".format(DIR_out, top, rep, lag, contract)
                    fix_plot_layout_and_save(fig, savename, xaxis_title="", 
                                            yaxis_title="PATE estimator during overlapping publication dates", 
                                            title="", showgrid=False, showlegend=False,
                                            print_png=True)           
                    # plotting 1 vs no report
                    savename = "{}/pate_repnorep/{}_{}_lag_{}_contract_{}.html".format(DIR_out, top, rep, lag, contract)
                    fix_plot_layout_and_save(fig_norep, savename, xaxis_title="", 
                                            yaxis_title="PATE estimator - no publication vs publication dates", 
                                            title="", showgrid=False, showlegend=False,
                                            print_png=True)                    
                    # fig.show()                                           

    t1 = time.time()
    print("Time to completion: " + str(datetime.timedelta(seconds=t1-t0)))
