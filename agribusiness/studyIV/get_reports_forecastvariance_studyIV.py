import pandas as pd
import pathlib
import numpy as np
from scipy.io import loadmat
import os
import plotly.graph_objects as go
from plotly import io as pio
from plotly.subplots import make_subplots

def set_period_corn(x): 
        
    if pd.to_datetime(x).date().month in [4,5]:
        return "planting"
    elif pd.to_datetime(x).date().month in [9,10,11]:
        return "harvest"
    elif pd.to_datetime(x).date().month in [6,7,8]:
        return "grow"
    elif pd.to_datetime(x).date().month in [12,1,2,3]:
        return "rest"        

def set_period_wheat(x): 
        
    if pd.to_datetime(x).date().month in [9,10]:
        return "planting"
    elif pd.to_datetime(x).date().month in [6,7]:
        return "harvest"
    elif pd.to_datetime(x).date().month in [11,12,1,2,3,4,5]:
        return "grow"
    elif pd.to_datetime(x).date().month in [8]:
        return "rest"  


if __name__ == '__main__':

    pwd = pathlib.Path.cwd()
    DIR = "{}/results_structural_change_Sep2023/".format(pwd)
    DIR_out = "{}/results_structural_change_Sep2023/studyIV/".format(pwd)
    pathlib.Path(DIR_out).mkdir(parents=True, exist_ok=True)

    commodities = ["corn","wheat"]
    contracts = ["front", "second"]
    market = ["PriceRaw", "RealVolatility", "volume", "PriceReturns"]
    monthsstr = {1:"Jan", 2:"Feb", 3:"Mar", 4:"Apr", 5:"May", 6:"Jun", 7:"Jul", 8:"Aug", 9:"Sep", 10:"Oct", 11:"Nov", 12:"Dec"}
    
    for process in market:
        for comm in commodities:
            for contract in contracts:                
                if comm == "corn":
                    reports = ["WASDE", "PP", "ACR", "CPAS", "CPSS", "GS"]
                elif comm == "wheat":
                    reports = ["PP", "ACR", "CPAS", "GS", "SGAS", "CPSS", "WWS", "CPM", "WASDE"]
                boxdata = {"daily": dict(), "hourly": dict()}
                glrtdata = {"daily": dict(), "hourly": dict()}
                if comm=="corn":                               
                    comm_process_contract_daily  = np.zeros((5,3))
                else:
                    comm_process_contract_daily  = np.zeros((5,6))
                comm_process_contract_hourly = np.zeros((2,12))
                for report in reports:
                    print(report)                    
                    if report in ["PP", "ACR", "CPAS", "WWS", "SGAS"]:
                        model_lag = [1, 2, 3, 4, 5]
                        timescale = "daily"
                    elif report in ["CP"]:
                        model_lag = [15, 30, 45, 60]
                        timescale = "minute"
                    else:
                        model_lag = [1, 4]
                        timescale = "hourly"
                    for mlag in model_lag:
                        readdirin = "{}/{}/{}/{}/{}/".format(DIR, comm, contract, timescale, report)
                        current_files = [i for i in os.listdir(readdirin) if process in i and "forecast_variance" in i and "_updFeb24" in i and "_nodata" not in i]
                        current_dates = np.unique([i.split("_")[7] for i in current_files])
                        dates = []
                        for report_date in current_dates:                            
                            filesall = [j for j in current_files if report_date in j and "M1" not in j and "progress" not in j and "lag{}".format(mlag) in j 
                                           and "_nodata" not in j and "meancov" in j]
                            if len(filesall) == 0:
                                print(process, report, report_date, comm, contract, mlag)
                                continue
                            if len(filesall) > 1:
                                print("Check?!")
                                # print(filesall)
                            for filein in [filesall[0]]:                                
                                dat = loadmat("{}/{}".format(readdirin, filein))                                   
                                glrt_tests = dat["control_study_rejectnull"]
                                var_forecasts = dat["variance_forecast"]
                                dateTstamp = pd.Timestamp(str(report_date)).date()                                                                    
                                if any(np.isnan(i) for i in glrt_tests[:,3]):
                                    continue
                                # Granger, var_forecasts: for comm, contract, report, lag, market process - for all report dates: get difference of first line with all the rest 
                                # (model forecast from M1, vs forecasts from all the rest/M2) - if diff is positive then reject null of no causality, plot box plots per farming period
                                if mlag not in boxdata[timescale].keys():
                                    boxdata[timescale][mlag] = dict()
                                diffs = var_forecasts[0, 1] - var_forecasts[1:, mlag+1]  # H = 1 day forecast horizon
                                if comm == "corn":
                                    farmingtime = set_period_corn(report_date)
                                else:                                      
                                    farmingtime = set_period_wheat(report_date)
                                if farmingtime not in boxdata[timescale][mlag].keys():
                                    boxdata[timescale][mlag][farmingtime] = []
                                boxdata[timescale][mlag][farmingtime].extend(diffs.tolist())   
                                
                                # GLRT - glrt_tests: for comm, contract, lag - for all report dates: collect all p vals, box plots of 1-p val , 
                                # plot box plots per market process
                                glrtmonth = dateTstamp.month                                
                                if mlag not in glrtdata[timescale].keys():
                                    glrtdata[timescale][mlag] = dict()
                                if glrtmonth not in glrtdata[timescale][mlag].keys():
                                    glrtdata[timescale][mlag][glrtmonth] = []                              
                                # 1-p values
                                glrtdata[timescale][mlag][glrtmonth].extend([i.flatten()[0] for i in glrt_tests[:,3].flatten().tolist()])                               
                                
                # Granger, boxplots                         
                for timescale in ["daily", "hourly"]:
                    if timescale == "daily":
                        model_lag = [1, 2, 3, 4, 5]
                    else:
                        model_lag = [1, 4]
                    fig = go.Figure()
                    for mlag in model_lag:
                        ydata = []
                        xdata = []
                        for per in boxdata[timescale][mlag].keys():
                            ydata.extend([i for i in boxdata[timescale][mlag][per]])
                            xdata.extend([per for i in range(len(boxdata[timescale][mlag][per]))])
                        fig.add_trace(go.Box(y=ydata, x=xdata, name="Lag(s): {}".format(mlag)))
                        fig.add_hline(y=0, line_width=3, line_dash="dot", line_color="black")
                        fig.update_layout(boxmode="group")
                    savename = "{}/variance_forecasts_h1_box_{}_{}_{}_{}.png".format(DIR_out, timescale, comm, contract, process)
                    fig.update_xaxes(showline=True, linewidth=2, linecolor='black')
                    fig.update_yaxes(showline=True, linewidth=2, linecolor='black')
                    fig.update_layout(title="", plot_bgcolor='rgb(255,255,255)',
                                    yaxis=dict(
                                        title="Difference in forecast (h=1 day) variance",  # forecasts computed with M1, M2 models
                                        titlefont_size=18,
                                        tickfont_size=18,
                                        showgrid=True,
                                    ),
                                    xaxis=dict(
                                        title="Reports release periods ({})".format(comm),
                                        titlefont_size=18,
                                        tickfont_size=18,
                                        showgrid=True
                                    ),
                                    font=dict(
                                        size=14
                                    ),
                                    showlegend=True)
                    pio.write_image(fig, savename, width=1540, height=871, scale=1)
                    # GLRT                
                    for mlag in model_lag:
                        for month in range(1,13,1):
                            try:
                                iqr = np.percentile(glrtdata[timescale][mlag][month], 75) - np.percentile(glrtdata[timescale][mlag][month], 25)                        
                            except:
                                print(mlag, month, comm, contract)
                            if timescale == "daily" and comm=="corn":                    
                                if month==1:
                                    ll = 0
                                elif month==3:
                                    ll = 1
                                else:
                                    ll = 2
                                comm_process_contract_daily[model_lag.index(mlag), ll]  = iqr
                            elif timescale == "daily" and comm=="wheat":                    
                                if month==1:
                                    ll = 0
                                elif month==2:
                                    ll = 1
                                elif month==3:
                                    ll = 2
                                elif month==6:
                                    ll = 3
                                elif month==9:
                                    ll = 4
                                elif month==12:
                                    ll = 5
                                comm_process_contract_daily[model_lag.index(mlag), ll]  = iqr
                            else:
                                comm_process_contract_hourly[model_lag.index(mlag), month-1] = iqr   
                    if timescale == "daily":                                        
                        if comm=="corn":                            
                            xx = ["January", "March", "June"]
                            xxx = ["Soil rest-Jan", 
                                    "Soil rest-Mar", 
                                    "Growing-Jun"]        
                        else:                            
                            xx = ["January", "February", "March", "June", "September", "December"]
                            xxx = ["Growing-Jan", "Growing-Feb", 
                                    "Growing-Mar", 
                                    "Harvest-Jun",                                   
                                    "Planting-Sep","Growing-Dec"]
                        fig = make_subplots(rows=1,cols=1, specs=[[{'secondary_y': True}]])
                        fig.add_trace(go.Heatmap(z=comm_process_contract_daily,
                                                x=xx,
                                                y=[str(ml) for ml in model_lag],
                                                colorscale="greens_r"), secondary_y=False)   
                        fig.add_trace(go.Heatmap(z=comm_process_contract_daily,
                                                x=xxx,
                                                y=[str(ml) for ml in model_lag],
                                                colorscale="greens_r"), secondary_y=True)   
                        fig.update_layout(xaxis2 = {'anchor': 'y', 'overlaying': 'x', 'side': 'top'})  
                        fig.data[1].update(xaxis='x2')     
                        fig.update_yaxes(
                                tickmode = 'array',
                                tickvals = np.arange(0,len(model_lag),1),
                                ticktext = ["","","","","","","","","",""], secondary_y=True
                            )
                        fig.update_xaxes(showline=True, linewidth=2, linecolor='black')
                        fig.update_yaxes(showline=True, linewidth=2, linecolor='black')
                        fig.update_layout(title="", plot_bgcolor='rgb(255,255,255)',
                                        yaxis=dict(
                                            title="Lags (days)",
                                            titlefont_size=18,
                                            tickfont_size=18,
                                            showgrid=True,
                                        ),
                                        xaxis=dict(
                                            title="Report release month",
                                            titlefont_size=18,
                                            tickfont_size=18,
                                            showgrid=True
                                        ),
                                        font=dict(
                                            size=14
                                        ),
                                        showlegend=False)
                        savename= "{}/{}_{}_{}_{}_forecast_glrts.png".format(DIR_out,comm,process,contract,timescale)
                        pio.write_image(fig, savename, width=1024, height=571, scale=1)                        
                    else:
                        xx = ["January", "February", "March", "April", 
                            "May", "June", "July", "August", "September", 
                            "October", "November", "December"]  
                        if comm == "corn":
                            periods = {"planting": [4,5],
                                        "harvest" : [9,10,11],
                                        "grow"    : [6,7,8],
                                        "rest"    : [12,1,2,3]
                                        }            
                            xxx = ["Soil rest-Jan", "Soil rest-Feb", 
                                "Soil rest-Mar", "Planting-Apr", 
                                "Planting-May", "Growing-Jun", 
                                "Growing-Jul", "Growing-Aug", 
                                "Harvest-Sep", "Harvest-Oct", 
                                "Harvest-Nov", "Soil rest-Dec"]                           
                        else:
                            periods = {"planting": [9,10],
                                        "harvest" : [6,7],
                                        "grow"    : [11,12,1,2,3,4,5],
                                        "rest"    : [8]
                                        }
                            xxx = ["Growing-Jan", "Growing-Feb", 
                                "Growing-Mar", "Growing-Apr",
                                "Growing-May", "Harvest-Jun", 
                                "Harvest-Jul", "Soil rest-Aug", 
                                "Planting-Sep", "Planting-Oct", 
                                "Growing-Nov", "Growing-Dec"]                         
                        fig = make_subplots(rows=1,cols=1, specs=[[{'secondary_y': True}]])
                        fig.add_trace(go.Heatmap(z=comm_process_contract_hourly,
                                                x=xx,
                                                y=[str(ml) for ml in model_lag],
                                                colorscale="greens_r"), secondary_y=False)
                        fig.add_trace(go.Heatmap(z=comm_process_contract_hourly,
                                                x=xxx,
                                                y=[str(ml) for ml in model_lag],
                                                colorscale="greens_r"), secondary_y=True)   
                        fig.update_layout(xaxis2 = {'anchor': 'y', 'overlaying': 'x', 'side': 'top'})  
                        fig.data[1].update(xaxis='x2')      
                        fig.update_xaxes(showline=True, linewidth=2, linecolor='black')
                        fig.update_yaxes(showline=True, linewidth=2, linecolor='black')
                        fig.update_layout(title="", plot_bgcolor='rgb(255,255,255)',
                                        yaxis=dict(
                                            title="Lags (hours)",
                                            titlefont_size=18,
                                            tickfont_size=18,
                                            showgrid=True,
                                        ),
                                        xaxis=dict(
                                            title="Report release month",
                                            titlefont_size=18,
                                            tickfont_size=18,
                                            showgrid=True
                                        ),
                                        font=dict(
                                            size=14
                                        ),
                                        showlegend=False)
                        savename= "{}/{}_{}_{}_{}_forecast_glrts.png".format(DIR_out,comm,process,contract,timescale)
                        pio.write_image(fig, savename, width=1024, height=571, scale=1)
                
                
 