import pandas as pd
import pathlib
import numpy as np
from plotly import graph_objects as go, io as pio
from plotly.subplots import make_subplots

def set_periodcolor_corn(x): 
        
    if x in [4,5]:
        return "planting"
    elif x in [9,10,11]:
        return "harvest"
    elif x in [6,7,8]:
        return "grow"
    elif x in [12,1,2,3]:
        return "rest"        

def set_periodcolor_wheat(x): 
        
    if x in [9,10]:
        return "planting"
    elif x in [6,7]:
        return "harvest"
    elif x in [11,12,1,2,3,4,5]:
        return "grow"
    elif x in [8]:
        return "rest"        

if __name__ == '__main__':

    pwd = pathlib.Path.cwd()
    DIR = "{}/results_structural_change_Sep2023/".format(pwd)
    DIR_out = "{}/results_structural_change_Sep2023/studyI/".format(pwd)
    pathlib.Path(DIR_out).mkdir(parents=True, exist_ok=True)

    commodities = ["corn","wheat"]
    contracts = ["front", "second"]
    market = ["PriceRaw", "RealVolatility", "volume", "PriceReturns"]
    df = pd.read_csv("{}studyI_data.csv".format(DIR_out))
    
    for process in market:
        for comm in commodities:
            for contract in contracts:
                print(process, comm, contract) 
                if comm=="corn":                               
                    comm_process_contract_daily  = np.zeros((5,3))
                else:
                    comm_process_contract_daily  = np.zeros((5,6))
                comm_process_contract_hourly = np.zeros((2,12))  ##2 for lags 1,4 , 3 for 1,4,8
                for timescale in ["daily", "hourly"]:         
                    if timescale == "daily":                    
                        model_lag = [1, 2, 3, 4, 5]                        
                    else:
                        model_lag = [1, 4] #, 8]                        
                    for mlag in model_lag:
                        for month in range(1,13,1):
                            dfpart = df[(df.lagstruct==mlag) & (df.months==month) & \
                                         (df.timescales==timescale) & (df.commodity==comm) & \
                                            (df.marketproc==process) & (df.contracts==contract)]
                            if len(dfpart)==0:
                                continue
                            iqr = np.percentile(dfpart.oneminusp.values, 75) - np.percentile(dfpart.oneminusp.values, 25)
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
                        savename= "{}/{}_{}_{}_{}.png".format(DIR_out,comm,process,contract,timescale)
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
                        savename= "{}/{}_{}_{}_{}.png".format(DIR_out,comm,process,contract,timescale)
                        pio.write_image(fig, savename, width=1024, height=571, scale=1)                      
                    # fig.show()
                    # ipdb.set_trace()
        

