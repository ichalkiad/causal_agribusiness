import time
import datetime
import pandas as pd
import pathlib
import numpy as np
import plotly.graph_objects as go
from src.text_utils import fix_plot_layout_and_save
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
    DIR_in = "{}/results_out_causality_Sep2023/studyII/".format(pwd)
    DIR_out = "{}/results_out_causality_Sep2023/studyII/".format(pwd)
    pathlib.Path(DIR_out).mkdir(parents=True, exist_ok=True)
    def retmonth(x): return pd.Timestamp(x).month
    t0 = time.time()   
    for top in ["wheat", "corn"]:
        datain = pd.read_csv("{}/{}.csv".format(DIR_in, top))   
        for lex in ["customdict", "usdacmecftc"]:
            for causal_rel in ["mean", "meancov"]:
                for contract in ["front", "second"]:                    
                    for lag in [1, 3, 5]:                        
                        datanow = datain.loc[datain.lexicon == lex]
                        datanow = datanow.loc[datanow.causalloc == causal_rel]
                        datanow = datanow.loc[datanow.contractval == contract]
                        datanow = datanow.loc[datanow.lagval == lag]
                        output_image = []
                        for causaldirection in ["m2sent", "sent2m"]:
                            datanowpart = datanow.loc[datanow.msentdirection == causaldirection]
                            for marketproc in ["price", "priceret", "vol", "realvol", "volrealvol"]:
                                datanowpart2 = datanowpart.loc[datanowpart.marketprocess == marketproc]                                
                                datanowpart2["monthsval"] = datanowpart2.datesval.apply(retmonth)
                                monthsum = np.zeros((12,))
                                for i in range(1,13,1):
                                    monthlydata = datanowpart2.loc[datanowpart2.monthsval == i]
                                    # summary of monthly data - now: median                                                                        
                                    monthsum[i-1] = np.percentile(monthlydata.oneminusp.values, q=50, interpolation="higher")
                                output_image.append(monthsum)                        
                        fig = make_subplots(rows=1,cols=1, specs=[[{'secondary_y': True}]])
                        fig.add_trace(go.Heatmap(
                                        z=output_image,
                                        x=["January", "February", 
                                           "March", "April", "May",
                                           "June", "July", "August",
                                           "September", "October",
                                           "November", "December"],
                                        y=["Price (m2s)", "PriceRet (m2s)", 
                                           "Volume (m2s)", "Real. Volat. (m2s)", 
                                           "Vol. s.i. Real. volat. (m2s)",
                                           "Price (s2m)", "PriceRet (s2m)", 
                                           "Volume (s2m)", "Real. Volat. (s2m)", 
                                           "Vol. s.i. Real. volat. (s2m)"], zmin=0, zmax=1, 
                                            colorscale = "Greys",
                                        hoverongaps = False), secondary_y=False)
                        if top == "corn":
                            periods = {"planting": [4,5],
                                       "harvest" : [9,10,11],
                                       "grow"    : [6,7,8],
                                       "rest"    : [12,1,2,3]
                                       }            
                            x = ["Soil rest-Jan", "Soil rest-Feb", 
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
                            x = ["Growing-Jan", "Growing-Feb", 
                                 "Growing-Mar", "Growing-Apr",
                                  "Growing-May", "Harvest-Jun", 
                                  "Harvest-Jul", "Soil rest-Aug", 
                                  "Planting-Sep", "Planting-Oct", 
                                  "Growing-Nov", "Growing-Dec"]                           
                            # y=["m2sent-price", "m2sent-priceret", 
                            #                "m2sent-vol", "m2sent-realvol", 
                            #                "m2sent-volrealvol",
                            #                "sent2m-price", "sent2m-priceret", 
                            #                "sent2m-vol", "sent2m-realvol", 
                            #                "sent2m-volrealvol"]
                        fig.add_trace(go.Heatmap(
                                        z=output_image,
                                        x=x,
                                        y=["Price (m2s)", "PriceRet (m2s)", 
                                           "Volume (m2s)", "Real. Volat. (m2s)", 
                                           "Vol. s.i. Real. volat. (m2s)",
                                           "Price (s2m)", "PriceRet (s2m)", 
                                           "Volume (s2m)", "Real. Volat. (s2m)", 
                                           "Vol. s.i. Real. volat. (s2m)"], zmin=0, zmax=1, 
                                            colorscale = "Greys",
                                        hoverongaps = False), secondary_y=False)
                        fig.add_trace(go.Heatmap(
                                        z=output_image,
                                        x=["January", "February", 
                                           "March", "April", "May",
                                           "June", "July", "August",
                                           "September", "October",
                                           "November", "December"],
                                        y=["Price (m2s)", "PriceRet (m2s)", 
                                           "Volume (m2s)", "Real. Volat. (m2s)", 
                                           "Vol. s.i. Real. volat. (m2s)",
                                           "Price (s2m)", "PriceRet (s2m)", 
                                           "Volume (s2m)", "Real. Volat. (s2m)", 
                                           "Vol. s.i. Real. volat. (s2m)"], zmin=0, zmax=1, 
                                            colorscale = "Greys",
                                        hoverongaps = False), secondary_y=True)
                        fig.update_layout(xaxis2 = {'anchor': 'y', 'overlaying': 'x', 'side': 'top'})  
                        fig.data[1].update(xaxis='x2')                                                        
                        fig.update_yaxes(
                                tickmode = 'array',
                                tickvals = np.arange(0,10,1),
                                ticktext = ["","","","","Market to<br>Sent.","","","","","Sent. to<br>Market"], secondary_y=True
                            )
                        
                        # fig.show()
                        savename = "{}/{}_{}_{}_{}_lag{}.html".format(DIR_out, top, contract, causal_rel, lex, lag)
                        fix_plot_layout_and_save(fig, savename, xaxis_title="Months", yaxis_title="1-pvalue", title="", 
                                                 showgrid=False, showlegend=True, print_png=True)                                      

    t1 = time.time()
    print("Time to completion: " + str(datetime.timedelta(seconds=t1-t0)))
