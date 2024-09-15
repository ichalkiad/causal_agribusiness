import pandas as pd
import pathlib
import numpy as np
from scipy.io import loadmat
import os

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
    year = []
    months = []
    oneminusp = []
    reportsall = []
    lagstruct = []
    commods = []
    contractsall = []
    marks = []
    reject = []
    tstats = []
    yearmonth = []
    timescales = []
    rdays = []
    period = []  
    for process in market:
        for comm in commodities:
            for contract in contracts:            
                if comm == "corn":
                    reports = ["PP", "ACR", "CPAS", "WASDE", "CPSS", "GS"]
                elif comm == "wheat":
                    reports = ["PP", "ACR", "CPAS", "GS", "SGAS", "CPSS", "WWS", "CPM", "WASDE"]
                for report in reports:                         
                    if report in ["PP", "ACR", "CPAS", "WWS", "SGAS"]:
                        model_lag = [1, 2, 3, 4, 5]
                        timescale = "daily"                    
                    else:
                        model_lag = [1, 4]
                        timescale = "hourly"
                    for mlag in model_lag:
                        readdirin = "{}/{}/{}/{}/{}/".format(DIR, comm, contract, timescale, report)    
                        # change if statements if needed to reflect proper filenames!!!
                        current_files = [i for i in os.listdir(readdirin) if process in i and "forecast_variance" in i and "_updFeb24" in i and "_nodata" not in i]
                        current_dates = np.unique([i.split("_")[7] for i in current_files])
                        dates = []
                        for report_date in current_dates:
                            print(report_date)                            
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
                                dateTstamp = pd.Timestamp(str(report_date)).date()                                                         
                                # start from 1, since at position 0 is the pre-report window                                       
                                for jj in range(1,len(dat["control_study_rejectnull"]),1):
                                    try:
                                        if dat["control_study_rejectnull"][jj][3][0][0] <= 0.05:
                                            reject.append(1)                                            
                                        else:
                                            reject.append(0)
                                    except:                                        
                                        if len(dat["control_study_rejectnull"][jj][3]) == 0 or len(dat["control_study_rejectnull"][jj][3][0]) == 0:                                            
                                            print(filein)
                                            continue
                                    oneminusp.append(1 - dat["control_study_rejectnull"][jj][3][0][0])
                                    tstats.append(dat["control_study_rejectnull"][jj][4][0][0])
                                    months.append(dateTstamp.month)    
                                    if comm == "corn":
                                        period.append(set_period_corn(dateTstamp))
                                    else:
                                        period.append(set_period_wheat(dateTstamp))
                                    year.append(dateTstamp.year)                                    
                                    reportsall.append(report)
                                    lagstruct.append(mlag)
                                    commods.append(comm)
                                    contractsall.append(contract)
                                    marks.append(process)
                                    yearmonth.append("{} {}".format(year[-1], monthsstr[months[-1]]))
                                    timescales.append(timescale)                                    
                                    rdays.append(dateTstamp.strftime(format = "%d %Y %b"))
                                    # print(report, timescale)
                                     
    pdout = pd.DataFrame.from_dict({"year" : year,
                                    "months" : months,                                    
                                    "reportsall" : reportsall,
                                    "lagstruct" : lagstruct,
                                    "commodity" : commods,
                                    "contracts" : contractsall,
                                    "marketproc" : marks,
                                    "reject" : reject,
                                    "oneminusp" : oneminusp,
                                    "tstatistic" : tstats,
                                    "farmingperiod" : period,
                                    "timescales": timescales,
                                    "reportdate" : rdays})
    pdout.to_csv("{}studyIV_data.csv".format(DIR_out), index=False)