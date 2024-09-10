import logging
from bs4 import BeautifulSoup
from urllib.request import Request, urlopen
from urllib.error import HTTPError
import numpy as np
import pandas as pd
import uuid
from webdriver_manager.chrome import ChromeDriverManager
from selenium import webdriver
import pathlib
import pickle


MONTHS = {"Jan": "01", "Feb": "02", "Mar": "03", "Apr": "04", "May": "05", "Jun": "06", "Jul": "07",
          "Aug": "08", "Sep": "09", "Oct": "10", "Nov": "11", "Dec": "12"}


def get_date(datetxt):

    datetxt = datetxt.replace(",", "").split()
    month = MONTHS[datetxt[0]]
    day = datetxt[1]
    if len(day) == 1:
        day = "0" + day
    year = datetxt[2]

    date = np.datetime64("{}-{}-{}".format(year, month, day))

    return date


def process_page(url):

    print(url)
    chrome_options = webdriver.ChromeOptions()
    chrome_options.add_argument('--headless')
    chrome_options.headless = True
    chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
    chrome_options.add_experimental_option('useAutomationExtension', False)
    chrome_options.add_argument("window-size=1280,800")
    chrome_options.add_argument("user-agent=Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) "
                                "Chrome/44.0.2403.157 Safari/537.36")
    browser = webdriver.Chrome(ChromeDriverManager().install(), options=chrome_options)
    browser.create_options()

    req = Request(url, headers={'User-Agent': 'Mozilla/5.0', 'Connection': 'keep-alive',
                        'Pragma': 'no-cache', 'Cache-Control': 'no-cache',
                        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8'})
    try:
        ret = urlopen(req)
        http_code = ret.getcode()
    except HTTPError as err:
        browser.quit()
        return None, err.code
    else:
        print(http_code)
    webpage = ret.read()
    soup = BeautifulSoup(webpage, "html.parser")

    datelist = []
    dates_part = soup.find("table",  class_="release-items")
    date_col = dates_part.find_all("tr",  class_="release attributes row")
    for i in date_col:
        date = i.find("td", class_="attribute date_uploaded").get_text().strip()
        date = get_date(date)
        datelist.append(date)
    print(datelist)
    try:
        browser.quit()
    except:
        logging.error("Browser exiting failure?")
        raise NotImplementedError

    # print(datelist)
    return datelist


def process_usda(url, page, max_pages):

    p_prev = 1
    datelist = []
    for p in range(page, max_pages):
        print(p)
        url = url.replace("page={}".format(p_prev), "page={}".format(p))
        try:
            datelist.extend(process_page(url))
            p_prev = p
        except:
            print('Error in page {}'.format(p))
        # print(p_prev)

    return datelist


def process_page_thesaurus(url):

    print(url)
    chrome_options = webdriver.ChromeOptions()
    chrome_options.add_argument('--headless')
    chrome_options.headless = True
    chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
    chrome_options.add_experimental_option('useAutomationExtension', False)
    chrome_options.add_argument("window-size=1280,800")
    chrome_options.add_argument("user-agent=Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) "
                                "Chrome/44.0.2403.157 Safari/537.36")
    browser = webdriver.Chrome(ChromeDriverManager().install(), options=chrome_options)
    browser.create_options()

    req = Request(url, headers={'User-Agent': 'Mozilla/5.0', 'Connection': 'keep-alive',
                        'Pragma': 'no-cache', 'Cache-Control': 'no-cache',
                        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8'})
    try:
        ret = urlopen(req)
        http_code = ret.getcode()
    except HTTPError as err:
        browser.quit()
        return None, err.code
    else:
        print(http_code)
    webpage = ret.read()
    soup = BeautifulSoup(webpage, "html.parser")

    termslist = []
    terms = soup.find("main").find("div", class_="content").find("div", {"id": "termlist"}).find_all("li")
    for l in terms:
        term = l.get_text().strip().lower()
        if term != " " and term not in termslist:
            termslist.append(term)
    print(termslist[-1])
    try:
        browser.quit()
    except:
        logging.error("Browser exiting failure?")
        raise NotImplementedError

    return termslist


if __name__ == "__main__":

    pwd = pathlib.Path.cwd()
    DIR_out = "{}/cbot_data/".format(pwd) 
    

    url = "https://usda.library.cornell.edu/concern/publications/tm70mv177?locale=en&page=1&release_end_date=2022-06-30&release_start_date=1970-01-01#release-items"
    datelist = process_usda(url,  page=1, max_pages=65)
    print(datelist)
    cpm = pd.DataFrame.from_dict({"dates": datelist})
    cpm.to_csv("{}/cpm_alldates.csv".format(DIR_out))
    print(cpm)

    
    url = "https://usda.library.cornell.edu/concern/publications/n870zq819?locale=en&page=1&release_end_date=2022-06-30&release_start_date=1970-01-01#release-items"
    datelist = process_usda(url, page=1, max_pages=3)
    print(datelist)
    wws = pd.DataFrame.from_dict({"dates": datelist})
    wws.to_csv("{}/wws_alldates.csv".format(DIR_out))
    print(wws)
    
    url = "https://usda.library.cornell.edu/concern/publications/fq977t77k?locale=en&page=1&release_end_date=2022-06-30&release_start_date=1970-01-01#release-items"
    datelist = process_usda(url,  page=1, max_pages=65)
    print(datelist)
    cpss = pd.DataFrame.from_dict({"dates": datelist})
    cpss.to_csv("{}/cpss_alldates.csv".format(DIR_out))
    print(cpss)

    url = "https://usda.library.cornell.edu/concern/publications/x633f100h?locale=en&page=1&release_end_date=2022-06-30&release_start_date=1970-01-01#release-items"
    datelist = process_usda(url, page=1, max_pages=7)
    print(datelist)
    pp = pd.DataFrame.from_dict({"dates": datelist})
    pp.to_csv("{}/pp_alldates.csv".format(DIR_out))
    print(pp)

    url = "https://usda.library.cornell.edu/concern/publications/j098zb09z?locale=en&page=1&release_end_date=2022-06-30&release_start_date=1970-01-01#release-items"
    datelist = process_usda(url, page=1, max_pages=6)
    print(datelist)
    acr = pd.DataFrame.from_dict({"dates": datelist})
    acr.to_csv("{}/acr_alldates.csv".format(DIR_out))
    print(acr)

    url = "https://usda.library.cornell.edu/concern/publications/k3569432s?locale=en&page=1&release_end_date=2022-06-30&release_start_date=1970-01-01#release-items"
    datelist = process_usda(url, page=1, max_pages=7)
    print(datelist)
    cpas = pd.DataFrame.from_dict({"dates": datelist})
    cpas.to_csv("{}/cpas_alldates.csv".format(DIR_out))
    print(cpas)

    url = "https://usda.library.cornell.edu/concern/publications/5t34sj573?locale=en&page=1&release_end_date=2022-06-30&release_start_date=1970-01-01#release-items"
    datelist = process_usda(url, page=1, max_pages=4)
    print(datelist)
    sgas = pd.DataFrame.from_dict({"dates": datelist})
    sgas.to_csv("{}/sgas_alldates.csv".format(DIR_out))
    print(sgas)
    
    url = "https://usda.library.cornell.edu/concern/publications/3t945q76s?locale=en&page=1&release_end_date=2022-06-30&release_start_date=1970-01-01#release-items"
    datelist = process_usda(url, page=1, max_pages=67)
    print(datelist)
    wasde = pd.DataFrame.from_dict({"dates": datelist})
    wasde.to_csv("{}/wasde_alldates.csv".format(DIR_out))
    print(wasde)
    
    url = "https://usda.library.cornell.edu/concern/publications/xg94hp534?locale=en&page=1&release_end_date=2022-06-30&release_start_date=1970-01-01#release-items"
    datelist = process_usda(url, page=1, max_pages=22)
    print(datelist)
    gs = pd.DataFrame.from_dict({"dates": datelist})
    gs.to_csv("{}/gs_alldates.csv".format(DIR_out))
    print(gs)
    
    url = "https://usda.library.cornell.edu/concern/publications/8336h188j?locale=en&page=1&release_end_date=2022-06-30&release_start_date=1970-01-01#release-items"
    datelist = process_usda(url,  page=1, max_pages=98)
    print(datelist)
    cp = pd.DataFrame.from_dict({"dates": datelist})
    cp.to_csv("{}/cp_alldates.csv".format(DIR_out))
    print(cp)
    
    
    # USDA Thesaurus construction
    DIR_out = "{}/data_nlp/".format(pwd) 
    thesaurus = []
    for p in range(1, 2912, 1):
        url = "https://agclass.nal.usda.gov/mtwdk.exe?k=default&l=60&s=1&t=1&w=&cp={}&x=0&tt=0&n=50".format(p)
        thesaurus.extend(process_page_thesaurus(url))
    print(thesaurus)
    print(len(thesaurus))
    with open("{}usda_thesaurus.pickle".format(DIR_out), "wb") as f:
        pickle.dump(thesaurus, f, protocol=4) # 4 for opening by python 3.4
    
    with open("{}usda_thesaurus.pickle".format(DIR_out), "rb") as f:
            thes = pickle.load(f)

    thes_upd = []
    for w in thes:
        if "(" in w:
            ws = w.split("(")
            if ws[0].strip() not in thes_upd:
                thes_upd.append(ws[0].strip())
            if ws[1].replace(")", "").strip() not in thes_upd:
                thes_upd.append(ws[1].replace(")", "").strip())
            # print(w)
            # print(ws[0].strip(), ws[1].replace(")", "").strip())
        else:
            thes_upd.append(w)
    with open("{}usda_thesaurus_splitparenth.pickle".format(DIR_out), "wb") as f:
        pickle.dump(thes_upd, f, protocol=4) # 4 for opening by python 3.4
    






















