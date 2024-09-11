import json
import pickle
import time
import datetime
import pandas as pd
from openpyxl import load_workbook
import pathlib
import string
import numpy as np
import re
from collections import Counter
import sys
import plotly.graph_objects as go
from agribusiness.src.timeseries_construction_sentiment import quick_run
from agribusiness.src.text_utils_agr import preprocessor_builder, fix_plot_layout_and_save


def flatten(l):
    return [item for sublist in l for item in sublist]

def postclean(dframe, comm, dirout="/tmp/"):
    # plotting text data statistics

    months = {1:"Jan", 2:"Feb", 3:"Mar", 4:"Apr", 5:"May", 6:"Jun", 7:"Jul", 8:"Aug", 9:"Sep", 10:"Oct", 11:"Nov", 12:"Dec"}

    print(len(dframe))
    keywordpresencewheat = lambda x: x if "wheat" in x.lower() or "bushel" in x.lower() or "zwz2" in x.lower() \
                    or "zwt" in x.lower() or "zwh3" in x.lower() \
                    or "zwk3" in x.lower() or "zwn3" in x.lower() \
                    or "zwu3" in x.lower() or "zwz3" in x.lower() \
                    or "zwh4" in x.lower() or "zwk4" in x.lower() \
                    or "zwn4" in x.lower() or "zwu4" in x.lower() \
                    or "zwz4" in x.lower() or "zwh5" in x.lower() \
                    or "zwk5" in x.lower() or "zwn5" in x.lower() \
                    or "kez2" in x.lower() \
                    or "ket" in x.lower() or "keh3" in x.lower() \
                    or "kek3" in x.lower() or "ken3" in x.lower() \
                    or "keu3" in x.lower() or "kez3" in x.lower() \
                    or "keh4" in x.lower() or "kek4" in x.lower() \
                    or "ken4" in x.lower() or "keu4" in x.lower() \
                    or "kez4" in x.lower() or "keh5" in x.lower() \
                    or "kek5" in x.lower() or "ken5" in x.lower() else np.nan

    keywordpresencecorn = lambda x: x if "corn" in x.lower() or "bushel" in x.lower() or "zcz2" in x.lower() \
                    or "zct" in x.lower() or "zch3" in x.lower() \
                    or "zck3" in x.lower() or "zcn3" in x.lower() \
                    or "zcu3" in x.lower() or "zcz3" in x.lower() \
                    or "zch4" in x.lower() or "zck4" in x.lower() \
                    or "zcn4" in x.lower() or "zwu4" in x.lower() \
                    or "zcz4" in x.lower() or "zch5" in x.lower() \
                    or "zck5" in x.lower() or "zcn5" in x.lower() \
                    or "zcu5" in x.lower() or "zcz5" in x.lower() \
                    or "zcz6" in x.lower() or "zcn6" in x.lower() else np.nan

    for i in dframe.text:
        if comm == "wheat":
            if not isinstance(keywordpresencewheat(i), list) and not isinstance(keywordpresencewheat(i), str):
                print(i)
        else:
            if not isinstance(keywordpresencecorn(i), list) and not isinstance(keywordpresencecorn(i), str):
                print(i)

    if comm == "wheat":
        dframe.text = dframe.text.apply(keywordpresencewheat)
    else:
        dframe.text = dframe.text.apply(keywordpresencecorn)
    dframe = dframe.drop(dframe[dframe.text.isnull()].index).reset_index(drop=True)

    print(len(dframe))

    articles = dict()
    sentences = dict()
    tokens = dict()

    articlesy = dict()
    sentencesy = dict()
    tokensy = dict()

    for i in range(1, 13, 1):
        articles[months[i]] = 0
        sentences[months[i]] = 0
        tokens[months[i]] = 0
    
    for i in range(len(dframe)):
        
        row = dframe.loc[i]
        month = row["date"].month
        year = row["date"].year
        if str(year) not in articlesy.keys():
            articlesy[str(year)] = 0
        if str(year) not in sentencesy.keys():
            sentencesy[str(year)] = 0
        if str(year) not in tokensy.keys():
            tokensy[str(year)] = 0
        t = len(row.tokens)
        s = len(row.sentences)
        articles[months[month]] += 1
        sentences[months[month]] += s
        tokens[months[month]] += t
        articlesy[str(year)] += 1
        sentencesy[str(year)] += s
        tokensy[str(year)] += t

    print("Total articles: {}".format(np.sum(list(articlesy.values()))))
    print("Total sentences: {}".format(np.sum(list(sentencesy.values()))))
    print("Total tokens: {}".format(np.sum(list(tokensy.values()))))
    
    dfa_m = pd.DataFrame(articles, index=[0])
    dft_m = pd.DataFrame(tokens, index=[0])
    dfs_m = pd.DataFrame(sentences, index=[0])
    fig_m = go.Figure(data=[
                        go.Bar(name='Articles', x=list(months.values()), y=dfa_m.values[0],
                               marker_color="darkgreen"),
                        go.Bar(name='Sentences', x=list(months.values()), y=dfs_m.values[0],
                               marker_color="goldenrod"),
                        go.Bar(name='Tokens', x=list(months.values()), y=dft_m.values[0],
                               marker_color="lightblue")
                    ])
    savename = "{}{}_monthlytextdata.html".format(dirout, comm)
    fix_plot_layout_and_save(fig_m, savename, xaxis_title="Month", yaxis_title="Volume of text data",
                                title="", showgrid=False,
                                showlegend=True,
                                print_png=True)
    
    dfa_y = pd.DataFrame(articlesy, index=[0])
    dft_y = pd.DataFrame(tokensy, index=[0])
    dfs_y = pd.DataFrame(sentencesy, index=[0])
    fig_y = go.Figure(data=[
                        go.Bar(name='Articles', x=sorted(list(articlesy.keys())), y=dfa_y.values[0],
                               marker_color="darkgreen"),
                        go.Bar(name='Sentences', x=sorted(list(articlesy.keys())), y=dfs_y.values[0],
                               marker_color="goldenrod"),
                        go.Bar(name='Tokens', x=sorted(list(articlesy.keys())), y=dft_y.values[0],
                               marker_color="lightblue")
                    ])
    savename = "{}{}_yearlytextdata.html".format(dirout, comm)
    fix_plot_layout_and_save(fig_y, savename, xaxis_title="Month", yaxis_title="Volume of text data",
                                title="", showgrid=False,
                                showlegend=True,
                                print_png=True)

    return dframe

def remove_nontext_heuristic(dframe):

    from collections import Counter

    keepidx = []
    indx = dframe[dframe["source"] == "Dow Jones Institutional News"].index.values.tolist()
    for i in indx:
        cnts = Counter(dframe.text.loc[i].lower().split())
        if cnts["the"] >= 0.04 * len(''.join([j for j in dframe.text.loc[i] if not j.isdigit()]).split()):
           keepidx.append(i)
        else:
           if cnts["and"] >= 0.02 * len(''.join([j for j in dframe.text.loc[i] if not j.isdigit()]).split()) or cnts["a"] >= 0.02 * len(''.join([j for j in dframe.text.loc[i] if not j.isdigit()]).split()):
              keepidx.append(i)
           #else:
           #   print(dframe.text.loc[i])
    idxextra = dframe[dframe["source"] !="Dow Jones Institutional News"].index.values.tolist()
    keepidx.extend(idxextra)

    return dframe.iloc[keepidx]

def extract_dowjones_data_xslx():

    pwd = pathlib.Path.cwd()
    wb = load_workbook(filename="{}/data_text/DJ.xlsx".format(pwd), read_only=True)
    DIR_in = pwd
    corn = dict(text=[], date=[], source=[])
    wheat = dict(text=[], date=[], source=[])
    start = -1
    dictionary = "usdacmecftc"
    wordlist1 = "usda_cme_cftc_trie.pickle"
    wordlist2 = "usda_cme_cftc.pickle"
    topic_basis = "{}/data_nlp/{}".format(DIR_in, wordlist1)
    word_basis= "{}/data_nlp/{}".format(DIR_in, wordlist2)
    stopwords = "{}/data_nlp/special_stopwords.txt".format(DIR_in)
    DIR_out = "{}/dowjones_data_cleantext_{}/".format(DIR_in, dictionary)
    pathlib.Path(DIR_out).mkdir(parents=True, exist_ok=True)
    replace_multilines = lambda x: re.sub(r'\n+', '. ', x)
    replace_dots_spaces = lambda x: re.sub(r'\.\s*[\s*\.*]\s*\.*', '. ', x)
    replace_escapeapostr = lambda x: re.sub(r"\'s", "", x)

    for row in wb["Sheet1"]:
        start += 1
        if start == 0:
            continue
        date = row[-1].value
        
        cells = row
        news_text = cells[1].value
        news_source = cells[2].value
        news_date = cells[-1].value

        news_text_lower = news_text.lower() 
        if cells[-2].value == 1 and cells[-3].value == 0:
            # wheat
            
            if news_text_lower is None or "wheat" not in news_text_lower or \
                "bushel" not in news_text_lower or "zwz2" not in news_text_lower \
                    or "zwt" not in news_text_lower or "zwh3" not in news_text_lower \
                    or "zwk3" not in news_text_lower or "zwn3" not in news_text_lower \
                    or "zwu3" not in news_text_lower or "zwz3" not in news_text_lower \
                    or "zwh4" not in news_text_lower or "zwk4" not in news_text_lower \
                    or "zwn4" not in news_text_lower or "zwu4" not in news_text_lower \
                    or "zwz4" not in news_text_lower or "zwh5" not in news_text_lower \
                    or "zwk5" not in news_text_lower or "zwn5" not in news_text_lower \
                    or "kez2" not in news_text_lower \
                    or "ket" not in news_text_lower or "keh3" not in news_text_lower \
                    or "kek3" not in news_text_lower or "ken3" not in news_text_lower \
                    or "keu3" not in news_text_lower or "kez3" not in news_text_lower \
                    or "keh4" not in news_text_lower or "kek4" not in news_text_lower \
                    or "ken4" not in news_text_lower or "keu4" not in news_text_lower \
                    or "kez4" not in news_text_lower or "keh5" not in news_text_lower \
                    or "kek5" not in news_text_lower or "ken5" not in news_text_lower:
                continue
            wheat["text"].append(news_text)
            wheat["date"].append(news_date)
            wheat["source"].append(news_source)
        elif cells[-2].value == 0 and cells[-3].value == 1:
            # corn
            if "concert" in news_text_lower or "kiwanuka" in news_text_lower \
                or news_text_lower is None or "corn" not in news_text_lower or \
                "bushel" not in news_text_lower or "zcz2" not in news_text_lower \
                    or "zct" not in news_text_lower or "zch3" not in news_text_lower \
                    or "zck3" not in news_text_lower or "zcn3" not in news_text_lower \
                    or "zcu3" not in news_text_lower or "zcz3" not in news_text_lower \
                    or "zch4" not in news_text_lower or "zck4" not in news_text_lower \
                    or "zcn4" not in news_text_lower or "zwu4" not in news_text_lower \
                    or "zcz4" not in news_text_lower or "zch5" not in news_text_lower \
                    or "zck5" not in news_text_lower or "zcn5" not in news_text_lower \
                    or "zcu5" not in news_text_lower or "zcz5" not in news_text_lower \
                    or "zcz6" not in news_text_lower or "zcn6" not in news_text_lower :
                continue
            corn["text"].append(news_text)
            corn["date"].append(news_date)
            corn["source"].append(news_source)

    wheat_pd = pd.DataFrame.from_dict(wheat)
    print(len(wheat_pd))
    wheat_pd = wheat_pd.sort_values(by="date").drop_duplicates(subset=["text"]).reset_index(drop=True)
    corn_pd = pd.DataFrame.from_dict(corn)
    print(len(corn_pd))
    corn_pd = corn_pd.sort_values(by="date").drop_duplicates(subset=["text"]).reset_index(drop=True)
    print("Remove duplicates")
    print(len(wheat_pd))
    print(len(corn_pd))

    with open("{}/wheat_dataframe.pickle".format(DIR_out), "wb") as f:
        pickle.dump(wheat_pd, f, pickle.HIGHEST_PROTOCOL)
    with open("{}/corn_dataframe.pickle".format(DIR_out), "wb") as f:
        pickle.dump(corn_pd, f, pickle.HIGHEST_PROTOCOL)
    
    # Cleaning functions
    token_removal_corn = lambda x: x if "corn" in x and Counter(x)["corn"] > 1 else np.nan
    token_removal_wheat = lambda x: x if "wheat" in x and Counter(x)["wheat"] > 1 else np.nan
    isnumber = lambda x: x.isnumeric()
    empty_token_removal = lambda x: [i for i in x if i != "" and not isnumber(i)]
    empty_tokenlist_removal = lambda x: x if len(x) > 0 else np.nan
    splitstr = lambda x: x.split("@")
    lowerstr = lambda x: x.lower()    

    # with open("{}/wheat_dataframe.pickle".format(DIR_out), "rb") as f:
    #     wheat_pd = pickle.load(f)
    # with open("{}/corn_dataframe.pickle".format(DIR_out), "rb") as f:
    #     corn_pd = pickle.load(f)
    wheat_pd = remove_nontext_heuristic(wheat_pd)
    wheat_pd = wheat_pd.reset_index(drop=True)
    wheat_pd.text = wheat_pd.text.apply(replace_multilines)
    wheat_pd.text = wheat_pd.text.apply(replace_dots_spaces)
    
    corn_pd = remove_nontext_heuristic(corn_pd)
    corn_pd = corn_pd.reset_index(drop=True)
    corn_pd.text = corn_pd.text.apply(replace_multilines)
    corn_pd.text = corn_pd.text.apply(replace_dots_spaces)

    tokenizer = preprocessor_builder(spacy_model="en_core_web_lg",
                         spacy_disable_list=["ner", "parser", "tagger"],
                         character_set=string.printable,
                         special_chars=string.punctuation + '°',
                         disable_parse=True, 
                         remove_oov=False,
                         acronyms_f=None,
                         latin_f=None,
                         metric_units_f=None,
                         chemicals_f=None,
                         stopwords_f=stopwords,
                         compounds_f=None,
                         topic_basis=topic_basis,
                         word_basis=word_basis)
   
    with open("{}/data_nlp/{}".format(DIR_in, wordlist1), "rb") as f:
        word_basis_trie = pickle.load(f)
    sentences_extr = lambda x: tokenizer.segment_sentences(x, word_basis_trie) if (x is not None and len(x) > 10) else np.nan
    
    # CORN
    corn_pd.text = corn_pd.text.apply(replace_escapeapostr)
    corn_pd["tokens"] = corn_pd.text.apply(tokenizer.tokenize)
    with open("{}/corn_dataframe_withtokens_removeTHErule.pickle".format(DIR_out), "wb") as f:
        pickle.dump(corn_pd, f, pickle.HIGHEST_PROTOCOL)
    # Load tokenized dataframe: contains only text with the word "corn"
    # with open("{}/corn_dataframe_withtokens_removeTHErule.pickle".format(DIR_out), "rb") as f:
    #     corn_pd = pickle.load(f)
    print("Length of DF after tokenisation: {}".format(len(corn_pd)))
    # Keep rows with: token list containing "corn" more than once, have at least 1 non number token
    corn_pd.tokens = corn_pd.tokens.apply("@".join)
    corn_pd.tokens = corn_pd.tokens.apply(lowerstr)
    corn_pd.tokens = corn_pd.tokens.apply(splitstr)
    corn_pd.tokens = corn_pd.tokens.apply(token_removal_corn)
    corn_pd = corn_pd.drop(corn_pd[corn_pd.tokens.isnull()].index).reset_index(drop=True)
    corn_pd.tokens = corn_pd.tokens.apply(empty_token_removal)
    corn_pd.tokens = corn_pd.tokens.apply(empty_tokenlist_removal)
    corn_pd = corn_pd.drop(corn_pd[corn_pd.tokens.isnull()].index).reset_index(drop=True)
    # Extract sentences from remaining articles - use it for cleaning, not really using sentences at the moment
    corn_pd["sentences"] = corn_pd.text.apply(sentences_extr)
    # Remove sentences with length < 10 - get rid of mostly numeric sentences
    corn_pd = corn_pd.drop(corn_pd[corn_pd.sentences.isnull()].index).reset_index(drop=True)          
    # Remove duplicate entries with same sentences, tokens, date
    corn_pd.sentences = corn_pd.sentences.apply("@".join)
    corn_pd.tokens = corn_pd.tokens.apply("@".join)
    corn_pd = corn_pd.sort_values(by="date").drop_duplicates(subset=["date", "sentences", "tokens"]).reset_index(drop=True)
    corn_pd.sentences = corn_pd.sentences.apply(splitstr)    
    corn_pd.tokens = corn_pd.tokens.apply(splitstr)
    print("Length of DF after cleaning: {}".format(len(corn_pd)))
    corn_pd.sentences = corn_pd.sentences.apply("@".join)
    corn_pd.tokens = corn_pd.tokens.apply("@".join)
    corn_pd = corn_pd.sort_values(by="date").drop_duplicates(subset=["sentences", "tokens"]).reset_index(drop=True)
    corn_pd.sentences = corn_pd.sentences.apply(splitstr)
    corn_pd.tokens = corn_pd.tokens.apply(splitstr)
  
    with open("{}/corn_dataframe_postprocessed_removeTHErule.pickle".format(DIR_out), "wb") as f:
        pickle.dump(corn_pd, f, pickle.HIGHEST_PROTOCOL)

    # WHEAT
    print("WHEAT")
    wheat_pd.text = wheat_pd.text.apply(replace_escapeapostr)
    wheat_pd["tokens"] = wheat_pd.text.apply(tokenizer.tokenize)
    with open("{}/wheat_dataframe_withtokens_removeTHErule.pickle".format(DIR_out), "wb") as f:
        pickle.dump(wheat_pd, f, pickle.HIGHEST_PROTOCOL)
    print("Length of DF after tokenisation: {}".format(len(corn_pd)))
    wheat_pd.tokens = wheat_pd.tokens.apply("@".join)
    wheat_pd.tokens = wheat_pd.tokens.apply(lowerstr)
    wheat_pd.tokens = wheat_pd.tokens.apply(splitstr)
    wheat_pd.tokens = wheat_pd.tokens.apply(token_removal_wheat)
    wheat_pd = wheat_pd.drop(wheat_pd[wheat_pd.tokens.isnull()].index).reset_index(drop=True)
    wheat_pd.tokens = wheat_pd.tokens.apply(empty_token_removal)
    wheat_pd.tokens = wheat_pd.tokens.apply(empty_tokenlist_removal)
    wheat_pd = wheat_pd.drop(wheat_pd[wheat_pd.tokens.isnull()].index).reset_index(drop=True)
    wheat_pd["sentences"] = wheat_pd.text.apply(sentences_extr)
    wheat_pd = wheat_pd.drop(wheat_pd[wheat_pd.sentences.isnull()].index).reset_index(drop=True)          
    wheat_pd.sentences = wheat_pd.sentences.apply("@".join)
    wheat_pd.tokens = wheat_pd.tokens.apply("@".join)
    wheat_pd = wheat_pd.sort_values(by="date").drop_duplicates(subset=["date", "sentences", "tokens"]).reset_index(drop=True)
    wheat_pd.sentences = wheat_pd.sentences.apply(splitstr)    
    wheat_pd.tokens = wheat_pd.tokens.apply(splitstr)
    print("Length of DF after cleaning: {}".format(len(wheat_pd)))
    wheat_pd.sentences = wheat_pd.sentences.apply("@".join)
    wheat_pd.tokens = wheat_pd.tokens.apply("@".join)
    wheat_pd = wheat_pd.sort_values(by="date").drop_duplicates(subset=["sentences", "tokens"]).reset_index(drop=True)
    wheat_pd.sentences = wheat_pd.sentences.apply(splitstr)
    wheat_pd.tokens = wheat_pd.tokens.apply(splitstr)
    with open("{}/wheat_dataframe_postprocessed_removeTHErule.pickle".format(DIR_out), "wb") as f:
        pickle.dump(wheat_pd, f, pickle.HIGHEST_PROTOCOL)


    print("Sentence removal - heuristic cleaning")
    print(len(corn_pd))
    print(len(wheat_pd))


if __name__ == '__main__':

    pwd = pathlib.Path.cwd()
    DIR = pwd
    DIR_out = pwd

    sentiment_extraction = False
    if sentiment_extraction is False:
        # reading and cleaning raw DJ data
        extract_dowjones_data_xslx()
    else:
        kwargs_list = ["corn", "wheat"]
        sites = ["DowJones"]
        # Load main config file
        with open("{}/src/config_dframes.json".format(DIR), "rt") as config_file:  
            cfg = json.load(config_file)

        t0 = time.time()
        cfg["run_output_folder"] = DIR_out
        cfg["save_token_distributions"] = False
        cfg["adjust_sentiment"] = False
        for top in range(len(kwargs_list)):
            for site in sites:
                comm = kwargs_list[top]
                quick_run(cfg)

        t1 = time.time()
        print("Time to completion: " + str(datetime.timedelta(seconds=t1-t0)))
