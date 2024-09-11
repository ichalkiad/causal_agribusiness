import pickle
import os
import numpy as np
import logging
import time
from plotly import io as pio
pio.renderers.default = "browser"
from pathlib import Path
import pandas as pd
from agribusiness.src.bow import BoWEmbedder
from agribusiness.src.text_feature_extractors import bow_per_token_empirical_entropy, \
                                                        bow_recursive_avg_cumulative_frequencies, get_sentiment_embedding, \
                                                                            bow_recursive_avg_token_frequencies, word_ngrams
from agribusiness.src.text_utils_agr import save_sparse_oneoff, load_sparse_oneoff, get_sentiment_counts, load_config





def quick_run(kwargs_cfg, dates_dict=None):
    # Create output directory
    if not os.path.exists(kwargs_cfg["run_output_folder"]):
        Path(kwargs_cfg["run_output_folder"]).mkdir(parents=True, exist_ok=True)
    if not os.path.exists("{}/{}".format(kwargs_cfg["run_output_folder"], kwargs_cfg["run_output_name"])):
        Path("{}/{}".format(kwargs_cfg["run_output_folder"], kwargs_cfg["run_output_name"])).mkdir(parents=True,
                                                                                                    exist_ok=True)

    if kwargs_cfg["save_token_distributions"] and not os.path.exists("{}/{}/{}".format(kwargs_cfg["run_output_folder"],
                                                                     kwargs_cfg["run_output_name"],
                                                                     kwargs_cfg["save_counts_distribution"])):
        Path("{}/{}/{}".format(kwargs_cfg["run_output_folder"], kwargs_cfg["run_output_name"],
                    kwargs_cfg["save_counts_distribution"])).mkdir(parents=True, exist_ok=True)
    if kwargs_cfg["save_token_distributions"] and not os.path.exists("{}/{}/{}".format(kwargs_cfg["run_output_folder"],
                                                                     kwargs_cfg["run_output_name"],
                                                                     kwargs_cfg["save_recursive_avg_distribution"])):
        Path("{}/{}/{}".format(kwargs_cfg["run_output_folder"], kwargs_cfg["run_output_name"],
                    kwargs_cfg["save_recursive_avg_distribution"])).mkdir(parents=True, exist_ok=True)

    # Save config
    logging.basicConfig(filename="{}/{}/run_commodity.log".format(kwargs_cfg["run_output_folder"],
                                                             kwargs_cfg["run_output_name"]), level=logging.INFO)
    logging.info(kwargs_cfg)

    DIR = kwargs_cfg["input_data"]
    comm = kwargs_cfg["comm"]
    source = kwargs_cfg["source"]

    # Open file for count matrices
    counts_path = "{}/{}/{}_{}_counts.pickle".format(kwargs_cfg["run_output_folder"], kwargs_cfg["run_output_name"],
                                                     source, comm)
    if kwargs_cfg["adjust_sentiment"]:
        with open(kwargs_cfg["positive_sentiment_trie"], "rb") as g:
            sent_pos = pickle.load(g)
        with open(kwargs_cfg["negative_sentiment_trie"], "rb") as g:
            sent_neg = pickle.load(g)
        with open(kwargs_cfg["neutral_sentiment_trie"], "rb") as g:
            sent_neutral = pickle.load(g)

    # Load input data
    with open("{}/{}_dataframe_postprocessed_removeTHErule.pickle".format(DIR, kwargs_cfg["comm"]), "rb") as f:
        reporting = pickle.load(f)
    with open("{}/{}_dataframe_postprocessed_removeTHErule.pickle".format(DIR, kwargs_cfg["comm"]), "rb") as f:
        reporting_raw = pickle.load(f)

    kwargs = kwargs_cfg["frequency_based_features"]
    kwargs["context"] = "global"
    kwargs["run_output_folder"] = kwargs_cfg["run_output_folder"]
    kwargs["run_output_name"] = kwargs_cfg["run_output_name"]
    kwargs["save_recursive_avg_distribution"] = kwargs_cfg["save_recursive_avg_distribution"]

    with open("{}/{}".format(kwargs_cfg["input_nlp_data"], kwargs_cfg["topic_trie"]), "rb") as f:
        word_basis_trie = pickle.load(f)
    with open("{}/{}".format(kwargs_cfg["input_nlp_data"], kwargs_cfg["topic"]), "rb") as f:
        word_basis = pickle.load(f)
        assert isinstance(word_basis, list)
    kwargs["projection_basis"] = word_basis

    daystamp = []
    ngram_title = []
    ngram_author = []
    ngram_allwords_post = []
    ngram_total_entr = []
    ngram_total_freq = []

    if kwargs_cfg["adjust_sentiment"]:
        logging.info("Sentiment will be produced for positive, negative and neutral polarity.")
        daystamp_pos = []
        ngram_title_pos = []
        ngram_author_pos = []
        ngram_allwords_post_pos = []
        ngram_pos_entr = []
        ngram_pos_freq = []
        daystamp_neg = []
        ngram_title_neg = []
        ngram_author_neg = []
        ngram_allwords_post_neg = []
        ngram_neg_entr = []
        ngram_neg_freq = []
        daystamp_neutral = []
        ngram_title_neutral = []
        ngram_author_neutral = []
        ngram_allwords_post_neutral = []
        ngram_neutral_entr = []
        ngram_neutral_freq = []

        if kwargs_cfg["save_counts"]:
            acc_counts_pos = np.zeros((len(word_basis, )))
            acc_counts_neg = np.zeros((len(word_basis, )))
            acc_counts_neutral = np.zeros((len(word_basis, )))

        try:
            acc_counts_pos_tmp = load_sparse_oneoff(counts_path.replace(".pickle", "_positive.pickle"))
            if acc_counts_pos_tmp is not None:
                acc_counts_pos = acc_counts_pos_tmp
            else:
                logging.error("Error loading previous count file for positive sentiment.")
                # raise NotImplementedError
            acc_counts_neg_tmp = load_sparse_oneoff(counts_path.replace(".pickle", "_negative.pickle"))
            if acc_counts_neg_tmp is not None:
                acc_counts_neg = acc_counts_neg_tmp
            else:
                logging.error("Error loading previous count file for negative sentiment.")
                # raise NotImplementedError
            acc_counts_neutral_tmp = load_sparse_oneoff(counts_path.replace(".pickle", "_neutral.pickle"))
            if acc_counts_neutral_tmp is not None:
                acc_counts_neutral = acc_counts_neutral_tmp
            else:
                logging.error("Error loading previous count file for neutral sentiment.")
                # raise NotImplementedError
        except:
            # Exit here when it runs on server
            print("Error loading previous count file for polarities.")
            logging.error("Error loading previous count file for polarities.")
            raise NotImplementedError

    index = 0

    if kwargs_cfg["save_counts"]:
        acc_counts = np.zeros((len(word_basis, )))
    try:
        acc_counts_tmp = load_sparse_oneoff(counts_path)
        if acc_counts_tmp is not None:
            acc_counts = acc_counts_tmp
    except:
        # Exit here when it runs on server as initial matrix of counts will be provided
        print("Error loading previous count file.")
        logging.error("Error loading previous count file.")
        raise NotImplementedError

    embedder = BoWEmbedder(kwargs["ngram_size"], kwargs["projection_basis"])
    start_date = reporting["date"].loc[0]
    end_date = reporting["date"].loc[len(reporting["date"])-1]
    t0 = time.time()

    for daily_articles_idx in range(len(reporting["text"])):

        daily_articles = reporting.loc[daily_articles_idx]
        # extract tokens
        pp_tokens = daily_articles["tokens"]
        # make sure all are lowercase as long as our bases are lowercase
        pp_tokens = [w.lower() if not w.islower() else w for w in pp_tokens]

        # Remove OOV words
        pp_tokens_in_dict = []

        for w in pp_tokens:
            if w in word_basis_trie:
                pp_tokens_in_dict.append(w)

        # should be the same if we have removed OOV during text post-processing
        # assert pp_tokens_old==pp_tokens_in_dict
        pp_tokens = pp_tokens_in_dict
        datetime = daily_articles["date"]

        date = str(datetime).split()[0]

        raw_content = reporting_raw.loc[daily_articles_idx]["text"]

        postproc_ngrams = word_ngrams(pp_tokens, ngram_range=(kwargs["ngram_size"], kwargs["ngram_size"]),
                                      step=kwargs["ngram_size"])

        for ngram in postproc_ngrams:

            # article date
            daystamp.append(date)
            try:
                ngram_title.append(reporting_raw.loc[daily_articles_idx, "sentences"][0][0]) # first text sentence if title is not present
            except:
                try:
                    ngram_title.append(reporting_raw.loc[daily_articles_idx, "sentences"][0])
                except:
                    ngram_title.append("")
            ngram_author.append(reporting_raw.loc[daily_articles_idx, "source"][0])
            ngram_allwords_post.append(ngram)

            counts_ngram = embedder.get_counts(ngram, kwargs["projection_basis"], kwargs)
            ts, token_freq_prev_step = bow_recursive_avg_token_frequencies(embedder=embedder, kwargs=kwargs,
                                                                           counts=counts_ngram,
                                                                           prev_counts=acc_counts)
            # Update the support of the words' distribution. Can be all basis, or only tokens that appear e.g.
            # k times depending on binary relation
            acc_counts += token_freq_prev_step
            # save running sum of counts
            if kwargs_cfg["save_counts"]:
                save_sparse_oneoff(acc_counts, counts_path)

            # Convert token frequencies to density. Note that the recursive proportions are computed over the
            # complete basis and all eligible (according to our condition) seen tokens, but the density whose entropy
            # we are using is based only on the ngram tokens, it is the density of the ngram token proportions
            # rather than the density of basis token proportions at the current time step.

            ngram_token_density = np.true_divide(np.array(ts), np.sum(ts)) if np.sum(ts) > 1e-10 else \
                np.zeros(np.array(ts).shape)
            try:
                assert (abs(np.sum(ngram_token_density) - 1) < 1e-10 if np.sum(ngram_token_density) > 1e-10
                    else abs(np.sum(ngram_token_density) < 1e-10))
            except AssertionError:
                logging.error("Error in density computation - numerical or more serious?...exiting")
                raise NotImplementedError

            entropy = bow_per_token_empirical_entropy(embedder, ngram_token_density)

            # collect abs. magnitude entropy
            ngram_total_entr.append(entropy)

            # recursive cumulative frequencies
            # token_freq_prev_step is current token freq embedding as assigned above
            # print(token_freq_prev_step)
            ts = bow_recursive_avg_cumulative_frequencies(ts)
            ngram_total_freq.append(ts)

            if kwargs_cfg["adjust_sentiment"]:
                counts_ngram_sentiment, pos_counts, neg_counts, neutral_counts, polarity \
                    = get_sentiment_counts(ngram, counts_ngram, sent_pos, sent_neg, sent_neutral)
                # print(counts_ngram_sentiment, pos_counts, neg_counts, neutral_counts, polarity)
                # entropy of positive words
                ts, token_freq_pos_prev_step, entropy, cumulative_ts \
                    = get_sentiment_embedding(pos_counts, embedder, index, kwargs, acc_counts_pos,
                                              sentiment="pos", save=False)  # kwargs_cfg["save_token_distributions"])
                if entropy != -1 and cumulative_ts != -1:
                    acc_counts_pos += token_freq_pos_prev_step
                    if kwargs_cfg["save_counts"]:
                        save_sparse_oneoff(acc_counts_pos, counts_path.replace(".pickle", "_positive.pickle"))

                    daystamp_pos.append(date)
                    ngram_title_pos.append(raw_content["Title"][0])
                    ngram_author_pos.append(raw_content["Author"][0])
                    ngram_allwords_post_pos.append(ngram)
                    ngram_pos_entr.append(entropy)
                    ngram_pos_freq.append(cumulative_ts)

                ts, token_freq_neg_prev_step, entropy, cumulative_ts \
                    = get_sentiment_embedding(neg_counts, embedder, index, kwargs, acc_counts_neg,
                                              sentiment="neg", save=False)  # kwargs_cfg["save_token_distributions"])
                if entropy != -1 and cumulative_ts != -1:
                    acc_counts_neg += token_freq_neg_prev_step
                    if kwargs_cfg["save_counts"]:
                        save_sparse_oneoff(acc_counts_neg, counts_path.replace(".pickle", "_negative.pickle"))
                    daystamp_neg.append(date)
                    ngram_title_neg.append(raw_content["Title"][0])
                    ngram_author_neg.append(raw_content["Author"][0])
                    ngram_allwords_post_neg.append(ngram)
                    ngram_neg_entr.append(entropy)
                    ngram_neg_freq.append(cumulative_ts)

                ts, token_freq_neutral_prev_step, entropy, cumulative_ts \
                    = get_sentiment_embedding(neutral_counts, embedder, index, kwargs, acc_counts_neutral,
                                              sentiment="neutral",
                                              save=False)  # kwargs_cfg["save_token_distributions"])
                if entropy != -1 and cumulative_ts != -1:
                    acc_counts_neutral += token_freq_neutral_prev_step
                    if kwargs_cfg["save_counts"]:
                        save_sparse_oneoff(acc_counts_neutral, counts_path.replace(".pickle", "_neutral.pickle"))
                    daystamp_neutral.append(date)
                    ngram_title_neutral.append(raw_content["Title"][0])
                    ngram_author_neutral.append(raw_content["Author"][0])
                    ngram_allwords_post_neutral.append(ngram)
                    ngram_neutral_entr.append(entropy)
                    ngram_neutral_freq.append(cumulative_ts)

            index += 1
            if index % 5000 == 0:
               print(index)
               

    if kwargs_cfg["adjust_sentiment"]:
        if len(ngram_neutral_entr) > 0:

            dataout = pd.DataFrame.from_dict({"Date": daystamp_neutral,
                                              "Title": ngram_title_neutral,
                                              "Author": ngram_author_neutral,
                                              "NgramPostprocessed": ngram_allwords_post_neutral,
                                              "CumulativeFreq": ngram_neutral_freq,
                                              "Entropy": ngram_neutral_entr
                                              })
            dfoutname = "{}/{}/{}_{}_ngram_sentiment_neutral.csv".format(kwargs_cfg["run_output_folder"],
                                                                       kwargs_cfg["run_output_name"],
                                                                       kwargs_cfg["start_date"],
                                                                       kwargs_cfg["end_date"])
            dataout.to_csv(dfoutname, index=False)

        if len(ngram_pos_entr) > 0:
            dataout = pd.DataFrame.from_dict({"Date": daystamp_pos,
                                              "Title": ngram_title_pos,
                                              "Author": ngram_author_pos,
                                              "NgramPostprocessed": ngram_allwords_post_pos,
                                              "CumulativeFreq": ngram_pos_freq,
                                              "Entropy": ngram_pos_entr
                                              })
            dfoutname = "{}/{}/{}_{}_ngram_sentiment_positive.csv".format(kwargs_cfg["run_output_folder"],
                                                                            kwargs_cfg["run_output_name"],
                                                                            kwargs_cfg["start_date"],
                                                                            kwargs_cfg["end_date"])
            dataout.to_csv(dfoutname, index=False)

        if len(ngram_neg_entr) > 0:
            dataout = pd.DataFrame.from_dict({"Date": daystamp_neg,
                                              "Title": ngram_title_neg,
                                              "Author": ngram_author_neg,
                                              "NgramPostprocessed": ngram_allwords_post_neg,
                                              "CumulativeFreq": ngram_neg_freq,
                                              "Entropy": ngram_neg_entr
                                              })
            dfoutname = "{}/{}/{}_{}_ngram_sentiment_negative.csv".format(kwargs_cfg["run_output_folder"],
                                                                             kwargs_cfg["run_output_name"],
                                                                             kwargs_cfg["start_date"],
                                                                             kwargs_cfg["end_date"])
            dataout.to_csv(dfoutname, index=False)

    # entropy using non sentiment adjusted counts
    dataout = pd.DataFrame.from_dict({"Date": daystamp,
                                      "Title": ngram_title,
                                      "Author": ngram_author,
                                      "NgramPostprocessed": ngram_allwords_post,
                                      "CumulativeFreq": ngram_total_freq,
                                      "Entropy": ngram_total_entr
                                      })
    dfoutname = "{}/{}/{}_{}_ngram_sentiment_total.csv".format(kwargs_cfg["run_output_folder"],
                                                                     kwargs_cfg["run_output_name"],
                                                                     kwargs_cfg["start_date"],
                                                                     kwargs_cfg["end_date"])
    dataout.to_csv(dfoutname, index=False)

    logging.info("Time elapsed: {}".format(time.time() - t0))
    if dates_dict is not None:
        dates_dict["{}_{}".format(source, comm)] = {"start": start_date,
                                                "end": end_date}

    return dates_dict


if __name__ == '__main__':
    kwargs_cfg = load_config()
    quick_run(kwargs_cfg)
