import numpy as np
from plotly import io as pio, express as px
import pandas as pd
import matplotlib.pyplot as plt


def bow_recursive_avg_token_frequencies(embedder, kwargs, counts=[], prev_counts=[]):
    """
    Need to return doc_counts, as the condition might affect it - e.g. we might be interested in the tokens that
    appear exactly k times. Then the next step should consider the counts only of such tokens in the previous step.

    :param embedder:
    :param kwargs:
    :param counts:
    :param prev_step_counts:
    :return:
    """

    relative_f, doc_counts = embedder.recursive_avg_frequencies_k(counts_list=counts,
                                                                    prev_counts=prev_counts,
                                                                    k=kwargs["k_overlap"], relativise=True,
                                                                    condition=kwargs["condition"],
                                                                    bounds=[int(i) for i in list(kwargs["bounds"].split(","))])

    embedded_doc = relative_f.tolist()

    return embedded_doc, doc_counts

def bow_per_token_empirical_entropy(embedder, tokens_density):
    """
    ngram_token_density = [empirical_distr_tokens[i]/np.sum(empirical_distr_tokens[i])\
                               if np.sum(empirical_distr_tokens[i])>0 else np.array(empirical_distr_tokens[i]) \
                                    for i in range(len(empirical_distr_tokens))]

    # density - sum of token frequencies in each ngram with non zero counts should sum to 1
    assert all([abs(np.sum(i)-1) < 1e-07 for i in ngram_token_density if np.sum(i)>0])
    """

    embedded_doc = embedder.get_empirical_distribution_entropy(tokens_density)

    return embedded_doc

def bow_recursive_avg_cumulative_frequencies(empirical_distr_tokens):
    """
     :param empirical_distr_tokens: list of token frequencies for single basis
     :param kwargs:
     :return: for 1 subbase: return list of single scalar
              for N subbases: return list of len n_gram number, each element is a list of len N with
              elements the cumulative frequencies of each subbase
    """
    embedded_doc = np.sum(empirical_distr_tokens)

    return embedded_doc

def plot_histogram_as_density(data, wordlist, savename="/tmp/", t=0, tokens=[]):
    
    # for ngram recursive avg frequencies
    if isinstance(data, np.ndarray):
        ndens = data
        elements = np.argwhere(ndens > 0)
        elements = elements.reshape((elements.size,))
        if len(tokens) == 0:
            ntokens = [wordlist[i] for i in elements.tolist()]
        else:
            ntokens = tokens
        dframe = pd.Dataframe.from_dict({"Tokens": ntokens, "Recursive proportions<br>(normalised)": ndens[elements]})
        fig = px.bar(dframe, y="Recursive proportions<br>(normalised)", x="Tokens")
        fig.update_xaxes(showline=True, linewidth=2, linecolor='black')
        fig.update_yaxes(showline=True, linewidth=2, linecolor='black')
        fig.update_layout(title="Density of recursive proportions<br>at time step {}".format(t), title_x=0.5,
                          plot_bgcolor='rgb(255,255,255)',
                          yaxis=dict(
                              title="Recursive proportions<br>(normalised)",
                              titlefont_size=18,
                              tickfont_size=18,
                              showgrid=False
                          ),
                          xaxis=dict(
                              title="Tokens",
                              titlefont_size=18,
                              tickfont_size=18,
                              showgrid=False
                          ),
                          font=dict(
                              size=18
                          ),
                          showlegend=False)
        # pio.write_html(fig, savename + "timestep_{}.html".format(t), auto_open=False)
        try:
            pio.write_image(fig, savename + "timestep_{}.png".format(t), width=800, height=571, scale=1)
            # pio.write_image(fig, savename + "timestep_{}.eps".format(t), width=800, height=571, scale=1)
        except:
            plt.figure(frameon=False)
            axes = plt.gca()
            axes.set(ylim=(0, 1))
            axes.xaxis.set_major_locator(plt.MaxNLocator(20))
            plt.bar(dframe["Tokens"], dframe["Recursive proportions<br>(normalised)"])
            plt.xlabel("Tokens")
            plt.ylabel("Recursive proportions<br>(normalised)")
            plt.xticks(rotation='vertical')
            plt.tight_layout()
            plt.margins(x=0)
            plt.margins(y=0)
            plt.savefig(savename + "timestep_{}.png".format(t), transparent=False)
            plt.savefig(savename + "timestep_{}.eps".format(t), transparent=False)
            plt.close()
    else:
        # for counts as a sparse matrix!
        elements = np.argwhere(data > 0)[:, 1]
        ntokens = [wordlist[i] for i in elements.tolist()]
        dframe = pd.Dataframe.from_dict({"Tokens": ntokens, "Counts": data.data / np.sum(data.data)})
        fig = px.bar(dframe, y="Counts", x="Tokens")
        fig.update_layout(title="Cumulative counts density at time step {}".format(t), title_x=0.5)

        pio.write_html(fig, savename + "timestep_{}.html".format(t), auto_open=False)
        try:
            pio.write_image(fig, savename + "timestep_{}.png".format(t))
            pio.write_image(fig, savename + "timestep_{}.eps".format(t))
        except:
            plt.figure(frameon=False)
            axes = plt.gca()
            axes.set(ylim=(0, 1))
            axes.xaxis.set_major_locator(plt.MaxNLocator(20))
            plt.bar(dframe["Tokens"], dframe["Counts"])
            plt.xlabel("Tokens")
            plt.ylabel("Counts")
            plt.xticks(rotation='vertical')
            plt.tight_layout()
            plt.margins(x=0)
            plt.margins(y=0)
            plt.savefig(savename + "timestep_{}.png".format(t), transparent=False)
            plt.savefig(savename + "timestep_{}.eps".format(t), transparent=False)
            plt.close()
    return 0

def get_sentiment_embedding(sentiment_counts, embedder, index, kwargs, acc_prev_step, sentiment,
                            save):
    if isinstance(sentiment_counts, list) and len(sentiment_counts) == 0:
        if not isinstance(acc_prev_step, np.ndarray):
            acc_prev = acc_prev_step.toarray()
        else:
            acc_prev = acc_prev_step
        return acc_prev, -1, -1, -1
    elif sentiment_counts.getnnz() > 0:
        count_sent = sentiment_counts.copy()
        # Normalize over the support of the sentiment
        ts, token_freq_sent_prev_step = bow_recursive_avg_token_frequencies(embedder=embedder, kwargs=kwargs,
                                                                            counts=count_sent,
                                                                            prev_counts=acc_prev_step)

        ngram_token_density_sent = np.true_divide(np.array(ts), np.sum(ts)) if np.sum(ts) > 1e-10 else \
            np.zeros(np.array(ts).shape)
        assert (abs(np.sum(ngram_token_density_sent) - 1) < 1e-10 if np.sum(ngram_token_density_sent) > 1e-10 else
                abs(np.sum(ngram_token_density_sent) < 1e-10))
        entropy = bow_per_token_empirical_entropy(embedder, ngram_token_density_sent)
        ts_cumul = bow_recursive_avg_cumulative_frequencies(ts)
        if save:
            # For the neutral words, distribution is expected to be uniform, plot to get the neutral words
            plot_histogram_as_density(ngram_token_density_sent, kwargs["projection_basis"],
                                      savename=kwargs["run_output_folder"] + kwargs["run_output_name"]
                                               + kwargs["save_recursive_avg_distribution"][:-1]
                                               + "_{}/".format(sentiment),
                                      t=index)

        return ts, token_freq_sent_prev_step, entropy, ts_cumul

def word_ngrams(tokens, ngram_range, separator=' ', step=1):
    # Split tokens into n_grams that overlap by n-step number of tokens.
    # Function also returns remaining tokens even if they do not form a complete n_gram.
    
    min_n, max_n = ngram_range
    if max_n != 1:
        original_tokens = tokens
        if min_n == 1:
            tokens = list(original_tokens)
            min_n += 1
        else:
            tokens = []

        n_original_tokens = len(original_tokens)

        tokens_append = tokens.append
        # space_join = separator.join
        # returns ngrams as list of tokens
        num = 0
        i = 0
        for num in range(min_n, min(max_n + 1, n_original_tokens + 1)):
            for i in range(0, n_original_tokens - num + 1, step):
                tokens_append(original_tokens[i: i + num])
        if len(original_tokens[i + num:]) > 0:
            tokens_append(original_tokens[i + num:])

    return tokens