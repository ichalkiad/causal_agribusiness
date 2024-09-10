import numpy as np

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



