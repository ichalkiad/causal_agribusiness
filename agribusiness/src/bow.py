from sklearn.feature_extraction.text import _make_int_array
import numpy as np
from sklearn.utils import _IS_32BIT
import scipy.sparse as sp
from collections import Counter

class BoWEmbedder:

        def __init__(self,n , base, cfg_weights={}, dep_graph_weights={}):
            """
            :param base: list of lists that make up the base with respect to which we will get the counts
            """
            self.ngram_size = n
            self.base = base
            #Number of sub-bases
            self.p = len(base)
            #Dimension of each sub-base
            self.q = [len(sub_basis) for sub_basis in base]
            #Weights for CFG or dependency labels
            self.cfg_weights = cfg_weights
            self.dep_graph_weights = dep_graph_weights

        """
        
                Time series constructors based on counts
                
        """
        def _count_vocab(self, ngram, base, q):
                """

                :param ngram: single ngram extracted from a processed document
                :return Xs: list of sparse matrices each containing the counts wrt to base for each ngram

                Example:

                For n=1 and base = [vocab], where vocab is the complete vocabulary, one gets the
                standard BoW,non-time dependent embedding:

                embdd = embedder._count_vocab(ngrams,base)   # ngrams: list of tokens in document

                For n=1 with any base, one has the case of a window fixed on the left at the start of the sentence
                and adding one token to the right at every timestep.

                For n>1 and any base, one has the case of segmenting the document in non-overlapping pieces of n tokens, and then computing
                the counts of each wrt to base.

                Counts are per segment and non aggregation or normalisation is performed in this method.

                """
                j_indices = []
                indptr = []

                # Number of times each word appears in the ngram
                ngram_hist = Counter(ngram)

                values = _make_int_array()
                indptr.append(0)
                # Turn to set for efficiency
                set_base = set(base)
                feature_basis_counter = {}
                for basis_element in ngram:
                    if basis_element in set_base:
                        element_idx = base.index(basis_element)
                        if element_idx not in feature_basis_counter.keys():
                            feature_basis_counter[element_idx] = ngram_hist[basis_element]
                        else:
                            # all occurrences of the word in the ngram have been included the first time we
                            # encountered the word
                            continue
                    else:
                        # OOV
                        continue

                j_indices.extend(feature_basis_counter.keys())
                values.extend(feature_basis_counter.values())
                indptr.append(len(j_indices))

                if indptr[-1] > 2147483648:  # = 2**31 - 1
                    if _IS_32BIT:
                        raise ValueError(('sparse CSR array has {} non-zero '
                                          'elements and requires 64 bit indexing, '
                                          'which is unsupported with 32 bit Python.')
                                         .format(indptr[-1]))
                    indices_dtype = np.int64
                else:
                    indices_dtype = np.int32

                j_indices = np.asarray(j_indices, dtype=indices_dtype)
                indptr = np.asarray(indptr, dtype=indices_dtype)
                values = np.frombuffer(values, dtype=np.intc)

                X = sp.csr_matrix((values, j_indices, indptr),
                                  shape=(len(indptr) - 1, q),
                                  dtype=int)
                X.sort_indices()

                return X

        def get_counts(self,ngram,base,kwargs={}):

            #:param processed_doc_tokens: tokens of document after post-processing (NOT necessarily one token==one word - e.g. track and field might be one token according to vocabulary )
            #:param ngram: size n of unit of text analysis
            #:param context: global context based on document up to t or local window context
            #:param debug: run with intermediate printouts and messages
            #:param kwargs: context_window_size: size of local context window
            #:return: global context: list of sublists, with each sublist corresponding to a sub-base.
            #                        Each sublist contains sparse matrices with the counts of the n-grams w.r.t the
            #                        sub-base
            #        local context: list of sub-lists. Each sublist corresponds to a window. To treat each window as a document
            #                        as pointed out in the note, each window sub-list has the same format as the output
            #                        for the case of global context: list of sub-lists correspoding to sub-bases, each
            #                        containing sparse count matrices

            # Get count matrix
            doc_counts = self._count_vocab(ngram,base,len(base))
            

            return doc_counts




        # eq. 16,17 , 18 for renormalisation (N is i, m is n)  - aggregate_base: [[f_11,..f_1q]]
        @staticmethod
        def recursive_avg_frequencies_k(counts_list, prev_counts, k, relativise=True, condition="at least", bounds=[]):
            """

            :param counts_list: list of sparse matrices containing the counts w.r.t to a single basis
            :param k: number of co-occurrences we are looking for, for each basis word
            :param n: size of n-gram
            :param condition: exact or at least
            :return: renormalised relative frequencies from counts that equal exactly k - eq. 18 in embeddings note
            """

            recursive_avg = np.zeros(prev_counts.shape)
            doc_counts = counts_list.toarray().squeeze()
            #Apply condition of binary relation for selecting elements
            if condition=="hard equal":
                doc_counts[np.nonzero(doc_counts > k)] = 0
                doc_counts[np.nonzero(doc_counts < k)] = 0
            elif condition=="at least":
                doc_counts[np.nonzero(doc_counts < k)] = 0
            elif condition=="bounded" and len(bounds) > 0:
                # bounds included in counting
                doc_counts[np.nonzero(doc_counts < bounds[0])] = 0
                doc_counts[np.nonzero(doc_counts > bounds[1])] = 0
            elif condition=="unique":
                absolute_counts = np.count_nonzero(doc_counts, axis=-1)
                if absolute_counts < k:
                    return recursive_avg, doc_counts

            if relativise:
                numerator = np.sum(prev_counts)
                current_nonzeros_idx = np.nonzero(doc_counts)[0]
                if current_nonzeros_idx.shape[0] == 0:
                    recursive_avg = np.zeros(prev_counts.shape)
                else:
                    # Find intersection of current set of indices with non-zero values,
                    # and all previously seen indices with non-zero values
                    update_idx = np.array(list(set(np.nonzero(prev_counts)[0].tolist())
                                               .intersection(set(current_nonzeros_idx.tolist()))), dtype=int)
                    # Newly seen tokens only
                    new_idx = np.array(list(set(current_nonzeros_idx.tolist())
                                            .difference(set(update_idx.tolist()))), dtype=int)
                    # Total number of tokens seen so far
                    total_nonzeros = np.sum(doc_counts) + numerator
                    if update_idx.size > 0:
                        try:
                            recursive_avg[update_idx] = np.true_divide(prev_counts[update_idx]
                                                               + doc_counts[update_idx], total_nonzeros)
                        except:
                            recursive_avg[0][update_idx] = np.true_divide(prev_counts[update_idx]
                                                                       + doc_counts[update_idx], total_nonzeros)
                    try:
                        recursive_avg[new_idx] = np.true_divide(doc_counts[new_idx], total_nonzeros)
                    except:
                        recursive_avg[0][new_idx] = np.true_divide(doc_counts[new_idx], total_nonzeros)
            else:
                # in this case only sum counts
                recursive_avg = prev_counts + doc_counts

            return recursive_avg, doc_counts

 
        @staticmethod
        def get_empirical_distribution_entropy(empirical_distribution_density):

            if not isinstance(empirical_distribution_density, np.ndarray):
                empirical_distribution_density = np.array(empirical_distribution_density)

            try:
                nonz = [i for i in range(empirical_distribution_density.shape[0])
                    if abs(empirical_distribution_density[i]) > 1e-10]
            except:
                # after previous counts loading
                nonz = [i for i in range(empirical_distribution_density.shape[1])
                        if abs(empirical_distribution_density[0][i]) > 1e-10]
            if len(nonz) == 0:
                entropy = 0.0
            else:
                # use multiply to explicitly show elementwise multiplication
                try:
                    entropy_val = np.sum(np.multiply(empirical_distribution_density[nonz],
                                                 np.log(empirical_distribution_density[nonz])))
                except:
                    entropy_val = np.sum(np.multiply(empirical_distribution_density[0][nonz],
                                                     np.log(empirical_distribution_density[0][nonz])))
                if entropy_val == 0:
                    entropy = 0.0
                else:
                    entropy = -entropy_val

            return entropy
































