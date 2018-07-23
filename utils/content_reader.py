import numpy as np
import pickle
from os.path import join

class ContentReader:
    """
    Reads content of a given doc/query content
    """
    def __init__(self, external_doc_ids, q_max_len, d_max_len, train_queries, index, padding=False):
        self.index = index
        self.token2id, _, _ = self.index.get_dictionary()
        self.external_doc_ids = external_doc_ids
        self.q_max_len = q_max_len
        self.d_max_len = d_max_len
        self.train_queries = train_queries
        self.padding = padding

    def get_query(self, query):
        """
        Reads the q_id query content
        :param query: id
        :return: list
        """
        q_words = [self.token2id[qi] if qi in self.token2id else 0 for qi in self.train_queries[query].split()]
        if self.padding:
            if len(q_words) < self.q_max_len:
                q_words = list(np.pad(q_words, (self.q_max_len - len(q_words), 0), "constant", constant_values=0))
            elif len(q_words) > self.q_max_len:
                q_words = q_words[:self.q_max_len]
        return q_words

    def get_document(self, d_id):
        """
        Reads the q_id query content
        :param d_id: str
        :return: list
        """
        doc = self.external_doc_ids[d_id]
        doc_words = list(self.index.document(doc)[1])
        if self.padding:
            if len(doc_words) < self.d_max_len:
                doc_words = list(np.pad(doc_words, (self.d_max_len - len(doc_words), 0), "constant", constant_values=0))
            elif len(doc_words) > self.d_max_len:
                doc_words = doc_words[:self.d_max_len]
        return doc_words


class ContentPickleReader:
    def __init__(self, input_files):
        self.input_files = input_files

    def pickle_data(self):
        """
        Reads the list of data inputs and corresponding dictionary
        :param query: self
        :return: tuple
        """
        data_list = pickle.load(open(join(self.input_files, "data_list.pickle"), 'rb'))  # type: list
        index_dict = pickle.load(open(join(self.input_files, "index_dict.pickle"), 'rb'))  # type: dict
        return data_list, index_dict
