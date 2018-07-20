import sys
import random
import pyndri
from tqdm import tqdm
import numpy as np
import json
import os
from os.path import join
from os import  listdir

sys.path.append('../utils')

from model_utils import read_lablers_to_relations, get_queries
from content_reader import ContentReader

def chunks(l, n):
    """
    Yield successive n-sized chunks from shuffled l.
    """
    random.shuffle(l)
    for i in range(0, len(l), n):
        yield l[i:i + n]

def select_rel_by_qids(qid_list, relations_list):
    """
    select a set of relations based on q_id
    :param qid_list: list
    :param relations_list: list
    :return:
    """
    # select relations
    return set([re for re in relations_list if re[0] in qid_list])


if __name__ == "__main__":
    config_file = sys.argv[1]
    config = json.load(open(config_file))
    batch_size = config["batch_size"]
    print("[First]:\nRead label files to relations...")
    relations, relation_labeler = read_lablers_to_relations(config["relations"])  # relation .label file: qrels
    rel_label = {relation: [int(relation_labeler[labeler][relation]) for labeler in relation_labeler if relation in
                            relation_labeler[labeler]] for relation in relations}  # combine different judgements
    labels = {relation: np.average([r/max(rel_label[relation]) if max(rel_label[relation]) > 0 else r
                                    for r in rel_label[relation]]) for relation in relations}
    uniq_queries = set()
    uniq_documents = set()
    for rel in relations:
        uniq_queries.add(rel[0])
        uniq_documents.add(rel[1])

    # extracted queries in .txt files
    queries = get_queries(config["queries"])

    queries_length = {q: len(queries[q].split()) for q in queries}

    out = config["output"]  # output folder
    index = pyndri.Index(config["index"])  # documents index

    print("Reading data index ...")
    externalDocId = {}
    documents_length = {}
    for doc_id in range(index.document_base(), index.maximum_document()):  # type: int
        extD_id, content = index.document(doc_id)
        if extD_id in uniq_documents:
            externalDocId[extD_id] = doc_id
            documents_length[extD_id] = len(content)

    for fold in listdir(config["split_data"]):
        train = [l.strip() for l in open(join(join(config["split_data"], fold), "train_.txt")).readlines()]
        valid = [l.strip() for l in open(join(join(config["split_data"], fold), "valid_.txt")).readlines()]
        train_relations = select_rel_by_qids(train, relations)
        valid_relations = select_rel_by_qids(valid, relations)
        to_save = {"train": train_relations, "valid": valid_relations}
        os.mkdir(join(out, fold))
        for set_ in to_save:
            baches = chunks(list(to_save[set_]), batch_size)
            reader = ContentReader(external_doc_ids=externalDocId,
                                   q_max_len=0,
                                   d_max_len=0,
                                   train_queries=queries,
                                   index=index,
                                   padding=True)

            print("Saving batches of {} unique relations...".format(len(to_save[set_])))
            out_fold = join(join(out, fold), set_)
            os.mkdir(out_fold)
            # with open(out, 'w') as output:
            if 1:
                for id_b, batch in tqdm(enumerate(baches)):
                    reader.q_max_len = max([queries_length[relation[0]] for relation in batch])
                    reader.d_max_len = max([documents_length[relation[1]] for relation in batch])
                    list_line = []
                    list_queries = []
                    list_documents = []
                    relevance = []

                    for batch_element in batch:
                        list_queries.append(reader.get_query(batch_element[0]))
                        list_documents.append(reader.get_document(batch_element[1]))
                        relevance.append(labels[batch_element])

                    # list_line.append([list_queries, list_documents])
                    list_line.append([np.array(list_queries), np.array(list_documents)])
                    # list_line.append(relevance)
                    list_line.append(np.array(relevance))

                    # output.write(str(list_line)+"\n")
                    np.save(join(out_fold, str(id_b)+".npy"), np.array(list_line))

    print("Done")
