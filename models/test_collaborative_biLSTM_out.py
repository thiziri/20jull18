import sys
import json
import pyndri
import numpy as np
from os.path import join
from keras.models import model_from_json

sys.path.append("../utils")

from model_utils import get_input_label_size, get_queries, run_test_data
from tqdm import tqdm

def select_rel_by_qids(qid_list, relations_list):
    """
    select a set of relations based on q_id
    :param qid_list: list
    :param relations_list: list
    :return:
    """
    # select relations
    return set([re for re in relations_list if re[0] in qid_list])


if __name__ == '__main__':
    config_file = sys.argv[1]
    configure = json.load(open(config_file))
    config = configure["main_configuration"]
    config_data = config["data_sets"]
    config_model = config["model"]
    config_model_param = config_model["parameters"]
    config_model_train = config_model["train"]
    config_model_test = config_model["test"]
    print("Data extraction\nConfiguration: ")
    print(json.dumps(config, indent=2), end='\n')

    print("Loading model a model...")

    # load json and create model
    json_file = open(join(config_model_train["train_details"], config_model_param["model_name"]+'.json'), 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    model_weights = config_model_train["weights"]+'_iter_{x:04d}.h5'.format(x=config_model_test["test_period"])
    loaded_model.load_weights(model_weights)
    print("Loaded model from disk: ", model_weights)

    # evaluate loaded model on test data
    loaded_model.compile(optimizer=config_model_param["optimizer"], loss=config_model_train["loss_function"],
                  metrics=config_model_train["metrics"])
    print("Compiled.")

    print("Reading data index ...")
    index = pyndri.Index(config_data["index"])
    token2id, _, _ = index.get_dictionary()
    externalDocId = {}
    for doc_id in range(index.document_base(), index.maximum_document()):  # type: int
        extD_id, _ = index.document(doc_id)
        externalDocId[extD_id] = doc_id
    queries_temp = get_queries(config_data["queries"])
    test_queries = {q_id: queries_temp[q_id] for q_id in queries_temp if q_id in
                    [l.strip() for l in open(config_model_test["test_queries"]).readlines()]}

    # label_len = get_input_label_size(config_data)
    if config_model_test["if_reranking"]:
        out = open(join(config_model_test["save_rank"], config_model_param["model_name"]+"predict.txt"), 'w')
        relations = run_test_data(config_model_test["rank"], config_model_test["top_rank"])  # [(q, doc)]
        relations_to_process = select_rel_by_qids(test_queries, relations)
        print("Please, wait while predicting ...")
        for relation in tqdm(relations_to_process):
            query = [token2id[qi] if qi in token2id else 0 for qi in test_queries[relation[0]].strip().split()]  # get query terms

            doc = list(index.document(externalDocId[relation[1]])[1])   # get document terms
            x_test = [np.array([query]), np.array([doc])]
            rank = loaded_model.predict(x_test, verbose=0)
            score = np.average(np.array(rank[0]))
            out.write("{q}\tQ0\t{d}\t1\t{s}\t{m}\n".format(q=relation[0], d=relation[1], s=score,
                                                           m=config_model_param["model_name"]))
            # break
        print("Prediction done.")
