import numpy as np
import os
import tensorflow as tf
import matplotlib.pyplot as plt
from os.path import join
from tqdm import tqdm
from keras import optimizers
from keras.layers.merge import concatenate
from keras.models import Model
from keras.layers import Lambda


def make_parallel(model, gpu_count):
    """
    Make a given keras model runs in parallel
    :param model: keras.model
    :param gpu_count: int
    :return: keras.model
    ref: https://github.com/kuza55/keras-extras/issues/21
    """
    def get_slice(data, idx, parts):
        shape = tf.shape(data)
        size = tf.concat([shape[:1] // parts, shape[1:]], axis=0)
        stride = tf.concat([shape[:1] // parts, shape[1:] * 0], axis=0)
        start = stride * idx
        return tf.slice(data, start, size)

    outputs_all = []
    for i in range(len(model.outputs)):
        outputs_all.append([])

    # Place a copy of the model on each GPU, each getting a slice of the batch
    for i in range(gpu_count):
        with tf.device('/gpu:%d' % i):
            with tf.name_scope('tower_%d' % i) as scope:

                inputs = []
                # Slice each input into a piece for processing on this GPU
                for x in model.inputs:
                    input_shape = tuple(x.get_shape().as_list())[1:]
                    slice_n = Lambda(get_slice, output_shape=input_shape, arguments={'idx': i, 'parts': gpu_count})(x)
                    inputs.append(slice_n)

                outputs = model(inputs)

                if not isinstance(outputs, list):
                    outputs = [outputs]

                # Save all the outputs for merging back together later
                for l in range(len(outputs)):
                    outputs_all[l].append(outputs[l])

    # merge outputs on CPU
    with tf.device('/cpu:0'):
        merged = []
        for outputs in outputs_all:
            merged.append(concatenate(outputs, axis=0))

        return Model(input=model.inputs, output=merged)

def get_optimizer(optimizer):
    all_classes = {
        'sgd': optimizers.SGD,
        'rmsprop': optimizers.RMSprop,
        'adagrad': optimizers.Adagrad,
        'adadelta': optimizers.Adadelta,
        'adam': optimizers.Adam,
        'adamax': optimizers.Adamax,
        'nadam': optimizers.Nadam,
        'tfoptimizer': optimizers.TFOptimizer
    }
    return all_classes[optimizer]


""" 
read embed_norm to dictionary
return: dict
"""
def read_embedding(filename):
    embed = {}
    with open(filename) as embed_file:
        for line in tqdm(embed_file):
            line = line.strip().split()
            embed[int(line[0])] = list(map(float, line[1:]))
    print('[%s]\n\tEmbedding size: %d' % (filename, len(embed)), end='\n')
    return embed


"""
Convert Embedding Dict 2 numpy array
return: numpy matrix
"""
def convert_embed_2_numpy(embed_dict, max_size=0, embed=None):
    feat_size = len(embed_dict[list(embed_dict.keys())[0]])
    if embed is None:
        embed = np.zeros((max_size, feat_size), dtype=np.float32)

    if len(embed_dict) > len(embed):
        raise Exception("vocab_size %d is larger than embed_size %d, change the vocab_size in the config!"
                        % (len(embed_dict), len(embed)))

    for k in tqdm(embed_dict):
        embed[k] = np.array(embed_dict[k])
    print('Generate numpy embed:', str(embed.shape), end='\n')
    return embed


"""
read label files to relations
return: (set, dict)
"""
def read_lablers_to_relations(labelers_dir):
    relation_labeler = {}
    relations = set()
    for labeler_file in os.listdir(labelers_dir):
        relation_labeler[labeler_file] = {}
        with open(join(labelers_dir, labeler_file), 'r') as labeler:
            for l in tqdm(labeler.readlines()):
                relation = (l.strip().split()[0], l.strip().split()[1])  # (q, d)
                relations.add(relation)
                relation_labeler[labeler_file][relation] = l.strip().split()[2]  # rel
    return relations, relation_labeler


"""
read query files where each line is in the format: "q_id    q_text"
return: dict
"""
def get_queries(query_file):
    with open(query_file, "r") as f:
        return {l.strip().split("\t")[0]: l.strip().split("\t")[1] for l in f}


"""
Compute size of input labels vector, according to the input data configuration
return: int
"""
def get_input_label_size(config_data):
    if config_data["if_masking"]:
        if config_data["mask"] == "bin":
            return config_data["labelers_num"]*len(config_data["labels_values"])
        if config_data["mask"] == "scalable":
            return len(config_data["labels_values"])
    return config_data["labelers_num"]


"""
Compute the mask vector of input labels vector according to the data configuration
return: list(int)"""
def get_mask(rel_labels, config_data):
    if not config_data["if_masking"]:
        return rel_labels
    if config_data["mask"] == "bin":
        mask_labels = []
        labels = config_data["labels_values"]
        for l in rel_labels:
            l_mask = []
            for l_i in l:
                l_i_mask = []
                for v in labels:
                    l_i_mask.append(1 if l_i == v else 0)
                l_mask = l_mask + l_i_mask
            mask_labels.append(l_mask)
        return mask_labels
    if config_data["mask"] == "scalable":
        mask_labels = []
        labels = config_data["labels_values"]
        for l in rel_labels:
            l_mask = []
            for v in labels:
                l_mask.append(l.count(v))
            mask_labels.append(l_mask)
        return mask_labels


"""
Plotting the training history of a model
"""
def plot_history(history, path, model_name, if_validate):
    loss_list = [s for s in history.history.keys() if 'loss' in s and 'val' not in s]
    if if_validate:
        val_loss_list = [s for s in history.history.keys() if 'loss' in s and 'val' in s]
        val_acc_list = [s for s in history.history.keys() if 'acc' in s and 'val' in s]
    acc_list = [s for s in history.history.keys() if 'acc' in s and 'val' not in s]

    if len(loss_list) == 0:
        print('Loss is missing in history')
        return

        # As loss always exists

    epochs = range(1, len(history.history[loss_list[0]]) + 1)

    # Loss
    plt.figure(1)
    for l in loss_list:
        plt.plot(epochs, history.history[l], 'b',
                 label='Training loss (' + str(str(format(history.history[l][-1], '.5f')) + ')'))
    if if_validate:
        for l in val_loss_list:
            plt.plot(epochs, history.history[l], 'g',
                     label='Validation loss (' + str(str(format(history.history[l][-1], '.5f')) + ')'))

    plt.title('Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(join(path,  model_name+"_train_loss.png"))

    # Accuracy
    plt.figure(2)
    for l in acc_list:
        plt.plot(epochs, history.history[l], 'b',
                 label='Training accuracy (' + str(format(history.history[l][-1], '.5f')) + ')')
    if if_validate:
        for l in val_acc_list:
            plt.plot(epochs, history.history[l], 'g',
                     label='Validation accuracy (' + str(format(history.history[l][-1], '.5f')) + ')')

    plt.title('Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    # plt.show()
    plt.savefig(join(path, model_name+"_train_acc.png"))


"""
Create relations from a run file. For re-ranking purpose.
Return: list of relations [(q, doc)]
"""
def run_test_data(run_file, k=1000):
    relations = []
    with open(run_file, "r") as rank:
        i = 0
        queries_rank = []
        for line in tqdm(rank):
            if line is not None:
                q = str(int(line.strip().split()[0]))
                if q in queries_rank:
                    i += 1
                else:
                    queries_rank.append(q)
                    i = 1
                doc = line.strip().split()[2]
                if i in range(k + 1):
                    relations.append((q, doc))
                else:
                    continue
    return relations

