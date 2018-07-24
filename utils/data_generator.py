import numpy as np
import keras
import ast
from os import listdir
from os.path import join

class DataGenerator(keras.utils.Sequence):
    """
    Generates data for Keras
    :return: data generator object
    """
    def __init__(self, list_IDs, labels, input, batch_size=32, shuffle=True):
        # Initialization
        self.batch_size = batch_size
        self.labels = labels
        self.list_IDs = list_IDs
        self.shuffle = shuffle
        self.input = input
        self.on_epoch_end()

    def __len__(self):
        """
        Denotes the number of batches per epoch
        :return: int
        """
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        """
        Generate one batch of data
        :param index: index of the current training item
        :return: tuple
        """
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def on_epoch_end(self):
        """
        Updates indexes after each epoch
        :return:
        """
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        """
        Generates data containing batch_size samples
        """

        # Initialization
        Xq = []
        Xd = []
        y = []  # np.empty(self.batch_size, dtype=int)

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            q, d = np.load(join(self.input, str(ID) + '.npy'))
            Xq.append(q)
            Xd.append(d)
            y.append(self.labels[ID])  # y[i,] = self.labels[ID]

        if len(Xq) != len(Xd):
            print(list_IDs_temp)

        X = [np.array(pad_sequences(Xq)), np.array(pad_sequences(Xd))]
        return X, np.array(y)  # y


def pad_sequences(listoflists):
    temp_list = []
    len_max = int(sum([len(l) for l in listoflists])/len(listoflists))
    for l in listoflists:
        # print(l)
        if len(l) < len_max:
            temp_list.append(list(np.pad(l, (len_max - len(l), 0), "constant", constant_values=0)))
            # print(temp_list)
        elif len(l) > len_max:
            temp_list.append(l[:len_max])
        else:
            temp_list.append(l)
    return temp_list


def line2batch_generator(input_file):
    """
    Generates batches from input buffer
    :param input_file: buffer
    :return: batch
    """
    while True:
        with open(input_file) as f:
            for line in f:
                line_list = ast.literal_eval(line.strip())
                yield [np.array(l) for l in line_list[0]], np.array(line_list[1])

def npy2batch_generator(input_folder):
    """
    Generates batches from npy files
    :param input_folder: folder of npy files
    :return: batch
    """
    while True:
        for f in listdir(input_folder):
            yield tuple(np.load(join(input_folder, f)))


def npyLoader(npy_files, labels, batch_size):
    """
    Generator of batches from npy files
    :param npy_files: list
    :param labels: list
    :param batch_size: int
    :return:
    """

    files = listdir(npy_files)
    L = len(files)

    # this line is just to make the generator infinite, keras needs that
    while True:

        batch_start = 0
        batch_end = batch_size

        while batch_start < L:
            limit = min(batch_end, L)
            Xq, Xd = [], []
            Y = []
            for f in files[batch_start:limit]:
                q, d = np.load(join(npy_files, f))
                Xq.append(q)
                Xd.append(d)
                Y.append(labels[int(f.split('.')[0])])

            yield ([np.array(pad_sequences(Xq)), np.array(pad_sequences(Xd))], np.array(Y))  # a tuple with two numpy arrays with batch_size samples

            batch_start += batch_size
            batch_end += batch_size
