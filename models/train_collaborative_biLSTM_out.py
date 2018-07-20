import sys
import json
import tensorflow as tf
from keras.layers.merge import concatenate
from keras.models import Model
from keras.utils import plot_model

sys.path.append("../utils")

from model_utils import convert_embed_2_numpy, read_embedding
from model_utils import get_input_label_size, plot_history, get_optimizer
from keras.layers import Dense, Input, LSTM, Dropout, Bidirectional, Embedding, Dot
from keras.layers.normalization import BatchNormalization
from os.path import join
from keras import callbacks
from data_generator import DataGenerator, line2batch_generator, npy2batch_generator
from keras.utils.training_utils import multi_gpu_model


if __name__ == '__main__':
    config_file = "/home/thiziri/Desktop/collaborative_labels_out.json"  # sys.argv[1]
    configure = json.load(open(config_file))
    config = configure["main_configuration"]
    config_data = config["data_sets"]
    config_model = config["model"]
    config_model_param = config_model["parameters"]
    config_model_train = config_model["train"]
    config_model_test = config_model["test"]
    print("Data extraction\nConfiguration: ")
    print(json.dumps(config, indent=2), end='\n')

    G = int(sys.argv[2])

    print("Read embeddings ...")
    embed_tensor = convert_embed_2_numpy(read_embedding(config_data["embed"]), config_data["vocab_size"])

    print("Create a model...")

    query = Input(name="in_query", batch_shape=[None, None], dtype='int32')
    doc = Input(name="in_doc", batch_shape=[None, None], dtype='int32')

    embedding = Embedding(config_data['vocab_size'], config_data['embed_size'], weights=[embed_tensor],
                          trainable=config_model_train['train_embed'], name="embeddings", mask_zero=True)
    del embed_tensor
    q_embed = embedding(query)
    d_embed = embedding(doc)
    # lstm_layer = Bidirectional(LSTM(config_model_param["number_lstm_units"], dropout=config_model_param["lstm_dropout"],
    #                                recurrent_dropout=config_model_param["lstm_dropout"]))
    q_lstm_layer = Bidirectional(LSTM(config_model_param["number_q_lstm_units"],
                                      dropout=config_model_param["q_lstm_dropout"],
                                      recurrent_dropout=config_model_param["q_lstm_dropout"]))
    d_lstm_layer = Bidirectional(LSTM(config_model_param["number_d_lstm_units"],
                                      dropout=config_model_param["d_lstm_dropout"],
                                      recurrent_dropout=config_model_param["d_lstm_dropout"]))
    q_vector = q_lstm_layer(q_embed)
    d_vector = d_lstm_layer(d_embed)
    input_vector = concatenate([q_vector, d_vector])
    # merge_layer = Dot(axes=1,normalize=False)([q_vector, d_vector]) ######## similarity dotprod between the 2 vectors
    # c = concatenate([q_vector, merge_layer, d_vector], axis=1)
    merged = BatchNormalization()(input_vector)
    merged = Dropout(config_model_param["dropout_rate"])(merged)
    dense = Dense(config_model_param["layers_size"][0], activation=config_model_param['hidden_activation'],
                  name="MLP_combine_0")(merged)
    i = 0
    for i in range(config_model_param["num_layers"]-2):
        dense = BatchNormalization()(dense)
        dense = Dropout(config_model_param["dropout_rate"])(dense)
        dense = Dense(config_model_param["layers_size"][i+1], activation=config_model_param['hidden_activation'],
                      name="MLP_combine_"+str(i+1))(dense)
    dense = BatchNormalization()(dense)
    dense = Dropout(config_model_param["dropout_rate"])(dense)
    if config_model_param["predict_labels"]:
        out_size = get_input_label_size(config_data)
    else:
        out_size = 1
    out_labels = Dense(out_size, activation=config_model_param['output_activation'], name="MLP_out")(dense)
    model = Model(inputs=[query, doc], outputs=out_labels)
    optimizer = get_optimizer(config_model_param["optimizer"])(lr=config_model_param["learning_rate"])

    # https://www.pyimagesearch.com/2017/10/30/how-to-multi-gpu-training-with-keras-python-and-deep-learning/
    if G <= 1:
        print("[INFO] training with 0/1 GPU...")
        model_gpu = model
    else:
        print("[INFO] training with {} GPUs...".format(G))

        # we'll store a copy of the model on *every* GPU and then combine
        # the results from the gradient updates on the CPU
        with tf.device("/cpu:0"):
            # initialize the model
            model_gpu = model

        # make the model parallel
        model_gpu = multi_gpu_model(model_gpu, gpus=G)

    model_gpu.compile(optimizer=optimizer, loss=config_model_train["loss_function"],
              metrics=config_model_train["metrics"])  # model

    print(model.summary())
    plot_model(model, to_file=join(config_model_train["train_details"], config_model_param['model_name']+".png"))
    model_json = model.to_json()
    with open(join(config_model_train["train_details"], config_model_param["model_name"] + ".json"), "w") as json_file:
        json_file.write(model_json)
    print("Saved model to disc.")

    print("Reading training data:")
    training_generator = npy2batch_generator(config_data['input_train'])  # line2batch_generator(config_data['input_train'])
    validation_generator = npy2batch_generator(config_data["input_valid"]) if config_data["input_valid"] else None

    steps_per_epoch = int(config_data["num_samples"]/config_model_train["batch_size"])+1  # |relations|/batch_size

    print("Model training...")
    mc = callbacks.ModelCheckpoint(config_model_train["weights"]+'_iter_{epoch:04d}.h5', save_weights_only=True,
                                   period=config_model_train["save_period"])

    if validation_generator is not None:
        history = model_gpu.fit_generator(generator=training_generator,
                                          validation_data=validation_generator,
                                          epochs=config_model_train["epochs"],
                                          verbose=config_model_train["verbose"],
                                          steps_per_epoch=steps_per_epoch,
                                          validation_steps=steps_per_epoch,
                                          callbacks=[mc])  # model
    else:
        history = model_gpu.fit_generator(generator=training_generator,
                                          epochs=config_model_train["epochs"],
                                          verbose=config_model_train["verbose"],
                                          steps_per_epoch=steps_per_epoch,
                                          callbacks=[mc])  # model

    # save trained model
    # print("Saving model and its weights ...")
    # model_gpu.save_weights(config_model_train["weights"]+".h5")  # model

    print("Plotting history ...")
    plot_history(history, config_model_train["train_details"], config_model_param["model_name"],
                 validation_generator is not None)
    print("Done.")

