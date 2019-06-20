import logging
import platform

import gc
import keras
import matplotlib.pyplot as plt
import numpy as np
from keras import Model
from keras.engine.saving import load_model
from keras.layers import (Embedding, Dropout, Convolution1D, MaxPool1D, concatenate,
                          Bidirectional, LSTM)
from keras.layers import Input, Flatten, Dense
from keras.models import Sequential
from keras.utils.vis_utils import plot_model

from data_mining.config import Config
from data_mining.data_helper import DataHelper


def get_text_cnn(vocab_size, max_sequence_len, embedding_dim, num_classes, embedding_matrix=None):
    weights = None
    # train_able=True
    if embedding_matrix:
        weights = np.asarray([embedding_matrix])
    # 模型结构：词嵌入-卷积池化*3-拼接-全连接-dropout-全连接
    main_input = Input(shape=(max_sequence_len,))
    # 词嵌入（使用预训练的词向量）
    embedder = Embedding(vocab_size, embedding_dim, input_length=max_sequence_len,
                         weights=weights, trainable=False)
    embed = embedder(main_input)
    # 词窗大小分别为3,4,5
    cnn1 = Convolution1D(256, 3, padding='same', strides=1, activation='relu')(embed)
    cnn1 = MaxPool1D(pool_size=4)(cnn1)
    cnn2 = Convolution1D(256, 4, padding='same', strides=1, activation='relu')(embed)
    cnn2 = MaxPool1D(pool_size=4)(cnn2)
    cnn3 = Convolution1D(256, 5, padding='same', strides=1, activation='relu')(embed)
    cnn3 = MaxPool1D(pool_size=4)(cnn3)
    # 合并三个模型的输出向量
    cnn = concatenate([cnn1, cnn2, cnn3], axis=-1)
    flat = Flatten()(cnn)
    drop = Dropout(0.2)(flat)
    output = Dense(num_classes, activation='sigmoid')(drop)
    model = Model(inputs=main_input, outputs=output)
    return model


def get_bi_lstm(vocab_size, max_sequence_len, embedding_dim, num_classes, embedding_matrix=None):
    weights = None
    # train_able=True
    if embedding_matrix:
        weights = np.asarray([embedding_matrix])
    model = Sequential()
    model.add(Embedding(vocab_size, embedding_dim, input_length=max_sequence_len,
                        weights=weights, trainable=False))
    model.add(Bidirectional(LSTM(256, dropout=0.2, recurrent_dropout=0.1, return_sequences=True)))
    model.add(Bidirectional(LSTM(256, dropout=0.2, recurrent_dropout=0.1)))
    model.add(Dense(num_classes, activation='sigmoid'))
    return model


def get_cnn_pair_rnn(vocab_size, max_sequence_len, embedding_dim, num_classes, embedding_matrix=None):
    """并联 cnn rnn
    # 模型结构：词嵌入-卷积池化-全连接 ---拼接-全连接
    #                -双向GRU-全连接
    :param vocab_size:
    :param max_sequence_len:
    :param embedding_dim:
    :param num_classes:
    :param embedding_matrix:
    :return:
    """
    weights = None
    # train_able=True
    if embedding_matrix is not None:
        weights = np.asarray([embedding_matrix])
    model = Sequential()
    model.add(Embedding(vocab_size, embedding_dim, input_length=max_sequence_len,
                        weights=weights, trainable=True))
    sentence_input = Input(shape=(max_sequence_len,), dtype='float64')
    embed = Embedding(vocab_size, embedding_dim, input_length=max_sequence_len)(sentence_input)
    cnn = Convolution1D(256, 3, padding='same', strides=1, activation='relu')(embed)
    cnn = MaxPool1D(pool_size=4)(cnn)
    cnn = Flatten()(cnn)
    cnn = Dense(256)(cnn)
    rnn = Bidirectional(LSTM(256, dropout=0.2, recurrent_dropout=0.1))(embed)
    rnn = Dense(256)(rnn)
    con = concatenate([cnn, rnn], axis=-1)
    output = Dense(num_classes, activation='sigmoid')(con)
    model = Model(inputs=sentence_input, outputs=output)
    return model


def plot_history(history):
    plt.subplot(211)
    plt.title("accuracy")
    plt.plot(history.history["acc"], color="r", label="train")
    plt.plot(history.history["val_acc"], color="b", label="val")
    plt.legend(loc="best")

    plt.subplot(212)
    plt.title("loss")
    plt.plot(history.history["loss"], color="r", label="train")
    plt.plot(history.history["val_loss"], color="b", label="val")
    plt.legend(loc="best")

    plt.tight_layout()
    plt.show()


def train():
    data_helper = DataHelper()
    model = get_cnn_pair_rnn(vocab_size=data_helper.vocab_size,
                             max_sequence_len=data_helper.sequence_length,
                             embedding_dim=data_helper.embedding_dim,
                             embedding_matrix=data_helper.pre_embeddings,
                             num_classes=data_helper.num_tags)
    model.compile(
        loss="binary_crossentropy",  # 'binary_crossentropy',categorical_crossentropy
        optimizer='adam',
        metrics=['accuracy'])
    model.summary()
    if platform.system() == "Linux":
        pass
    elif platform.system() == "Windows":
        plot_model(model, to_file=Config.model_image, show_shapes=True)

    x_train, y_train = data_helper.get_data(y_vectorize=True)
    del data_helper
    gc.collect()
    history = model.fit(x=x_train,
                        y=y_train,
                        batch_size=1024,
                        epochs=1,
                        verbose=1,
                        validation_split=0.1,
                        )

    model.save(Config.sentiment_model_path)
    plot_history(history)
    logging.info("save model : {}".format(Config.sentiment_model_path))


class Prediction():
    def __init__(self):
        self.model = self.load_model()
        self.data_helper = DataHelper()

    def load_model(self):
        keras.backend.clear_session()  # https://blog.csdn.net/lhs960124/article/details/79028691
        model = load_model(Config.sentiment_model_path)
        # TypeError: __init__() missing 1 required positional argument: 'attention_size'
        model._make_predict_function()
        return model

    def predict(self, poem_str):
        """
        :param sentence_li:  list of sentence
                    [
                    ["我"，"爱"，"北"，"京"]，
                    ["我"，"爱"，"北"，"京"]，
                    ]
        :return: list of tags
            [
                ["tag_name1","tag_name2"],
                ["tag_name1"],
            ]
        """
        sentence_ids = self.data_helper.get_x_data(poem_str)
        pred_tag_ids_arr = self.model.predict(np.asarray(sentence_ids))
        pred_tags_li = []
        for pred_sentence_tag_vec in pred_tag_ids_arr:
            tag_ids = np.where(pred_sentence_tag_vec >= 0.5)[0].tolist()
            tag_names = [self.data_helper.id2tag[tid] for tid in tag_ids]
            pred_tags_li.append(tag_names)
        return pred_tags_li
