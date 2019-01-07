# -*- coding: utf-8 -*-
# Copyright 2017 Kakao, Recommendation Team
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import tensorflow as tf
import numpy as np
import h5py
import datetime

import keras
from keras.models import Model
from keras.layers.merge import dot
from keras.layers import Dense, Input
from keras.layers.core import Reshape
from keras.preprocessing.text import Tokenizer
from keras.layers import GlobalMaxPool1D
from keras.layers import GRU
from keras.layers import TimeDistributed

from keras.layers.embeddings import Embedding
from keras.layers.core import Dropout, Activation

from keras.layers import LSTM
from keras.layers import Bidirectional
from keras.layers import MaxPool2D, Conv2D
from keras.layers import Concatenate
from keras.layers import K
from keras.layers import Flatten
from keras.layers import BatchNormalization
from keras import backend
from keras.engine.topology import Layer
from keras import initializers, regularizers, constraints

from misc import get_logger, Option
from keras.optimizers import RMSprop
from keras.layers import Lambda
from keras.backend import permute_dimensions
from keras.callbacks import ReduceLROnPlateau
opt = Option('./config.json')

def top1_acc(x, y):
    return keras.metrics.top_k_categorical_accuracy(x, y, k=1)

def transpose(x):
    from keras import backend
    return backend.permute_dimensions(x, [0, 1, 3, 2])

class Attention(Layer):
    def __init__(self, step_dim,
                 W_regularizer=None, b_regularizer=None,
                 W_constraint=None, b_constraint=None,
                 bias=True, **kwargs):
        self.supports_masking = True
        self.init = initializers.get('glorot_uniform')

        self.W_regularizer = regularizers.get(W_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)

        self.W_constraint = constraints.get(W_constraint)
        self.b_constraint = constraints.get(b_constraint)

        self.bias = bias
        self.step_dim = step_dim
        self.features_dim = 0
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3

        self.W = self.add_weight((input_shape[-1],),
                                 initializer=self.init,
                                 name='{}_W'.format(self.name),
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)
        self.features_dim = input_shape[-1]

        if self.bias:
            self.b = self.add_weight((input_shape[1],),
                                     initializer='zero',
                                     name='{}_b'.format(self.name),
                                     regularizer=self.b_regularizer,
                                     constraint=self.b_constraint)
        else:
            self.b = None

        self.built = True

    def compute_mask(self, input, input_mask=None):
        return None

    def call(self, x, mask=None):
        features_dim = self.features_dim
        step_dim = self.step_dim

        eij = K.reshape(K.dot(K.reshape(x, (-1, features_dim)),
                        K.reshape(self.W, (features_dim, 1))), (-1, step_dim))

        if self.bias:
            eij += self.b

        eij = K.tanh(eij)
        a = K.exp(eij)

        if mask is not None:
            a *= K.cast(mask, K.floatx())

        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())

        a = K.expand_dims(a)
        weighted_input = x * a
        return K.sum(weighted_input, axis=1)

    def compute_output_shape(self, input_shape):
        return input_shape[0],  self.features_dim
    
def checkFloat(s):
    try: 
        float(s)
        return True
    except ValueError:
        return False

class TextOnly:
    def __init__(self):
        self.logger = get_logger('textonly')

    def get_model(self, num_classes, activation=K.tanh):
        
        max_len = opt.max_len
        voca_size = opt.unigram_hash_size + 1
        
        # ## load pretrained model
        # li_df = []
        # for i in range(1,10):
        #     f = h5py.File("D:/data/train.chunk.0" + str(i), "r")
        #     df = f['train']['product']
        #     li_df += list(df)
        #
        # texts = [str(v.decode('utf-8')) for v in li_df]
        # tokenizer = Tokenizer(num_words=voca_size)
        # tokenizer.fit_on_texts(texts)
        # word_index = tokenizer.word_index
        # # print("[", datetime.now(), "]", 'Found %s unique tokens.' % len(word_index))
        # f.close()
        #
        # embeddings_index = {}
        # with open("./glove.840B.300d.txt", "r", encoding="utf-8") as f:
        #     for line in f:
        #         values = line.split()
        #         pivot = 0
        #         for i, v in enumerate(values):
        #             if checkFloat(v):
        #                 pivot = i
        #                 break
        #
        #         word = " ".join(values[:pivot])
        #         coefs = np.asarray(values[pivot:], dtype='float32')
        #         embeddings_index[word] = coefs
        #
        # # print("[", datetime.now(), "]", 'Found %s word vectors.' % len(embeddings_index))
        #
        # embedding_matrix = np.zeros((voca_size, opt.embd_size))
        # for word, i in word_index.items():
        #     embedding_vector = embeddings_index.get(word)
        #     if i >= voca_size:
        #         break
        #     if embedding_vector is not None:
        #         # words not found in embedding index will be all-zeros.
        #         embedding_matrix[i] = embedding_vector
        #
        # # print("[", datetime.now(), "]", f'embedding_matrix.shape: {embedding_matrix.shape}');
        
        
        with tf.device('/gpu:0'):

            embd = Embedding(voca_size, opt.embd_size, input_length=max_len, name='uni_embd')
            t_uni = Input((max_len, ), name="input_1")
            t_uni_embd = embd(t_uni)  # token
            t_uni_embd = BatchNormalization()(t_uni_embd)
            
            bidirectional_out1 = Bidirectional(layer=LSTM(128, return_sequences=True, dropout=0.15,
                           recurrent_dropout=0.15))(t_uni_embd)

            bidirectional_outputs = Reshape([128, 256, -1])(bidirectional_out1)

#             bidirectional_outputs = Reshape([128, 128, -1])(t_uni_embd)

            filter_sizes = [2, 3, 4, 5, 6]
            small_filter_sizes = [1, 3]

            convolution_outputs = []
            for i, filter_size in enumerate(filter_sizes):
                conv = Conv2D(256, (filter_size, 256), strides=[1, 1], activation='relu')(bidirectional_outputs)
                #conv = Lambda(lambda x : tf.nn.tanh(x))(conv)
                maxpool = MaxPool2D(pool_size=(3, 1), strides=(2, 1))(conv)
                maxpool = Lambda(transpose)(maxpool)
                for j, small_filter_size in enumerate(small_filter_sizes):
                    conv2 = Conv2D(256, (small_filter_size, int(maxpool.shape[2])), strides=[1, 1], activation='relu')(maxpool)
                    # tanh를 사용할 거면 위에 relu 지워야함.
                    # conv2 = Lambda(lambda x : tf.nn.tanh(x))(conv2)
                    maxpool2 = MaxPool2D(pool_size=(int(conv2.shape[1]), 1), strides=(1, 1))((conv2))
                    convolution_outputs.append(maxpool2)

            convolution_concat = Concatenate(axis=3)(convolution_outputs)
            # 위 결과, [batch, 1, 1, output_channel의 갯수]
            flatten = Flatten()(convolution_concat)
            # 위 결과, [batch, output_channel의 갯수]
            flatten = Lambda(lambda x: K.expand_dims(x, 1))(flatten)
            # 위 결과, [batch, 1, output_channel의 갯수]

            attention_out = Attention(1)(flatten)

            attention_out = BatchNormalization()(attention_out)
            drop_out1 = Dropout(0.15)(attention_out)
            
            # fc_1 = Dense(1024, activation='relu')(drop_out1)
            # fc_1 = BatchNormalization()(fc_1)
            # drop_out2 = Dropout(0.25)(fc_1)
            fc_3 = Dense(num_classes, activation=activation)(drop_out1)
            
            outputs = Lambda(lambda x: (x + 1))(fc_3)
            outputs = Lambda(lambda x: x / 2)(outputs)
            
            model = Model(inputs=[t_uni], outputs=outputs)
            optm = keras.optimizers.Adam(opt.lr)
            
            model.compile(loss='binary_crossentropy', optimizer=optm, metrics=[top1_acc])
            model.summary(print_fn=lambda x: self.logger.info(x))
            
        return model
