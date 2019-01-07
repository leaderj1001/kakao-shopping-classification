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
import os

os.environ['OMP_NUM_THREADS'] = '1'
import re
import sys
import traceback
from collections import Counter
from multiprocessing import Pool

import tqdm
import fire
import h5py
import numpy as np
import mmh3
import six
from keras.utils.np_utils import to_categorical
from six.moves import cPickle

from keras.preprocessing.text import Tokenizer
from misc import get_logger, Option

opt = Option('./config.json')

re_sc = re.compile('[\!@#$%\^&\*\(\)-=\[\]\{\}\.,/\?~\+\'"|]')


class Reader(object):
    def __init__(self, data_path_list, div, begin_offset, end_offset):
        self.div = div
        self.data_path_list = data_path_list
        self.begin_offset = begin_offset
        self.end_offset = end_offset

    def is_range(self, i):
        if self.begin_offset is not None and i < self.begin_offset:
            return False
        if self.end_offset is not None and self.end_offset <= i:
            return False
        return True

    def get_size(self):
        offset = 0
        count = 0
        for data_path in self.data_path_list:
            h = h5py.File(data_path, 'r')
            sz = h[self.div]['pid'].shape[0]
            if not self.begin_offset and not self.end_offset:
                offset += sz
                count += sz
                continue
            if self.begin_offset and offset + sz < self.begin_offset:
                offset += sz
                continue
            if self.end_offset and self.end_offset < offset:
                break
            for i in range(sz):
                if not self.is_range(offset + i):
                    continue
                count += 1
            offset += sz
        return count

    def get_class(self, h, i):
        b = h['bcateid'][i]
        m = h['mcateid'][i]
        s = h['scateid'][i]
        d = h['dcateid'][i]
        return '%s>%s>%s>%s' % (b, m, s, d)

    def generate(self):
        offset = 0
        for data_path in self.data_path_list:
            h = h5py.File(data_path, 'r')[self.div]
            sz = h['pid'].shape[0]
            if self.begin_offset and offset + sz < self.begin_offset:
                offset += sz
                continue
            if self.end_offset and self.end_offset < offset:
                break
            for i in range(sz):
                if not self.is_range(offset + i):
                    continue
                class_name = self.get_class(h, i)
                yield h['pid'][i], class_name, h, i
            offset += sz

    def get_y_vocab(self, data_path):
        y_vocab = {}
        h = h5py.File(data_path, 'r')[self.div]
        sz = h['pid'].shape[0]
        for i in tqdm.tqdm(range(sz), mininterval=1):
            class_name = self.get_class(h, i)
            if class_name not in y_vocab:
                y_vocab[class_name] = len(y_vocab)
        return y_vocab


def preprocessing(data):
    try:
        cls, data_path_list, div, out_path, begin_offset, end_offset = data
        data = cls()
        data.load_y_vocab()
        data.preprocessing(data_path_list, div, begin_offset, end_offset, out_path)
    except Exception:
        raise Exception("".join(traceback.format_exception(*sys.exc_info())))


def build_y_vocab(data):
    try:
        data_path, div = data
        reader = Reader([], div, None, None)
        y_vocab = reader.get_y_vocab(data_path)
    except Exception:
        raise Exception("".join(traceback.format_exception(*sys.exc_info())))
    return y_vocab


cho = "ㄱㄲㄴㄷㄸㄹㅁㅂㅃㅅㅆㅇㅈㅉㅊㅋㅌㅍㅎ"  # len = 19
jung = "ㅏㅐㅑㅒㅓㅔㅕㅖㅗㅘㅙㅚㅛㅜㅝㅞㅟㅠㅡㅢㅣ"  # len = 21
# len = 27
jong = "ㄱ/ㄲ/ㄱㅅ/ㄴ/ㄴㅈ/ㄴㅎ/ㄷ/ㄹ/ㄹㄱ/ㄹㅁ/ㄹㅂ/ㄹㅅ/ㄹㅌ/ㄹㅍ/ㄹㅎ/ㅁ/ㅂ/ㅂㅅ/ㅅ/ㅆ/ㅇ/ㅈ/ㅊ/ㅋ/ㅌ/ㅍ/ㅎ".split(
    '/')
test = cho + jung + ''.join(jong)

hangul_length = len(cho) + len(jung) + len(jong)  # 67


# def decompose_as_one_hot(in_char, warning=True):
#     one_hot = []
#     # print(ord('ㅣ'), chr(0xac00))
#     # [0,66]: hangul / [67,194]: ASCII / [195,245]: hangul danja,danmo / [246,249]: special characters
#     # Total 250 dimensions.
#     if ord('가') <= in_char <= ord('힣'):  # 가:44032 , 힣: 55203
#         x = in_char - 44032  # in_char - ord('가')
#         y = x // 28
#         z = x % 28
#         x = y // 21
#         y = y % 21
#         # if there is jong, then is z > 0. So z starts from 1 index.
#         zz = jong[z - 1] if z > 0 else ''
#         if x >= len(cho):
#             if warning:
#                 print('Unknown Exception: ', in_char,
#                       chr(in_char), x, y, z, zz)
#
#         one_hot.append(x)
#         one_hot.append(len(cho) + y)
#         if z > 0:
#             one_hot.append(len(cho) + len(jung) + (z - 1))
#         return one_hot
#     else:
#         if in_char < 128:
#             result = hangul_length + in_char  # 67~
#         elif ord('ㄱ') <= in_char <= ord('ㅣ'):
#             # 194~ # [ㄱ:12593]~[ㅣ:12643] (len = 51)
#             result = hangul_length + 128 + (in_char - 12593)
#         elif in_char == ord('♡'):
#             result = hangul_length + 128 + 51  # 245~ # ♡
#         elif in_char == ord('♥'):
#             result = hangul_length + 128 + 51 + 1  # ♥
#         elif in_char == ord('★'):
#             result = hangul_length + 128 + 51 + 2  # ★
#         elif in_char == ord('☆'):
#             result = hangul_length + 128 + 51 + 3  # ☆
#         else:
#             if warning:
#                 print('Unhandled character:', chr(in_char), in_char)
#             # unknown character
#             result = hangul_length + 128 + 51 + 4  # for unknown character
#
#         return [result]

def decompose_as_one_hot(in_char, warning=True):
    one_hot = []
    # print(ord('ㅣ'), chr(0xac00))
    # [0,66]: hangul / [67,194]: ASCII / [195,245]: hangul danja,danmo / [246,249]: special characters
    # Total 250 dimensions.
    if ord('가') <= in_char <= ord('힣'):  # 가:44032 , 힣: 55203
        x = in_char - 44032  # in_char - ord('가')
        y = x // 28
        z = x % 28
        x = y // 21
        y = y % 21
        # if there is jong, then is z > 0. So z starts from 1 index.
        zz = jong[z - 1] if z > 0 else ''
        if x >= len(cho):
            if warning:
                print('Unknown Exception: ', in_char,
                      chr(in_char), x, y, z, zz)

        one_hot.append(x)
        one_hot.append(len(cho) + y)
        if z > 0:
            one_hot.append(len(cho) + len(jung) + (z - 1))
        return one_hot
    else:
        if in_char < 128:
            result = hangul_length + in_char  # 67~
        elif ord('ㄱ') <= in_char <= ord('ㅣ'):
            # 194~ # [ㄱ:12593]~[ㅣ:12643] (len = 51)
            result = hangul_length + 128 + (in_char - 12593)
        elif in_char == ord('♡'):
            result = hangul_length + 128 + 51  # 245~ # ♡
        elif in_char == ord('♥'):
            result = hangul_length + 128 + 51 + 1  # ♥
        elif in_char == ord('★'):
            result = hangul_length + 128 + 51 + 2  # ★
        elif in_char == ord('☆'):
            result = hangul_length + 128 + 51 + 3  # ☆
        elif 13312 <= in_char <= 40917:
            result = hangul_length + 128 + 51 + 3 + (in_char - 13312)
        elif in_char == 12288:
            result = hangul_length + 128 + 51 + 3 + 27606 + 1
        elif 11904 <= in_char <= 12245:
            result = hangul_length + 128 + 51 + 3 + 27606 + 1 + (in_char - 11904)
        elif 12353 <= in_char <= 12589:
            result = hangul_length + 128 + 51 + 3 + 27606 + 1 + 342 + (in_char - 12353)
        else:
            if warning:
                print('Unhandled character:', chr(in_char), in_char)
            # unknown character
            # result = hangul_length + 128 + 51 + 4  # for unknown character
            result = hangul_length + 128 + 51 + 4 + 27606 + 1 + 342 + 236  # for unknown character

        return [result]


def decompose_str_as_one_hot(string, warning=True):
    tmp_list = []
    for x in string:
        da = decompose_as_one_hot(ord(x), warning=warning)
        tmp_list.extend(da)
    return tmp_list


class Data:
    y_vocab_path = './data/y_vocab.cPickle' if six.PY2 else './data/y_vocab.py3.cPickle'
    tmp_chunk_tpl = './tmp/base.chunk.%s'

    def __init__(self):
        self.logger = get_logger('data')

    def load_y_vocab(self):
        self.y_vocab = cPickle.loads(open(self.y_vocab_path, 'rb').read())

    def build_y_vocab(self):
        pool = Pool(opt.num_workers)
        try:
            rets = pool.map_async(build_y_vocab,
                                  [(data_path, 'train')
                                   for data_path in opt.train_data_list]).get(999999)
            pool.close()
            pool.join()
            y_vocab = set()
            for _y_vocab in rets:
                for k in six.iterkeys(_y_vocab):
                    y_vocab.add(k)
            self.y_vocab = {y: idx for idx, y in enumerate(y_vocab)}
        except KeyboardInterrupt:
            pool.terminate()
            pool.join()
            raise
        self.logger.info('size of y vocab: %s' % len(self.y_vocab))
        cPickle.dump(self.y_vocab, open(self.y_vocab_path, 'wb'), 2)

    def _split_data(self, data_path_list, div, chunk_size):
        total = 0
        for data_path in data_path_list:
            h = h5py.File(data_path, 'r')
            sz = h[div]['pid'].shape[0]
            total += sz
        chunks = [(i, min(i + chunk_size, total))
                  for i in range(0, total, chunk_size)]
        return chunks

    def preprocessing(self, data_path_list, div, begin_offset, end_offset, out_path):
        self.div = div
        reader = Reader(data_path_list, div, begin_offset, end_offset)
        rets = []
        for pid, label, h, i in reader.generate():
            y, x = self.parse_data(label, h, i)
            if y is None:
                continue
            rets.append((pid, y, x))
        self.logger.info('sz=%s' % (len(rets)))
        open(out_path, 'wb').write(cPickle.dumps(rets, 2))
        self.logger.info('%s ~ %s done. (size: %s)' % (begin_offset, end_offset, end_offset - begin_offset))

    def _preprocessing(self, cls, data_path_list, div, chunk_size):
        chunk_offsets = self._split_data(data_path_list, div, chunk_size)
        num_chunks = len(chunk_offsets)
        self.logger.info('split data into %d chunks, # of classes=%s' % (num_chunks, len(self.y_vocab)))
        pool = Pool(opt.num_workers)
        try:
            pool.map_async(preprocessing, [(cls,
                                            data_path_list,
                                            div,
                                            self.tmp_chunk_tpl % cidx,
                                            begin,
                                            end)
                                           for cidx, (begin, end) in enumerate(chunk_offsets)]).get(999999)
            pool.close()
            pool.join()
        except KeyboardInterrupt:
            pool.terminate()
            pool.join()
            raise
        return num_chunks

    def parse_data(self, label, h, i):
        Y = self.y_vocab.get(label)
        if Y is None and self.div in ['dev', 'test']:
            Y = 0
        if Y is None and self.div != 'test':
            return [None] * 2
        Y = to_categorical(Y, len(self.y_vocab))

        product = h['product'][i]
        maker = h['maker'][i]
        brand = h['brand'][i]
        model_ = h['model'][i]
        if six.PY3:
            product = product.decode('utf-8')
            maker = maker.decode('utf-8')
            brand = brand.decode('utf-8')
            model_ = model_.decode('utf-8')

        # 상품 이름, 제조사, 상표, model 다 하나의 문장으로 만듦.
        product = product + ' ' + maker + ' ' + brand + ' ' + model_
        # 불필요한 느낌표, 물음표 다 제거
        # product = re_sc.sub(' ', product).strip().split()

        result = []
        vectorized_data = decompose_str_as_one_hot(product, warning=False)
        result.append(vectorized_data)

        zero_padding = np.zeros((len(result), opt.max_len), dtype=np.int32)
        for idx, seq in enumerate(result):
            length = len(seq)
            if length >= opt.max_len:
                length = opt.max_len
                zero_padding[idx, :length] = np.array(seq)[:length]
            else:
                zero_padding[idx, :length] = np.array(seq)
        # # 띄어쓰기 제거
        # words = [w.strip() for w in product]
        # # length 제한으로 자르기
        # words = [w for w in words
        #          if len(w) >= opt.min_word_length and len(w) < opt.max_word_length]
        # if not words:
        #     return [None] * 2
        #
        # hash_func = hash if six.PY2 else lambda x: mmh3.hash(x, seed=17)
        # # hash function으로 단어를 숫자로 매칭
        # x = [hash_func(w) % opt.unigram_hash_size + 1 for w in words]
        # # 단어, 단어 갯수의 tuple로 만듦.
        # xv = Counter(x).most_common(opt.max_len)
        #
        # x = np.zeros(opt.max_len, dtype=np.float32)
        # v = np.zeros(opt.max_len, dtype=np.int32)
        # for i in range(len(xv)):
        #     x[i] = xv[i][0]
        #     v[i] = xv[i][1]

        return Y, (zero_padding)

    def create_dataset(self, g, size, num_classes):
        shape = (size, opt.max_len)
        g.create_dataset('uni', shape, chunks=True, dtype=np.int32)
        # g.create_dataset('w_uni', shape, chunks=True, dtype=np.float32)
        g.create_dataset('cate', (size, num_classes), chunks=True, dtype=np.int32)
        g.create_dataset('pid', (size,), chunks=True, dtype='S12')

    def init_chunk(self, chunk_size, num_classes):
        chunk_shape = (chunk_size, opt.max_len)
        chunk = {}
        chunk['uni'] = np.zeros(shape=chunk_shape, dtype=np.int32)
        # chunk['w_uni'] = np.zeros(shape=chunk_shape, dtype=np.float32)
        chunk['cate'] = np.zeros(shape=(chunk_size, num_classes), dtype=np.int32)
        chunk['pid'] = []
        chunk['num'] = 0
        return chunk

    def copy_chunk(self, dataset, chunk, offset, with_pid_field=False):
        num = chunk['num']
        dataset['uni'][offset:offset + num, :] = chunk['uni'][:num]
        # dataset['w_uni'][offset:offset + num, :] = chunk['w_uni'][:num]
        dataset['cate'][offset:offset + num] = chunk['cate'][:num]
        if with_pid_field:
            dataset['pid'][offset:offset + num] = chunk['pid'][:num]

    def copy_bulk(self, A, B, offset, y_offset, with_pid_field=False):
        num = B['cate'].shape[0]
        y_num = B['cate'].shape[1]
        A['uni'][offset:offset + num, :] = B['uni'][:num]
        # A['w_uni'][offset:offset + num, :] = B['w_uni'][:num]
        A['cate'][offset:offset + num, y_offset:y_offset + y_num] = B['cate'][:num]
        if with_pid_field:
            A['pid'][offset:offset + num] = B['pid'][:num]

    def get_train_indices(self, size, train_ratio):
        train_indices = np.random.rand(size) < train_ratio
        train_size = int(np.count_nonzero(train_indices))
        return train_indices, train_size

    # parameter로 생성되는 파일 위치 지정가능
    def make_db(self, data_name, output_dir='./data/train_test', train_ratio=0.8):
        if data_name == 'train':
            div = 'train'
            data_path_list = opt.train_data_list
        elif data_name == 'dev':
            div = 'dev'
            data_path_list = opt.dev_data_list
        elif data_name == 'test':
            div = 'test'
            data_path_list = opt.test_data_list
        else:
            assert False, '%s is not valid data name' % data_name

        all_train = train_ratio >= 1.0
        all_dev = train_ratio == 0.0

        np.random.seed(17)
        self.logger.info('make database from data(%s) with train_ratio(%s)' % (data_name, train_ratio))

        self.load_y_vocab()
        num_input_chunks = self._preprocessing(Data,
                                               data_path_list,
                                               div,
                                               chunk_size=opt.chunk_size)
        if not os.path.isdir(output_dir):
            os.makedirs(output_dir)

        data_fout = h5py.File(os.path.join(output_dir, 'data.h5py'), 'w')
        meta_fout = open(os.path.join(output_dir, 'meta'), 'wb')

        reader = Reader(data_path_list, div, None, None)
        tmp_size = reader.get_size()
        train_indices, train_size = self.get_train_indices(tmp_size, train_ratio)

        dev_size = tmp_size - train_size
        if all_dev:
            train_size = 1
            dev_size = tmp_size
        if all_train:
            dev_size = 1
            train_size = tmp_size

        train = data_fout.create_group('train')
        dev = data_fout.create_group('dev')
        self.create_dataset(train, train_size, len(self.y_vocab))
        self.create_dataset(dev, dev_size, len(self.y_vocab))
        self.logger.info('train_size ~ %s, dev_size ~ %s' % (train_size, dev_size))

        sample_idx = 0
        dataset = {'train': train, 'dev': dev}
        num_samples = {'train': 0, 'dev': 0}
        chunk_size = opt.db_chunk_size
        chunk = {'train': self.init_chunk(chunk_size, len(self.y_vocab)),
                 'dev': self.init_chunk(chunk_size, len(self.y_vocab))}
        chunk_order = list(range(num_input_chunks))
        print(chunk_order)
        np.random.shuffle(chunk_order)
        for input_chunk_idx in chunk_order:
            path = os.path.join(self.tmp_chunk_tpl % input_chunk_idx)
            self.logger.info('processing %s ...' % path)
            data = list(enumerate(cPickle.loads(open(path, 'rb').read())))
            np.random.shuffle(data)
            for data_idx, (pid, y, vw) in data:
                if y is None:
                    continue
                v = vw
                is_train = train_indices[sample_idx + data_idx]
                if all_dev:
                    is_train = False
                if all_train:
                    is_train = True
                if v is None:
                    continue
                c = chunk['train'] if is_train else chunk['dev']
                idx = c['num']
                c['uni'][idx] = v
                # c['w_uni'][idx] = w
                c['cate'][idx] = y
                c['num'] += 1
                if not is_train:
                    c['pid'].append(np.string_(pid))
                for t in ['train', 'dev']:
                    if chunk[t]['num'] >= chunk_size:
                        self.copy_chunk(dataset[t], chunk[t], num_samples[t],
                                        with_pid_field=t == 'dev')
                        num_samples[t] += chunk[t]['num']
                        chunk[t] = self.init_chunk(chunk_size, len(self.y_vocab))
            sample_idx += len(data)
        for t in ['train', 'dev']:
            if chunk[t]['num'] > 0:
                self.copy_chunk(dataset[t], chunk[t], num_samples[t],
                                with_pid_field=t == 'dev')
                num_samples[t] += chunk[t]['num']

        for div in ['train', 'dev']:
            ds = dataset[div]
            size = num_samples[div]
            shape = (size, opt.max_len)
            ds['uni'].resize(shape)
            # ds['w_uni'].resize(shape)
            ds['cate'].resize((size, len(self.y_vocab)))

        data_fout.close()
        meta = {'y_vocab': self.y_vocab}
        meta_fout.write(cPickle.dumps(meta, 2))
        meta_fout.close()

        self.logger.info('# of classes: %s' % len(meta['y_vocab']))
        self.logger.info('# of samples on train: %s' % num_samples['train'])
        self.logger.info('# of samples on dev: %s' % num_samples['dev'])
        self.logger.info('data: %s' % os.path.join(output_dir, 'data.h5py'))
        self.logger.info('meta: %s' % os.path.join(output_dir, 'meta'))


if __name__ == '__main__':
    data = Data()
    fire.Fire({'make_db': data.make_db,
               'build_y_vocab': data.build_y_vocab})
