from utils.utils import cos_sim
from gensim.models import FastText

import pickle
import random
import os

import sentencepiece as spm
import numpy as np

model = FastText.load('ilbe_fasttext/ilbe_fasttext.model')

count = 0
maxlen = 250
minlen = 3
morphs_files = 'ilbe_nouns_40000.txt'
origin_files = 'ilbe_x_40000.txt'

word = {}
word['[PAD]'] = 0
word['[SEP]'] = 1
word['[START]'] = 2
word['[END]'] = 3
wordindex = 4

length = []


def set_word2index(l, o, sentencepiece):
    global wordindex
    global word
    global length
    x = []
    xo = []
    temp = []
    for ll in l:
        if ll not in word:
            word[ll] = wordindex
            wordindex += 1

        # x.append(word[ll])
        temp.append(model.wv[ll])
    x = sentencepiece.encode_as_ids(o)
    # print(x)
    xo = sentencepiece.encode_as_pieces(o)
    length.append(len(x))
    return x, xo, temp

def average_vector(temp___, length_):
    vector = np.zeros(300)
    for t2 in temp___:
        vector += t2
    vector = vector / length_

    return vector

def random_file(num=3):
    X = []
    O = []

    for _ in range(num):
        with open(f'preprocessing/{morphs_files}',encoding='utf-8') as f:
            with open(f'preprocessing/{origin_files}',encoding='utf-8') as ff:
                text = f.readlines()    
                origin = ff.readlines()
                temp = list(zip(text, origin))
                random.shuffle(temp)

                text, origin = zip(*temp)

                X = X + list(text)
                O = O + list(origin)

    with open(f'preprocessing/{morphs_files}',encoding='utf-8') as f:
        with open(f'preprocessing/{origin_files}',encoding='utf-8') as ff:
            text = f.readlines()    
            origin = ff.readlines()
            
            X = X + text
            O = O + origin
    
    with open('preprocessing/ilbe_compare.txt','w',encoding='utf-8') as f:
        f.writelines(X)
    with open('preprocessing/ilbe_origin.txt','w',encoding='utf-8') as f:
        f.writelines(O)

def spm_train(filename,save_path='ilbe_spm_model/ilbe_spm',vocab_size=10000):
    #with open(f'preprocessing/{filename}', encoding='utf-8') as ff:
    templates = '--input={} --model_prefix={} --vocab_size={}'
    cmd = templates.format(filename,save_path,vocab_size)
    spm.SentencePieceTrainer.Train(cmd)

def to_numpy(sentencepiece, batch_size=32):
    global word

    length = []

    X = []
    X2 = []
    Y = []

    if not os.path.exists('data'):
        os.makedirs('data')
    X = []
    X2 = []
    Y = []
    count = 1
    with open(f'preprocessing/{morphs_files}', encoding='utf-8') as f:
        with open(f'preprocessing/ilbe_compare.txt', encoding='utf-8') as ff:
            origin = open('preprocessing/ilbe_origin.txt',encoding='utf-8')
            origin_sp = open('preprocessing/ilbe_sp.txt', 'w', encoding='utf-8')
            origin_cp = open('preprocessing/ilbe_cp.txt','w', encoding='utf-8')
            for l, c, o in zip(f, ff, origin):
                l = l.strip()
                c = c.strip()

                l = l.split(' ')
                c = c.split(' ')

                x, xo, temp = set_word2index(l, o, sentencepiece)
                x2, xo2, temp2 = set_word2index(c, o, sentencepiece)
                origin_sp.write(' '.join(xo)+'\n')
                origin_cp.write(' '.join(xo2)+'\n')
                if len(temp) == 0 or len(temp2) == 0 or (len(x) >= maxlen or len(x) <= minlen) or (len(x2) >= maxlen or len(x2) <= minlen):
                    continue
                length_ = len(x)
                length2_ = len(x2)

                temp_ = np.array(temp)
                temp2_ = np.array(temp2)

                wordmap = {v: k for k, v in word.items()}
                if temp_.shape[0] != 0:
                    temp___ = np.matmul(temp_, np.transpose(temp_, [1, 0]))
                    temp___ = np.matmul(temp___, temp_)

                    temp2___ = np.matmul(temp2_, np.transpose(temp2_, [1, 0]))
                    temp2___ = np.matmul(temp2___, temp2_)

                    vector = average_vector(temp___, length_)
                    vector2 = average_vector(temp2___, length2_)

                    y = cos_sim(vector, vector2)
                    x = x + [0] * (maxlen - len(x))
                    x2 = x2 + [0] * (maxlen - len(x2))
                    X.append(x)
                    X2.append(x2)
                    Y.append(y)
                
                if len(X) == batch_size:
                    X = np.array(X)
                    X2 = np.array(X2)
                    Y = np.array(Y)

                    np.save(f'data/{count}_x',X)
                    np.save(f'data/{count}_x2',X2)
                    np.save(f'data/{count}_y',Y)

                    count += 1

                    X = []
                    X2 = []
                    Y = []
    origin_cp.close()
    origin_sp.close()
    with open('word.pkl','wb') as f:
        pickle.dump(word,f)

random_file()
inputf = f'preprocessing/{origin_files}'

spm_train(inputf)
sp = spm.SentencePieceProcessor()
vocab_file = 'ilbe_spm_model/ilbe_spm.model'
sp.load(vocab_file)

to_numpy(sp)
length = sorted(length,reverse=True)
print(length[:20])
print(len(length)/sum(length))
