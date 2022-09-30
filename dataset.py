from utils.utils import cos_sim
from gensim.models import FastText, Word2Vec

import pickle
import random
import os

import sentencepiece as spm
import numpy as np

model = Word2Vec.load('ilbe_fasttext/ilbe_fasttext.model')

count = 0
maxlen = 300
minlen = 0
morphs_files = 'raw_corpus.txt'
origin_files = 'ilbe_x_morphs.txt'

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
        try:
            temp.append(model.wv[ll])
        except Exception as ex:
            a = 1
            # print(str(ex))
    x = sentencepiece.encode_as_ids(o)
    # print(x)
    # flag = True

    xo = sentencepiece.encode_as_pieces(o)
    # print(xo)
    if len(x) <= maxlen or len(x) >= minlen:
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

    line_count = 0
    for _ in range(num):
        with open(f'preprocessing/{morphs_files}',encoding='utf-8') as f:
            with open(f'preprocessing/{origin_files}',encoding='utf-8') as ff:
                text = f.readlines()    
                origin = ff.readlines()
                
                text = text#[:80000]
                origin = origin#[:80000]
                temp = list(zip(text, origin))
                random.shuffle(temp)

                text, origin = zip(*temp)

                X = X + list(text)
                O = O + list(origin)

                # line_count += 1
                # if line_count >= 40000:
                #     break 
    with open(f'preprocessing/{morphs_files}',encoding='utf-8') as f:
        with open(f'preprocessing/{origin_files}',encoding='utf-8') as ff:
            text = f.readlines()    
            origin = ff.readlines()
            text = text#[:80000]
            origin = origin#[:80000]

            X = X + text
            O = O + origin

            # line_count += 1
            # if line_count >= 40000:
            #     break
    
    # temp = list(zip(X, O))
    # random.shuffle(temp)

    # X, O = zip(*temp)
    
    write_s1 = open('preprocessing/ilbe_s1_morphs','w',encoding='utf-8')
    write_s1_origin = open('preprocessing/ilbe_s1_origin','w',encoding='utf-8')

    X_o = []
    O_o = []
    with open(f'preprocessing/{morphs_files}',encoding='utf-8') as f:
        with open(f'preprocessing/{origin_files}',encoding='utf-8') as ff:
            text = f.readlines()
            origin = ff.readlines()
            text = text#[:80000]
            origin = origin#[:80000]
            for _ in range(num+1):
                X_o = X_o + text
                O_o = O_o + origin
                # write_s1.writelines(text)
                # write_s1_origin.writelines(origin)
            
            temp = list(zip(X,O,X_o,O_o))
            random.shuffle(temp)

            X, O, X_o, O_o = zip(*temp)
            with open('preprocessing/ilbe_compare.txt','w',encoding='utf-8') as f:
                f.writelines(X)
            with open('preprocessing/ilbe_origin.txt','w',encoding='utf-8') as f:
                f.writelines(O)

            write_s1.writelines(X_o)
            write_s1_origin.writelines(O_o)
            
            write_s1.close()
            write_s1_origin.close()
    
def spm_train(filename,save_path='ilbe_spm_model/ilbe_spm',vocab_size=10000):
    #with open(f'preprocessing/{filename}', encoding='utf-8') as ff:
    templates = '--input={} --model_prefix={} --vocab_size={}'
    cmd = templates.format(filename,save_path,vocab_size)
    spm.SentencePieceTrainer.Train(cmd)

def to_numpy(sentencepiece, batch_size=32):
    global word
    # file = open('result.txt','w',encoding='utf-8')
    # length = []

    X = []
    X2 = []
    Y = []

    if not os.path.exists('data'):
        os.makedirs('data')
    X = []
    X2 = []
    Y = []
    # fy = open('')
    count = 1
    from tqdm import tqdm
    with open(f'preprocessing/ilbe_s1_morphs', encoding='utf-8') as f:
        with open(f'preprocessing/ilbe_compare.txt', encoding='utf-8') as ff:
            # s1 = open('preprocessing/ilbe_s1_origin','w',encoding='utf-8')
            s1_origin = open('preprocessing/ilbe_s1_origin',encoding='utf-8')
            origin = open('preprocessing/ilbe_origin.txt',encoding='utf-8')
            origin_sp = open('preprocessing/ilbe_sp.txt', 'w', encoding='utf-8')
            origin_cp = open('preprocessing/ilbe_cp.txt','w', encoding='utf-8')
            line_count = 0
            for _ in tqdm(range(8920704)):
            # for l, c, o, cpo in zip(f, ff, s1_origin, origin):
                l = f.readline()
                c = ff.readline()
                o = s1_origin.readline()
                cpo = origin.readline()

                l = l.strip()
                c = c.strip()
                if l == '' or c=='':
                    continue
                l = l.split(' ')
                c = c.split(' ')

                # ll = ' '.join(l)
                # cc = ' '.join(c)
                # length.append(len(l))
                # continue
                x, xo, temp = set_word2index(l, o, sentencepiece)
                x2, xo2, temp2 = set_word2index(c, cpo, sentencepiece)
                origin_sp.write(' '.join(xo)+'\n')
                origin_cp.write(' '.join(xo2)+'\n')
                x = x[:maxlen]
                x2 = x2[:maxlen]
                if len(temp) == 0 or len(temp2) == 0 or (len(x) <= minlen) or (len(x2) <= minlen):
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
                    # print(temp___,vector)
                    # print(temp2___,vector2)
                    y = cos_sim(vector, vector2)
                    # print(y)
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
                    if count % 100 == 0:
                        print(f"save {count} numpy")
                    X = []
                    X2 = []
                    Y = []

                    if count == 15000:
                        break
                # line_count += 1
                # if line_count >= 40000:
                #     break
    origin_cp.close()
    origin_sp.close()
    with open('word.pkl','wb') as f:
        pickle.dump(word,f)

if False:
    random_file()
if False:
    inputf = f'preprocessing/{morphs_files}'

    spm_train(inputf)
sp = spm.SentencePieceProcessor()
vocab_file = 'ilbe_spm_model/ilbe_spm.model'
sp.load(vocab_file)

to_numpy(sp,batch_size=64)
# length = sorted(length,reverse=True)
# print(length[:1000])
# print(len(length)/sum(length))
