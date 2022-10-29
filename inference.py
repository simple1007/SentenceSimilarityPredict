import sys
sys.path.append('D:/SentenceSimilarityPredict/noun_add_verb/HeadTail_Tokenizer_POSTagger')

from utils.utils import cos_sim_per as cos_sim
# from konlpy.tag import Okt
from HeadTail_Tokenizer_POSTagger.head_tail import analysis

import tensorflow as tf
import numpy as np
import sentencepiece as spm

# o = Okt()
maxlen = 300

sp = spm.SentencePieceProcessor()
vocab_file = 'ilbe_ht_spm_model/ilbe_spm.model'
sp.load(vocab_file)

gender = np.load('emb/gen.npy')
society = np.load('emb/soc.npy')
# age = np.load('emb/age.npy')
# location = np.load('emb/location.npy')
# gender = np.load('emb/gen_sent.npy')
model = tf.keras.models.load_model('emb/embedding_model')
tag = ['N']#,'V']
# from konlpy.tag import Okt
# o = Okt()
# ['▁나', '▁밥', '▁학교']
while True:
    x = input("input sentence: ")
    if x.lower() == 'exit':
        break
    x = analysis(x)
    print(x)
    x = x[0].split(' ')
    
    temp = []
    for xx in x:
        xx = xx.split('+')[0]
        xx = xx.split('/')
        if xx[1][0] in tag:
            temp.append(xx[0])

    x = ' '.join(temp)
    temp = x.split(' ')
    tt = sp.encode_as_pieces(x)
    x = sp.encode_as_ids(x)
    # print(tt)
    print(tt)
    x = x + [0] * (maxlen - len(x))
    x = x[:maxlen]

    pred = model(np.array([x]))
    print(pred)
    temp_piece = []
    # cnt = 0
    start = 0
    tmp = 0
    # start = 1
    # temp_piece.append([tmp])
    for index, t in enumerate(tt):
        if (t.startswith('▁') or t == '▁'):
            if index != 0:
                temp_piece[-1].append(start-1)
            temp_piece.append([tmp])
        # elif index != 0:
        #     temp_piece[-1].append(tmp)    
        if index == len(tt)-1:
            temp_piece[-1].append(start)
        tmp = start + 1
        start += 1
        # cnt += 1
    print(temp_piece)
    # for pr in pred[0]:
    v_t = []
    print(pred.shape)
    for tp in temp_piece:
        sm = np.sum(pred[0][tp[0]:tp[1] + 1],axis=0)
        # sm = np.zeros(128)
        # for i in range(tp[0],tp[1]+1):
        #     sm += pred[0][i]
        # print(sm)
        sm = sm / (tp[1] - tp[0] + 1)
        v_t.append(sm)
    # print(v_t,gender)
        # print(pr,"젠더 혐오:",cos_sim(pr,gender),'사회/정치 혐오:',cos_sim(pr,society))
    # print(temp)
    for tp,vt in zip(temp,v_t):
        # print(vt.shape)
        # print(gender.shape)
        print(tp,"젠더 혐오:",cos_sim(vt,gender),'사회/정치 혐오:',cos_sim(vt,society))
        # temp.append(xx[0])
    # break
    # x = ' '.join(temp)
    # # x = ' '.join(x)
    # print(x)
    # x = sp.encode_as_ids(x)
    # x = x + [0] * (maxlen - len(x))
    # x = x[:maxlen]

    # pred = model(np.array([x]))
    # print("젠더 혐오:",cos_sim(pred[0],gender),'사회/정치 혐오:',cos_sim(pred[0],society))
    # # print("젠더 혐오:",cos_sim(pred[0],gender),"정치 혐오:",cos_sim(pred[0],society),"연령 혐오:",cos_sim(pred[0],age),"지역 혐오:",cos_sim(pred[0],location))