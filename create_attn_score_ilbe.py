from ast import Lambda
import numpy as np
from gensim.models import FastText

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

model = FastText.load('ilbe_fasttext/ilbe_fasttext.model')
# x = open('x.txt','w',encoding='utf-8')
# y = open('y.txt','w',encoding='utf-8')
# print(model.wv['pad'])

length = []
maxlen = 200
X = []
X2 = []
Y = []
word = {}
word['[PAD]'] = 0
word['[SEP]'] = 1
word['[START]'] = 2
word['[END]'] = 3
wordindex = 4

from numpy import dot
from numpy.linalg import norm
import pickle

# print(pred[0].shape,Y[0].shape)

def cos_sim(A, B):
#   print(A.tolist(),B.tolist())
    # return dot(A, B)/(norm(A)*norm(B))
    sim = dot(A, B)/(norm(A)*norm(B))
    # print(1,sim)
    # sim = (sim - (-1)) / (1 - (-1))
    sim = sim * 100.0

    # print(sim)
    return sim
    # print(sim)
    # if sim > 0.6:
    #     return 1
    # elif sim <= 0.6:
    #     return 0
    # if sim < 0:
    #     return 1 - (np.arccos(sim) / np.pi)
    # elif sim > 0:
    #     return 1 - ((2 * np.arccos(sim)) / np.pi)
    # else:
    #     return 0
count = 0
import random
morphs_files = 'morphs.txt'

def set_word2index(l):
    global wordindex
    global word
    x = []
    temp = []
    for ll in l:
        # if ll in model.wv:
        if ll not in word:
            word[ll] = wordindex
            wordindex += 1
        x.append(word[ll])  
        temp.append(model.wv[ll])
    
    return x, temp

def average_vector(temp___,length_):
    vector = np.zeros(300)
    for t2 in temp___:
        vector += t2
    vector = vector / length_

    return vector

if True:
    with open(f'preprocessing/{morphs_files}',encoding='utf-8') as f:
        text = f.readlines()    
        random.shuffle(text)
        with open('preprocessing/compare1.txt','w',encoding='utf-8') as f:
            f.writelines(text)
    with open(f'preprocessing/{morphs_files}',encoding='utf-8') as f:
        text = f.readlines()    
        random.shuffle(text)
        with open('preprocessing/compare2.txt','w',encoding='utf-8') as f:
            f.writelines(text)
    with open(f'preprocessing/{morphs_files}',encoding='utf-8') as f:
        text = f.readlines()    
        # random.shuffle(text)
        with open('preprocessing/compare3.txt','w',encoding='utf-8') as f:
            f.writelines(text)
    for iiii in range(2):
        with open(f'preprocessing/{morphs_files}',encoding='utf-8') as f:
            with open(f'preprocessing/compare{iiii+1}.txt',encoding='utf-8') as ff:
                for l,c in zip(f,ff):
                    l = l.strip()
                    c = c.strip()
                    
                    l = l.split(' ')
                    c = c.split(' ')

                    x, temp = set_word2index(l)
                    x2, temp2 = set_word2index(c)

                    if len(temp) == 0 or len(temp2) == 0:
                        continue
                    length_ = len(x)
                    length2_ = len(x2)
                    
                    temp_ = np.array(temp)
                    temp2_ = np.array(temp2)
                    
                    wordmap = {v:k for k,v in word.items()}
                    if temp_.shape[0] != 0:
                        temp___ = np.matmul(temp_,np.transpose(temp_,[1,0]))
                        temp___ = np.matmul(temp___,temp_)

                        temp2___ = np.matmul(temp2_,np.transpose(temp2_,[1,0]))
                        temp2___ = np.matmul(temp2___,temp2_)

                        vector = average_vector(temp___,length_)
                        vector2 = average_vector(temp2___,length2_)

                        y = cos_sim(vector,vector2)
                        x = x + [0] * (maxlen - len(x))
                        x2 = x2 + [0] * (maxlen - len(x2))
                        X.append(x)
                        X2.append(x2)
                        Y.append(y)
                        
    with open('word.pkl','wb') as f:
        pickle.dump(word,f)
# print(max(length))
# x.close()
# y.close()
# print(count)
# import sys
# sys.exit()

import tensorflow as tf
from tensorflow.keras.layers import LSTM, Bidirectional, TimeDistributed, Dense, Input, Embedding, Lambda, Add
# tf.keras.layers.Input((,300))
# print()
# tf.keras.preprocessing.text.Tokenizer
def dotp(x):
    return tf.matmul(x,x,transpose_b=False)
input = Input((maxlen))
input2 = Input((maxlen))
emb = Embedding(len(word.keys()),100)(input)
emb2 = Embedding(len(word.keys()),100)(input2)
# lstm = Bidirectional(LSTM(200,return_sequences=False))(emb)
bilstm = Bidirectional(LSTM(64))(emb)
bilstm2 = Bidirectional(LSTM(64))(emb2)
multiply = tf.multiply(bilstm,bilstm2)

# lstm = Bidirectional(LSTM(32))(lstm)
# output = Lambda(dotp)(lstm)

bilstm_ = Dense(32, activation = 'relu')(multiply)
# bilstm2_ = Dense(32)(bilstm2)
output = Dense(1)(bilstm_)

# output2 = Dense(32)(bilstm2)


print(output.shape)
print(type(emb))
print(type(output))
# print(output)
# output = Dense(72)(output)
# output = Dense(maxlen*maxlen)(lstm)

model = tf.keras.models.Model(inputs=[input,input2],outputs=output)
model.summary()
# import sys
# sys.exit()
model.compile(optimizer="adam",metrics=["mae"],loss="mse")
X = np.array(X)
X2 = np.array(X2)
Y = np.array(Y)
if False:
    print(X.shape,Y.shape,X[0],Y[1])
    model.fit([X,X2],Y,epochs=20,batch_size=32)
    
    model.save('embedding.model')

with open('word.pkl','rb') as f:
    word = pickle.load(f)
# import sys
# sys.exit()
from numpy import dot
from numpy.linalg import norm


# print(pred[0].shape,Y[0].shape)

# def cos_sim(A, B):
# #   print(A.tolist(),B.tolist())
#     sim = dot(A, B)/(norm(A)*norm(B))
#     if sim < 0:
#         return 1 - (np.arccos(sim) / np.pi)
#     elif sim > 0:
#         # if sim > 1.0:
#         #     sim == 1
#         # print(sim)
#         if sim > 1.0:
#             sim = int(sim)
#         return 1 - ((2 * np.arccos(sim)) / np.pi)
#     else:
#         return 0
if False:
    xt = np.array([X[0]])
    pred = model.predict(xt)

    for i in range(72):
        print(cos_sim(Y[0][i],pred[0][i]))

model = tf.keras.models.load_model('embedding.model')

input = model.input[0]
# emb = model.layers[1].output
# print(type(emb))
output = model.layers[4].output

# input = model.layers[0]((maxlen))
# emb = model.layers[1](len(word.keys()),100)(input)
# print(type(emb))
# output = model.layers[2]()(emb)
# print(type(output))
# print(input)
# print(emb)
# emb_ = emb(input)
# output_ = output(emb_)

w1 = ['나', '밥','공원']
w2 = ['나', '국수', '햄버거',  '학교']
w3 = ['성폭행','남성', '경찰','고소']
w4 = ['경찰','조사','결과','이','남성','사고','당한','건','전날','밤']

xt = []
www = [w1,w2,w3,w4]
for _ in range(4):
    temp = []#[word['[START]']]
    ww = www[_]
    for w in ww:
        temp.append(word[w])
    temp = temp# + [word['[SEP]']]
    temp = temp + [0] * (maxlen - len(temp))
    xt.append(temp)

# xx = [word['[START]']] + xt[0] + [word['[SEP]']] + xt[1] + [word['[END]']]
# xx2 = [word['[START]']] + xt[1] + [word['[SEP]']] + xt[2] + [word['[END]']]

# xx = xx + [0] * (maxlen - len(xx))
# xx2 = xx2 + [0] * (maxlen - len(xx2))

X = [np.array([xt[0]]),np.array([xt[1]])]
print(X[0].shape)
# print(X)
re = model.predict(X)
print(re.tolist())


X = [np.array([xt[0]]),np.array([xt[2]])]
re = model.predict(X)
print(re.tolist())

X = [np.array([xt[0]]),np.array([xt[2]])]
re = model.predict(X)
print(re.tolist())

X = [np.array([xt[1]]),np.array([xt[3]])]
re = model.predict(X)
print(re.tolist())

X = [np.array([xt[0]]),np.array([xt[3]])]
re = model.predict(X)
print(re.tolist())

X = [np.array([xt[3]]),np.array([xt[2]])]
re = model.predict(X)
print(re.tolist())

X = [np.array([xt[1]]),np.array([xt[2]])]
re = model.predict(X)
print(re.tolist())

# import sys
# sys.exit()

model = tf.keras.models.Model(inputs=input,outputs=output)
print('model.summary')
model.summary()

l1 = 5 + 2
l2 = 5 + 2
l3 = 4 + 2


xt = []
www = [w1,w2,w3,w4]
for _ in range(4):
    temp = []#[word['[START]']]
    ww = www[_]
    for w in ww:
        temp.append(word[w])
    temp = temp# + [word['[SEP]']]
    temp = temp + [0] * (maxlen - len(temp))
    xt.append(temp)

pred = model.predict(xt)

# s0 = np.matmul(pred[2],np.transpose(pred[2],[1,0]))
# s1 = np.matmul(pred[1],np.transpose(pred[1],[1,0]))
# ss = np.zeros(maxlen)
# s = np.zeros(maxlen)

# for ss_ in s0:
#     ss += ss_
# for s_ in s1:
#     s += s_
# s0 = ss
# s1 = s
# print(cos_sim(pred[0],pred[1]))

# s0 = np.matmul(pred[0],np.transpose(pred[0],[1,0]))
# s1 = np.matmul(pred[1],np.transpose(pred[1],[1,0]))
# ss = np.zeros(maxlen)
# s = np.zeros(maxlen)

# for ss_ in s0:
#     ss += ss_
# for s_ in s1:
#     s += s_
# s0 = ss
# s1 = s
# print(cos_sim(pred[0],pred[2]))
# print(pred[0].shape)
pred1 = pred[0]#[:l1,:]
# print(pred1)
# pred_ = np.zeros(20)
# for p1 in pred1:
#     pred_ += p1
# pred1 = pred_ / l1

pred2 = pred[1]#[:l2]
# pred_ = np.zeros(20)
# for p2 in pred2:
#     pred_ += p2
# pred2 = pred_ / l2 

pred3 = pred[2]#[:l3]
# print(pred3.shape)
# pred_ = np.zeros(20)
# for p3 in pred3:
#     pred_ += p3
# pred3 = pred_ / l3
pred4 = pred[3]#[:l3]

print(w1,w2)
print(cos_sim(pred1,pred2))
print(w1,w3)
print(cos_sim(pred1,pred3))
print(w3,w2)
print(cos_sim(pred3,pred2))
print(w4,w2)
print(cos_sim(pred4,pred2))
print(w4,w1)
print(cos_sim(pred4,pred1))
print(w4,w3)
print(cos_sim(pred3,pred2))
# pred1 = pred1[:l1]
# print(pred1.shape)
