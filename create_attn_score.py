from ast import Lambda
import numpy as np
from gensim.models import Word2Vec

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

model = Word2Vec.load('word2vec-KCC150/word2vec-KCC150.model')
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
    # sim = sim * 100.0

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
if False:
    with open('preprocessing/morphs.txt',encoding='utf-8') as f:
        text = f.readlines()    
        random.shuffle(text)
        with open('preprocessing/compare1.txt','w',encoding='utf-8') as f:
            f.writelines(text)
    with open('preprocessing/morphs.txt',encoding='utf-8') as f:
        text = f.readlines()    
        random.shuffle(text)
        with open('preprocessing/compare2.txt','w',encoding='utf-8') as f:
            f.writelines(text)
    with open('preprocessing/morphs.txt',encoding='utf-8') as f:
        text = f.readlines()    
        # random.shuffle(text)
        with open('preprocessing/compare3.txt','w',encoding='utf-8') as f:
            f.writelines(text)
    for iiii in range(3):
        with open('preprocessing/morphs.txt',encoding='utf-8') as f:
            with open(f'preprocessing/compare{iiii+1}.txt',encoding='utf-8') as ff:
                for l,c in zip(f,ff):
                    l = l.strip()
                    c = c.strip()
                    
                    l = l.split(' ')
                    c = c.split(' ')

                    temp = []
                    x = []
                    for ll in l:
                        if ll in model.wv:
                            if ll not in word:
                                word[ll] = wordindex
                                wordindex += 1
                            x.append(word[ll])  
                            temp.append(model.wv[ll])
                    temp2 = []
                    x2 = []
                    for ll in c:
                        if ll in model.wv:
                            if ll not in word:
                                word[ll] = wordindex
                                wordindex += 1
                            x2.append(word[ll])  
                            temp2.append(model.wv[ll])
                    if len(temp) == 0 or len(temp2) == 0:
                        continue
                    length_ = len(x)
                    length2_ = len(x2)
                    # temp = temp + [[0]*300] * (maxlen-np.array(temp).shape[0])
                    # x = x + [word['[PAD]']] * (maxlen-len(x))
                    # temp_s = 
                    temp_ = np.array(temp)
                    temp2_ = np.array(temp2)
                    # print(temp_.shape)
                    # print(temp_.shape)
                    # print(np.transpose(temp_,[1,0]).shape)
                    # print(temp_.shape)
                    wordmap = {v:k for k,v in word.items()}
                    if temp_.shape[0] != 0:
                        
                        # v = np.zeros(300)
                        # for t2 in temp2:
                        #     v += t2
                        # v = v / len(temp2)
                        
                        # v2 = np.zeros(300)
                        # for t2 in temp:
                        #     v2 += t2
                        # v2 = v2 / len(temp)
                        
                        # print(1,cos_sim(v,v2))

                        temp___ = np.matmul(temp_,np.transpose(temp_,[1,0]))
                        temp___ = np.matmul(temp___,temp_)

                        temp2___ = np.matmul(temp2_,np.transpose(temp2_,[1,0]))
                        temp2___ = np.matmul(temp2___,temp2_)

                        vector = np.zeros(300)
                        for t2 in temp___:
                            vector += t2
                        vector = vector / length_

                        vector2 = np.zeros(300)
                        for t2 in temp2___:
                            vector2 += t2
                        vector2 = vector2 / length2_

                        # x = [word['[START]']] + x + [word['[SEP]']] + x2 + [word['[END]']]
                        y = cos_sim(vector,vector2)
                        # print(2,y)
                        # print(y)
                        # print(y)
                        # if cos_sim(vector,vector2) > 0.7:
                        #     print(l,c)
                        #     count+=1
                        #     print(cos_sim(vector,vector2))
                        # continue
                        # print(0,1,wordmap[x[0]],wordmap[x[1]],temp2[0][1])
                        # print(1,0,wordmap[x[1]],wordmap[x[0]],temp2[1][0])
                        # print(0,2,wordmap[x[0]],wordmap[x[2]],temp2[0][2])
                        # print(2,0,wordmap[x[2]],wordmap[x[0]],temp2[2][0])
                        # print(2,3,wordmap[x[2]],wordmap[x[4]],temp2[2][4])
                        # print(3,2,wordmap[x[4]],wordmap[x[2]],temp2[4][2])
                        # vector = np.zeros(300)
                        # for t2 in temp2:
                        #     vector += t2
                        # vector = vector / length_
                        # print(vector)
                        # print(vector.shape)
                        # temp2 = np.matmul(temp2,np.reshape(temp_,(300,1))
                        # print(temp2.shape)

                        # import sys
                        # sys.exit()
                        # print(temp2[0])
                        # print(temp2[1])
                        # print(temp2[2])
                        # temp__ = []
                        # for t2 in temp2:
                        #     t2 = t2 / np.sqrt(300)
                        #     t2 = softmax(t2)
                        #     # print(t2.shape)
                        #     # import sys
                        #     # sys.exit()
                        # #     # print(t2)
                        # #     t2 = t2.sum() / maxlen
                        #     temp__.append(t2)
                        # #     # print(t2)
                        
                        # temp2 = np.array(temp__)
                        # print(softmax(temp2))
                        # print(temp2[0])
                        # print(temp2[1])
                        # print(temp2[2])
                        # import sys
                        # sys.exit()
                        # temp2 = np.matmul(temp2,temp_)
                        # print(temp2)
                        # print(temp2.shape)

                        # import sys
                        # sys.exit()
                        # temp2 = softmax(temp2)
                        # temp2 = temp2.flatten()
                        # print(temp2.shape)
                        # temp_ = np.matmul(temp,temp_)
                        x = x + [0] * (maxlen - len(x))
                        x2 = x2 + [0] * (maxlen - len(x2))
                        X.append(x)
                        X2.append(x2)
                        Y.append(y)
                        # print(temp2.shape)
                        # print(temp_.shape)
                        # print(temp.tolist())
                        # length.append(temp_.shape[0])
                        # x.write(str(temp_.tolist()).replace('\n','')+'\n')
                        # y.write(str(temp.tolist()).replace('\n','')+'\n')
                        # print(temp_.tolist())
                        # break
    with open('word.pkl','wb') as f:
        pickle.dump(word,f)
# print(max(length))
# x.close()
# y.close()
# print(count)
# import sys
# sys.exit()

import tensorflow as tf
from tensorflow.keras.layers import LSTM, Bidirectional, TimeDistributed, Dense, Input, Embedding, Lambda, Add, MaxPooling1D, GlobalAveragePooling1D
# tf.keras.layers.Input((,300))
# print()
# tf.keras.preprocessing.text.Tokenizer
def dotp(x):
    return tf.matmul(x,x,transpose_b=False)
input = Input((maxlen))
input2 = Input((maxlen))
emb = Embedding(len(word.keys()),32)(input)
emb2 = Embedding(len(word.keys()),32)(input2)
# lstm = Bidirectional(LSTM(200,return_sequences=False))(emb)
bilstm = Bidirectional(LSTM(64,return_sequences=True))(emb)
bilstm2 = Bidirectional(LSTM(64,return_sequences=True))(emb2)
multiply = tf.multiply(bilstm,bilstm2)
multiply = GlobalAveragePooling1D()(multiply)

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

random_ = np.arange(X.shape[0])

X = X[random_]
X2 = X2[random_]
Y = Y[random_]

if False:
    print(X.shape,Y.shape,X[0],Y[1])
    model.fit([X,X2],Y,epochs=7,batch_size=32)
    
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
emb = model.layers[0].output
# print(type(emb))
output = model.layers[4].output
# output = GlobalAveragePooling1D()(output)

input2 = model.input[1]
output2 = model.layers[5].output
# output2 = model.layers[5].output
# input = model.layers[0]((maxlen))
# emb = model.layers[1](len(word.keys()),100)(input)
# print(type(emb))
# output = model.layers[2]()(emb)
# print(type(output))
# print(input)
# print(emb)
# emb_ = emb(input)
# output_ = output(emb_)

w1 = ['나', '밥', '먹다','바다','가다']
w2 = ['나', '국수', '먹다', '웹툰','보다']
w3 = ['디지털', '포럼', '웹툰', '산업', '지속', '발전', '방안', '모색', '웹툰', '생태계', '대한', '이해', '위해', '세미나', '준비']
w4 = ['삼성','전자']
w5 = ['김정은']
xt = []
www = [w1,w2,w3]
for _ in range(3):
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

# import sys
# sys.exit()

model = tf.keras.models.Model(inputs=input,outputs=output)
print('model.summary')
model.summary()

model2 = tf.keras.models.Model(inputs=input2,outputs=output2)
print('model.summary2')
model2.summary()

l1 = 5 #+ 2
l2 = 5 #+ 2
l3 = 4 #+ 2
l4 = 2
l5 = 1

xt = []
xt2 = []
www = [w1,w2,w3,w4,w5]
for _ in range(5):
    temp = []#[word['[START]']]
    ww = www[_]
    for w in ww:
        temp.append(word[w])
    temp = temp# + [word['[SEP]']]
    temp = temp + [0] * (maxlen - len(temp))
    xt.append(temp)
    xt2.append(temp)

xt = np.array(xt)
xt2 = np.array(xt2)
print(xt.shape,xt2.shape)
pred = model(xt)
predd = model2(xt)
temp = []
print(np.array(pred).shape)
# for pr,pr_ in zip(pred[0],pred[1]):
#     # t = np.zeros(128)
#     # for pr_ in pr:
#     #     t += pr_
#     t = pr + pr_
#     temp.append(t)
#     # print(temp)
# pred = np.array(temp)
print(pred.shape)
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
pred1 = pred[0] + predd[0]
pred1 = pred1[:l1,:]
pred_ = np.zeros(128)
for i in pred1:
    pred_ += i
pred1 = pred_ / l1

pred2 = pred[1] + predd[1]
pred2 = pred2[:l2,:]
pred_ = np.zeros(128)
for i in pred2:
    pred_ += i
pred2 = pred_ / l1

pred3 = pred[2] + predd[2]
pred3 = pred3[:l3,:]
pred_ = np.zeros(128)
for i in pred3:
    pred_ += i
pred3 = pred_ / l1

pred4 = pred[3] + predd[3]
pred4 = pred4[:l4,:]
pred_ = np.zeros(128)
for i in pred4:
    pred_ += i
pred4 = pred_ / l4

pred5 = pred[4] + predd[4]
pred5 = pred5[:l5,:]
pred_ = np.zeros(128)
for i in pred5:
    pred_ += i
pred5 = pred_ / l5
# print(pred1)
# pred_ = np.zeros(128)
# for i in range(l1):
#     p1 = pred1[i]
#     pred_ += p1
# print(pred_.shape)
# import sys
# sys.exit()
# pred1 = pred_ / l1

# pred2 = pred[1] + predd[1]#[:l2]
# pred_ = np.zeros(128)
# for i in range(l2):
#     pred_ += p2
# pred2 = pred_ / l2 

# pred3 = pred[2] + predd[2]#[:l3]
# print(pred3.shape)
# pred_ = np.zeros(128)
# for p3 in range(l3):
#     pred_ += p3
# pred3 = pred_ / l3

print(w1,w2)
print(cos_sim(pred1,pred2))
print(w1,w3)
print(cos_sim(pred1,pred3))
print(w3,w2)
print(cos_sim(pred3,pred2))
print(w4,w5)
print(cos_sim(pred4,pred5))
model = Word2Vec.load('word2vec-KCC150/word2vec-KCC150.model')
print(cos_sim((model.wv[w4[0]]+model.wv[w4[1]])/2,model.wv[w5[0]]))
print(w3,w5)
print(cos_sim(pred3,pred5))
# print(pred1,pred2,pred3)
# pred1 = pred1[:l1]
# print(pred1.shape)
