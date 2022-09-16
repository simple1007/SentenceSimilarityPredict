import pickle
import tensorflow as tf
import numpy as np
import sentencepiece as spm

from gensim.models import FastText
from numpy import dot
from numpy.linalg import norm
from utils.utils import cos_sim

from tensorflow.keras.layers import LSTM, Bidirectional, TimeDistributed, Dense, Input, Embedding, Lambda, Add, MaxPooling1D, GlobalAveragePooling1D

EPOCH = 7
VOCAB_LEN = 10000
BATCH = 32
maxlen = 250

def dataset():
    for _ in range(EPOCH):
        for i in range(1,2392):
            X = np.load(f'data/{i}_x.npy')
            X2 = np.load(f'data/{i}_x2.npy')
            Y = np.load(f'data/{i}_y.npy')

            yield [X,X2], Y

class ModelBuild:
    def __init__(self,maxlen):
        super(ModelBuild, self).__init__()
        self.maxlen = maxlen
    
    def call(self,inputs):
        input = Input((self.maxlen))
        input2 = Input((self.maxlen))
        print(inputs)
        emb = Embedding(VOCAB_LEN,32)(input)
        emb2 = Embedding(VOCAB_LEN, 32)(input2)

        bilstm = Bidirectional(LSTM(64,return_sequences=True))(emb)
        bilstm2 = Bidirectional(LSTM(64,return_sequences=True))(emb2)
        multiply = tf.multiply(bilstm,bilstm2)
        multiply = GlobalAveragePooling1D()(multiply)

        bilstm_ = Dense(32, activation = 'relu')(multiply)

        output = Dense(1)(bilstm_)

        # model = tf.keras.models.Model(inputs=[input, input2], outputs=output)

        return output

if False:
    mb = ModelBuild(maxlen)
    model = mb.getmodel()
    model.summary()

    model.compile(optimizer="adam",metrics=["mae"],loss="mse")

    ds = dataset()
    model.fit(ds, epochs=EPOCH, batch_size=BATCH, steps_per_epoch=1913,validation_steps=478)

    model.save('embedding.model')

    sp = spm.SentencePieceProcessor()
    vocab_file = 'ilbe_spm_model/ilbe_spm.model'
    sp.load(vocab_file)

# import sys
# sys.exit()

with open('word.pkl','rb') as f:
    word = pickle.load(f)

from numpy import dot
from numpy.linalg import norm

if False:
    xt = np.array([X[0]])
    pred = model.predict(xt)

    for i in range(72):
        print(cos_sim(Y[0][i],pred[0][i]))

model = tf.keras.models.load_model('embedding.model')

input = model.input[0]
emb = model.layers[0].output
output = model.layers[4].output

input2 = model.input[1]
output2 = model.layers[5].output

w1 = '삼성 전자'#'나는 밥을 먹고 학교에 갔다.'#['나', '밥', '먹다','바다','가다']
w2 = '이건희'#'나는 국수를 먹고 학교에 갔다.'  # ['나', '국수', '먹다', '웹툰','보다']
w3 = '웹툰 마음의 소리 재밌어.'#[]#['디지털', '포럼', '웹툰', '산업', '지속', '발전', '방안', '모색', '웹툰', '생태계', '대한', '이해', '위해', '세미나', '준비']
w4 = []#['웹툰']
w5 = []#['마음','소리']
xt = []
www = [w1,w2,w3,w4,w5]
# inputf = f'preprocessing/{origin_files}'
# spm_train(inputf)
sp = spm.SentencePieceProcessor()
vocab_file = 'ilbe_spm_model/ilbe_spm.model'
sp.load(vocab_file)
ll = []
for _ in range(3):
    temp = []#[word['[START]']]
    ww = www[_]
    temp = sp.encode_as_ids(ww)
    ll.append(len(temp))
    # for w in ww:
    #     temp.append(word[w])
    # temp = temp# + [word['[SEP]']]
    temp = temp + [0] * (maxlen - len(temp))
    xt.append(temp)
l1 = ll[0]# + 2
l2 = ll[1]#5  # + 2
l3 = ll[2]#4  # + 2
# l4 = 2
# l5 = 1
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

# X = [np.array([xt[3]]),np.array([xt[4]])]
# re = model.predict(X)
# print(re.tolist())

# import sys
# sys.exit()

model = tf.keras.models.Model(inputs=input,outputs=output)
print('model.summary')
model.summary()

model2 = tf.keras.models.Model(inputs=input2,outputs=output2)
print('model.summary2')
model2.summary()



xt = []
xt2 = []
www = [w1,w2,w3,w4,w5]
for _ in range(3):
    temp = []#[word['[START]']]
    ww = www[_]
    # for w in ww:
    #     temp.append(word[w])
    temp = sp.encode_as_ids(ww)
    # temp = temp# + [word['[SEP]']]
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
pred1 = pred[0]# + predd[0]
pred1 = pred1[:l1,:]
pred_ = np.zeros(128)
for i in pred1:
    pred_ += i
pred1 = pred_ / l1

pred2 = pred[1] #+ predd[1]
pred2 = pred2[:l2,:]
pred_ = np.zeros(128)
for i in pred2:
    pred_ += i
pred2 = pred_ / l2

pred3 = pred[2]# + predd[2]
pred3 = pred3[:l3,:]
pred_ = np.zeros(128)
for i in pred3:
    pred_ += i
pred3 = pred_ / l3

# pred4 = pred[3] + predd[3]
# pred4 = pred4[:l4,:]
# pred_ = np.zeros(128)
# for i in pred4:
#     pred_ += i
# pred4 = pred_ / l4

# pred5 = pred[4] + predd[4]
# pred5 = pred5[:l5,:]
# pred_ = np.zeros(128)
# for i in pred5:
#     pred_ += i
# pred5 = pred_ / l5
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
# print(w3,w2)
# print(cos_sim(pred3,pred2))
# print(w4,w5)
# print(cos_sim(pred4,pred5))
# model = Word2Vec.load('word2vec-KCC150/word2vec-KCC150.model')
# print(cos_sim((model.wv[w5[0]]+model.wv[w5[1]])/2,model.wv[w4[0]]))
# print(w3,w5)
# print(cos_sim(pred3,pred5))
# print(pred1,pred2,pred3)
# pred1 = pred1[:l1]
# print(pred1.shape)
