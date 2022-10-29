from cmath import log10
import pickle
from random import seed
import tensorflow as tf
import numpy as np
import sentencepiece as spm

# from gensim.models import FastText
from numpy import dot
from numpy.linalg import norm
from utils.utils import cos_sim

from tensorflow.keras.layers import LSTM, Bidirectional, TimeDistributed, Dense, Input, Embedding, Lambda, Add, MaxPooling1D, GlobalAveragePooling1D

EPOCH = 50
VOCAB_LEN = 10000
BATCH = 32
maxlen = 300

def dataset():
    for _ in range(EPOCH):
        for i in range(1,11999):
            X = np.load(f'data/{i}_x.npy')
            X2 = np.load(f'data/{i}_x2.npy')
            Y = np.load(f'data/{i}_y.npy')

            yield [X,X2], Y

def dataset_ht():
    for _ in range(EPOCH):
        for i in range(1,6795):
            X = np.load('ht/%05d_x.npy'%i)
            # X2 = np.load(f'data/{i}_x2.npy')
            Y = np.load('ht/%05d_y.npy'%i)

            yield X, Y


class ModelBuild(tf.keras.Model):
    def __init__(self,maxlen):
        super(ModelBuild, self).__init__()
        self.maxlen = maxlen
        
        self.emb = Embedding(VOCAB_LEN,32)#(input)
        # self.emb2 = Embedding(VOCAB_LEN, 32)#(input2)

        self.bilstm = Bidirectional(LSTM(64,return_sequences=True))#(emb)
        # self.bilstm2 = Bidirectional(LSTM(64))#(emb2)
        #self.multiply = tf.multiply(bilstm,bilstm2)
        self.avg = GlobalAveragePooling1D()#(multiply)

        self.dense = Dense(32, activation = 'relu')#(multiply)

        self.result1 = Dense(1,name='1')#(bilstm_)
        self.result2 = Dense(1,name='2')
        # self.result3 = Dense(1,name='3')
        # self.result4 = Dense(1,name='4')

    def build(self, input_shape):
        super(ModelBuild, self).build(input_shape)
        # self.input_ = Input(input_shape[0])
        # self.input2_ = Input(input_shape[1])
        # print(inputs)
    def get_config(self):
        config = {'maxlen': self.maxlen,}
        base_config = super(ModelBuild, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
    
    @classmethod
    def from_config(cls, config):
        return cls(**config)

    def call(self,inputs):
        # print(inputs)
        x = self.emb(inputs)
        x = self.bilstm(x)
        # x = self.avg(x)
        # x2 = self.emb2(inputs[1])
        # x2 = self.bilstm2(x2)

        # multiply = tf.concat([x,x2],-1)
        
        a = self.avg(x)
        a = self.dense(x)

        return self.result1(a)#,self.result2(a),self.result3(a),self.result4(a)]
        # model = tf.keras.models.Model(inputs=[input, input2], outputs=output)

        # return output

if False:
    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

    callbacks = [EarlyStopping(monitor='val_loss',patience=3), ModelCheckpoint("hatespeech_sentence_embedding.model",monitor="val_loss",save_best_only=True)]
    # callbacks = [ModelCheckpoint("hatespeech_sentence_embedding.model",monitor="val_loss",save_best_only=True)]

    model = ModelBuild(maxlen)
    # model = mb.getmodel()
    # model.build([(None,maxlen),(None,maxlen)])
    model.build((None,maxlen))
    model.summary()

    model.compile(optimizer="adam",metrics=["mae"],loss="mse")

    ds = dataset_ht()
    model.fit(ds, epochs=EPOCH, batch_size=BATCH, steps_per_epoch=5436,validation_data=ds,validation_steps=1358, callbacks=callbacks)

    model.save('embedding.model')

    sp = spm.SentencePieceProcessor()
    vocab_file = 'ilbe_spm_model/ilbe_spm.model'
    sp.load(vocab_file)
    import sys
    sys.exit()
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

model = tf.keras.models.load_model("hatespeech_sentence_embedding.model")#,custom_objects={'ModelBuild':ModelBuild})


input = Input((maxlen))
emb = model.emb(input)
# input = model.layers[1].input
# emb = model.layers[0].output
output = model.bilstm(emb)
# input2 = Input((maxlen))
# emb2 = model.emb2(input2)
# output2 = model.bilstm2(emb2)

# w1 = '촛불 태극기 집회를 열자!'# 너 김정은 놈'#'나는 밥을 먹고 학교에 갔다.'#['나', '밥', '먹다','바다','가다']
# w2 = '서민실수요자는 부부합산 연소득 6천만 원, 주택가격 5억 원이하, 무주택세대주 등이 포함된다.'# 개 놈'#'당신 김정은이야?'#'나는 국수를 먹고 학교에 갔다.'  # ['나', '국수', '먹다', '웹툰','보다']
# w3 = '나는 밥을 먹고 마트에 갔다.'#'이게 언론조작이란거다 병신좌좀새끼들아'#'웹툰 마음의 소리 재밌어.'#[]#['디지털', '포럼', '웹툰', '산업', '지속', '발전', '방안', '모색', '웹툰', '생태계', '대한', '이해', '위해', '세미나', '준비']
# w4 = '빨갱이 빨갱이 김정일 놈'#['웹툰']
# w5 = '김정은 김정은 인민위원장'#['마음','소리']
# w6 = '좌빨 빨갱이 좌빨 가족 랍시고 데리 살기 애초 거르는 이득 임'
# w7 = '김정일 김정일 좌파 뇌교육 그 빛 발 개돼지 민주당 정도 나라 민주당 이나 원래 나라 각하 비정상 인물 셧 미얀마 수준 '
# w8 = '북한 김정은 길 주사파 남한 정치인 주체조선 자꾸 종용 김정은'
# w9 = '공산주의 김 익주 교수 탈 원전 필수 좌빨 아마 김일성 장학생 출신 의사'
# w10 = '사회주의 진짜 공산주의 독재 진중권 좌빨 실체 정체 알 소리 얼마나 좌빨 짱개 바퀴벌레 왜케 기어 분탕질'
# w11 = '간첩 중국 간첩 빨갱이 간 여 간첩 새끼'
# w12 = '좌좀 가치 공유 븅신 너 사회주의 빨 좌빨'

# w4 = '핵대중개새끼 파내서 자지를 잘라야돼'#['웹툰']
# w5 = '그 국민 중 큰 하나가 문재앙이잖어이 새낀 기독교인인척하면서 공산주의빨면 대놓고 가짜라는것은데눈먼사람들 이 간단한것도 안 보임'#['마음','소리']
# w6 = '개대중이 끝나고 노무현 좌파연임. 정동영같은새끼 나오지않는이상 또 반복될느낌'
# w7 = '아베통수라고 해서 들어왔더니난 뭐과거 제국주의일본이 한국과 아시아 많은 나라에게 아픔을 주고 전쟁을 일으켜 원폭을 사용했지만 희생자들에게는 애도의 마음을 전한다. 라고 말한줄 알았네.'
# w8 = '극기 불태운거랑 비슷한 넘, 광우병 걸린다며 미국소는 먹지 말자던 등신들, 증거도 없는데 박정희어르신 친일혈서 썼다고 거짓>말하는 안아무인 철면피들, 또는 김일성이 일본 혼내준적 있다고 미친소리하는 것들'
# w9 = '욱일기보다 빨갱이들이 더악질이다'
# w10 = '이래서 좌좀은 벌레야. 사내유보금이 얼마가 늘든 기업의 미래를 위해 반드시 보존 해야 될 돈인데 좌좀 병신들은 돈이 남으면 >무조건 나눠야 된다고 생각하는 병신들이 적금은 왜 넣나몰라.'
# w11 = '아 또 빨갱이 나오나연?!'
# w12 = '저씨발 빨갱이 새끼는 모자이크해서 올려라'
# w5 = ['']
# xt = []
# www = [w1,w2,w3,w4,w5,w6,w7,w8,w9,w10,w11,w12]
# inputf = f'preprocessing/{origin_files}'
# spm_train(inputf)
model = tf.keras.models.Model(inputs=input,outputs=output)
model.summary()
sp = spm.SentencePieceProcessor()
vocab_file = 'ilbe_ht_spm_model/ilbe_spm.model'
sp.load(vocab_file)
ll = []
xt = []
www = ['페미들 극혐 버려지들','친구','꼴페미 김정은 빨갱이들','나 밥 학교']
f = open('dictionary_gen.txt',encoding='utf-8')
gen_dic_x = []
gen_cnt = 0

from konlpy.tag import Okt
o = Okt()
length = []
for l in f:
    if l.startswith('//'):
        continue
    l = l.replace('#','')
    temp = []
    # l = o.nouns(l)
    # l = ' '.join(l)
    # l = ['틀딱들', '급식충', '사이']
    # l = ' '.join(l)
    temp = sp.encode_as_ids(l.strip())
    length.append(len(temp))
    temp = temp + [0] * (maxlen - len(temp))
    gen_cnt += 1
    gen_dic_x.append(temp)

temp_gen = model(np.array(gen_dic_x))

gen_dic = np.zeros(128)
# print(temp_gen[0].shape)
for index, tg in enumerate(temp_gen):
    lenn = length[index]
    # print(tg.shape)
    predd = tg[:lenn,:]
    print(predd.shape)
    predd = np.sum(predd,axis=0)
    print(predd.shape)
    gen_dic += predd / lenn

gen_dic = gen_dic / gen_cnt
np.save('emb/gen',gen_dic)
f.close()

f = open('dictionary_soc.txt',encoding='utf-8')
soc_dic_x = []
soc_cnt = 0
length = []
for l in f:
    if l.startswith('//'):
        continue
    l = l.replace('#','')
    temp = []
    # l = o.nouns(l)
    # l = ' '.join(l)
    # l = ['틀딱들', '급식충', '사이']
    # l = ' '.join(l)
    temp = sp.encode_as_ids(l.strip())
    length.append(len(temp))
    temp = temp + [0] * (maxlen - len(temp))
    soc_cnt += 1
    soc_dic_x.append(temp)

temp_soc = model(np.array(soc_dic_x))

soc_dic = np.zeros(128)
for index, tg in enumerate(temp_soc):
    lenn = length[index]
    predd = tg[:lenn,:]
    predd = np.sum(predd,axis=0)
    print(predd.shape)
    soc_dic += predd / lenn

soc_dic = soc_dic / soc_cnt
np.save('emb/soc',soc_dic)
f.close()

# f = open('dictionary_age.txt',encoding='utf-8')
# age_dic_x = []
# age_cnt = 0
# for l in f:
#     if l.startswith('//'):
#         continue
#     l = l.replace('#','')
#     temp = []
#     temp = sp.encode_as_ids(l.strip())
#     temp = temp + [0] * (maxlen - len(temp))
#     age_cnt += 1
#     age_dic_x.append(temp)

# temp_age = model(np.array(age_dic_x))

# age_dic = np.zeros(128)
# for tg in temp_age:
#     age_dic += tg

# age_dic = age_dic / age_cnt
# np.save('emb/age',age_dic)
# f.close()

# f = open('dictionary_location.txt',encoding='utf-8')
# location_dic_x = []
# location_cnt = 0
# for l in f:
#     if l.startswith('//'):
#         continue
#     l = l.replace('#','')
#     temp = []
#     temp = sp.encode_as_ids(l.strip())
#     temp = temp + [0] * (maxlen - len(temp))
#     location_cnt += 1
#     location_dic_x.append(temp)

# temp_location = model(np.array(location_dic_x))

# location_dic = np.zeros(128)
# for tg in temp_location:
#     location_dic += tg

# location_dic = location_dic / location_cnt
# np.save('emb/location',location_dic)
# f.close()

# for _ in range(4):
#     temp = []#[word['[START]']]
#     ww = www[_]
#     # ww = '페미들 극혐 버려지들'
#     temp = sp.encode_as_ids(ww)
#     print(sp.encode_as_pieces(ww))
#     ll.append(len(temp))
#     # for w in ww:
#     #     temp.append(word[w])
#     # temp = temp# + [word['[SEP]']]
#     temp = temp + [0] * (maxlen - len(temp))
#     xt.append(temp)
# l1 = ll[0]# + 2
# l2 = ll[1]#5  # + 2
# l3 = ll[2]#4  # + 2

# # l4 = ll[3]
# # l5 = ll[4]
# # l6 = ll[5]
# # l7 = ll[6]
# # l8 = ll[7]
# # l9 = ll[8]
# # l10 = ll[9]
# # l11 = ll[10]
# # l12 = ll[11]
# # l4 = 2
# # l5 = 1
# # xx = [word['[START]']] + xt[0] + [word['[SEP]']] + xt[1] + [word['[END]']]
# # xx2 = [word['[START]']] + xt[1] + [word['[SEP]']] + xt[2] + [word['[END]']]

# # xx = xx + [0] * (maxlen - len(xx))
# # xx2 = xx2 + [0] * (maxlen - len(xx2))

# # X = [np.array([xt[0]]),np.array([xt[1]])]
# # print(X[0].shape)
# # # print(X)
# # re = model.predict(X)
# # print(re.tolist())


# # X = [np.array([xt[0]]),np.array([xt[2]])]
# # re = model.predict(X)
# # print(re.tolist())

# # X = [np.array([xt[3]]),np.array([xt[4]])]
# # re = model.predict(X)
# # print(re.tolist())

# # import sys
# # sys.exit()

# # pred = model(np.array(xt))
# # print(pred)


# print('model.summary')
# model.summary()
# pr = model(np.array(xt))
# pred1 = pr[0]
# pred2 = pr[1]#(pr[1] + pr[2]) / 2
# pred3 = pr[2]
# pred4 = pr[3]
# # seed = np.load('gen_seed.npy')
model.save('emb/embedding_model')

# print(www[0])
# print('성:',cos_sim(pred1,gen_dic),'정치:',cos_sim(pred1,soc_dic),'연령:',cos_sim(pred1,age_dic),'지역:',cos_sim(pred1,location_dic))
# print(www[1])
# print('성:',cos_sim(pred2,gen_dic),'정치:',cos_sim(pred2,soc_dic),'연령:',cos_sim(pred2,age_dic),'지역:',cos_sim(pred2,location_dic))
# print(www[2])
# print('성:',cos_sim(pred3,gen_dic),'정치:',cos_sim(pred3,soc_dic),'연령:',cos_sim(pred3,age_dic),'지역:',cos_sim(pred3,location_dic))
# print(www[3])
# print('성:',cos_sim(pred4,gen_dic),'정치:',cos_sim(pred4,soc_dic),'연령:',cos_sim(pred4,age_dic),'지역:',cos_sim(pred4,location_dic))

# model = tf.keras.models.load_model('embedding.model')
# pred = model(np.array(xt))
# print(pred)
# model2 = tf.keras.models.Model(inputs=input2,outputs=output2)
# print('model.summary2')
# model2.summary()
import sys
sys.exit()
xt = []
xt2 = []
www = [w1,w2,w3,w4,w5,w6,w7,w8,w9,w10,w11,w12]
for _ in range(12):
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
pred1 = pred[0] #+ predd[0]
# pred1 = np.sum(pred1,axis=0)# / maxlen
print("pred1",pred1)
# pred1 = pred1[:l1,:]
# pred_ = np.zeros(128)
# for i in pred1:
#     pred_ += i
# pred1 = pred_ / l1

pred2 = pred[1] #+ predd[1]
# pred2 = np.sum(pred2,axis=0)# / maxlen
# pred2 = pred2[:l2,:]
# pred_ = np.zeros(128)
# for i in pred2:
#     pred_ += i
# pred2 = pred_ / l2

pred3 = pred[2] #+ predd[2]
# pred3 = np.sum(pred3,axis=0)# / maxlen
# pred3 = pred3[:l3,:]
# pred_ = np.zeros(128)
# for i in pred3:
#     pred_ += i
# pred3 = pred_ / l3

seed_ = np.zeros(128)
for i in range(3,12):
    pred__ = pred[i]
    # pred__ = np.sum(pred__,axis=0)
    # pred2__ = pred[i]
    # length = ll[i]
    # pred__ = pred__[:length,:]
    # pred2__ = pred__[:length,:]

    # # pred__ = pred__ + pred2__
    # pred_ = np.zeros(128)
    # for p in pred__:
    #     pred_ += p / length
    # # pred_ = pred_
    # # pred_ = pred_ / length
    seed_ += pred__ #/ maxlen

seed_ = seed_ / sum(ll[3:12])
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

print(w1,'seed')
print(cos_sim(pred1,seed_))
print(w3,'seed')
print(cos_sim(pred3,seed_))
print(w2,w3)
print(cos_sim(pred2,pred3))
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
