from utils.utils import cos_sim

import tensorflow as tf
import numpy as np
import sentencepiece as spm

maxlen = 300

sp = spm.SentencePieceProcessor()
vocab_file = 'ilbe_ht_spm_model/ilbe_spm.model'
sp.load(vocab_file)

gender = np.load('emb/gen.npy')
society = np.load('emb/soc.npy')

model = tf.keras.models.load_model('emb/embedding_model')

while True:
    x = input("input sentence: ")
    if x.lower() == 'exit':
        break
    x = sp.encode_as_ids(x)
    x = x + [0] * (maxlen - len(x))
    x = x[:maxlen]

    pred = model(np.array([x]))

    print("젠더 혐오:",cos_sim(pred[0],gender),"정치 혐오:",cos_sim(pred[0],society))